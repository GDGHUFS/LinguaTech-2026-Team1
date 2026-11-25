"""Owned two-GPU tensor-parallel vLLM process with its own CUDA context."""
import importlib.metadata
import multiprocessing as mp
import os
import signal
import traceback

from .contracts import Blocked


def visible_device(index):
    """Resolve a logical index without losing a scheduler's CUDA visibility mask."""
    mask = os.environ.get('CUDA_VISIBLE_DEVICES')
    return str(index) if mask is None else mask.split(',')[index].strip()


def _serve(connection, target, devices, eos_ids, seed):
    # This module deliberately imports neither torch nor vLLM before isolation.
    os.environ['CUDA_VISIBLE_DEVICES'] = devices
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    if os.name == 'posix':
        os.setsid()
    engine = None
    try:
        import torch
        from vllm import LLM, SamplingParams
        settings = target['vllm']
        engine = LLM(
            model=target['model_id'], revision=target['revision'],
            tokenizer_revision=target['revision'], trust_remote_code=False,
            dtype='bfloat16', seed=seed,
            tensor_parallel_size=settings['tensor_parallel_size'],
            distributed_executor_backend='mp',
            max_model_len=target['max_context_tokens'],
            gpu_memory_utilization=settings['gpu_memory_utilization'],
            enable_prefix_caching=True, enable_chunked_prefill=True,
            max_num_batched_tokens=settings['max_num_batched_tokens'],
            max_num_seqs=1, generation_config='vllm',
            enforce_eager=settings['enforce_eager'])
        params = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1,
                                repetition_penalty=1.0, seed=seed,
                                max_tokens=target['max_new_tokens'],
                                stop_token_ids=eos_ids, detokenize=False)
        connection.send(('ready', {
            'versions': {name: importlib.metadata.version(name)
                         for name in ('vllm', 'torch', 'transformers', 'safetensors')},
            'tensor_parallel_size': settings['tensor_parallel_size'],
            'device_names': [torch.cuda.get_device_name(i)
                             for i in range(settings['tensor_parallel_size'])]}))
        while True:
            ids = connection.recv()
            if ids is None:
                break
            try:
                result = engine.generate([{'prompt_token_ids': ids}], params,
                                         use_tqdm=False)[0].outputs[0]
                if result.finish_reason not in ('stop', 'length'):
                    raise RuntimeError(f'Unexpected vLLM finish reason: {result.finish_reason}')
                connection.send(('result', (list(result.token_ids), result.finish_reason)))
            except Exception:
                connection.send(('error', traceback.format_exc()))
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        connection.send(('error', traceback.format_exc()))
    finally:
        # V1's engine core owns additional processes; release them before exiting.
        core = getattr(getattr(engine, 'llm_engine', None), 'engine_core', None)
        if core is not None:
            core.shutdown()
        connection.close()


class VllmWorker:
    def __init__(self, target, eos_ids, seed):
        settings = target['vllm']
        self.timeout = settings['request_timeout_seconds']
        self.closed = False
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_serve,
            args=(child, target, ','.join(visible_device(i) for i in settings['generation_gpus']), eos_ids, seed),
            name='conshift-vllm')
        self.process.start()
        child.close()
        try:
            self.runtime = self._receive('ready', settings['startup_timeout_seconds'])
        except BaseException:
            self.close()
            raise

    def _receive(self, expected, timeout):
        if not self.connection.poll(timeout):
            self.close()
            raise Blocked(f'vLLM {expected} timed out after {timeout}s; worker stopped')
        try:
            kind, payload = self.connection.recv()
        except (EOFError, OSError) as exc:
            raise Blocked('vLLM worker exited unexpectedly') from exc
        if kind != expected:
            if expected == 'result' and kind == 'error':
                raise RuntimeError(f'vLLM generation failed: {payload}')
            raise Blocked(f'vLLM worker {kind}: {payload}')
        return payload

    def generate(self, ids):
        if self.closed or not self.process.is_alive():
            raise Blocked('vLLM worker is not running; restart collection with --resume')
        try:
            self.connection.send(ids)
            return self._receive('result', self.timeout)
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise Blocked('vLLM worker connection failed') from exc

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.is_alive():
            try:
                self.connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.process.join(timeout=10)
            if self.process.is_alive():
                if os.name == 'posix' and os.getpgid(self.process.pid) == self.process.pid:
                    os.killpg(self.process.pid, signal.SIGTERM)
                else:
                    self.process.terminate()
                self.process.join(timeout=10)
            if self.process.is_alive():
                if os.name == 'posix' and os.getpgid(self.process.pid) == self.process.pid:
                    os.killpg(self.process.pid, signal.SIGKILL)
                else:
                    self.process.kill()
                self.process.join(timeout=10)
        self.connection.close()
