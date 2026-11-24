"""Load pinned official deterministic checkers; never invoke a judge or fallback."""
import builtins
import importlib.util
import sys
import types
from pathlib import Path
from .evolif_adapter import JUDGE

MAPPING = {
    'case': ('change_case', 'ChangeCaseInstruction'),
    'countableItems': ('countable_items', 'CountableItemsInstruction'),
    'endwith': ('end_with', 'EndWithInstruction'),
    'existence': ('existence', 'ExistenceInstruction'),
    'forbidden': ('forbidden', 'ForbiddenInstruction'),
    'format': ('format_instruction', 'FormatInstruction'),
    'length': ('length', 'LengthInstruction'),
    'punctuation': ('punctuation', 'PunctuationInstruction'),
    'startwith': ('start_with', 'StartWithInstruction'),
}


def args_valid(i, a):
    if i == 'forbidden':
        return isinstance(a, list) and bool(a) and all(isinstance(x, str) and x for x in a)
    if not isinstance(a, dict) or not a:
        return False
    if i == 'existence':
        return all(isinstance(k, str) and bool(k) and type(v) is int and v > 0 for k, v in a.items())
    if i == 'countableItems':
        return type(a.get('num')) is int and a['num'] >= 0
    if i == 'length':
        return (a.get('mode') in {'word', 'paragraph', 'characters', 'sentence'}
                and a.get('relation') in {'less_than', 'more_than', 'exactly'}
                and type(a.get('number')) is int and a['number'] >= 0)
    if i == 'case':
        return a.get('mode') in {'all_upper', 'all_lower'} or (a.get('mode') == 'min_upper'
                and type(a.get('min')) is int and 0 <= a['min'] <= 100)
    if i == 'format':
        return a.get('mode') in {'json', 'xml', 'html', 'csv', 'markdown'}
    if i == 'punctuation':
        return a.get('mode') in {'must_include', 'must_not_include'} and isinstance(a.get('value'), str) and bool(a['value'])
    if i in {'startwith', 'endwith'}:
        if a.get('mode') == 'quotation':
            return all(isinstance(a.get(k), str) and a[k] for k in ('left', 'right'))
        return a.get('mode') in {'letter', 'keyword', 'emoji'} and isinstance(a.get('value'), str) and bool(a['value'])
    return False


class OfficialChecker:
    """Single-process adapter. Source files remain byte-identical to the lock."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.namespace = '_conshift_evolif_' + str(abs(hash(str(self.root))))
        package = types.ModuleType(self.namespace)
        package.__path__ = [str(self.root / 'src/instruction')]
        sys.modules[self.namespace] = package
        self.classes = {}
        import nltk
        resource_path = str(self.root.parent / 'nltk_data')
        if resource_path not in nltk.data.path:
            nltk.data.path.insert(0, resource_path)
        helper = self.load('instruction_utils')
        self.helper = helper
        for i, (module, cls) in MAPPING.items():
            self.classes[i] = getattr(self.load(module), cls)

    def load(self, module):
        name = self.namespace + '.' + module
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, self.root / 'src/instruction' / (module + '.py'))
        obj = importlib.util.module_from_spec(spec)
        # Preserve official source bytes while resolving its implicit text I/O
        # against the pinned code root and UTF-8 on Windows as on Linux.
        def source_open(file, mode='r', *args, **kwargs):
            p = Path(file)
            if not p.is_absolute():
                p = self.root / p
            if 'b' not in mode and not args and 'encoding' not in kwargs:
                kwargs['encoding'] = 'utf-8'
            return builtins.open(p, mode, *args, **kwargs)
        obj.open = source_open
        sys.modules[name] = obj
        try:
            spec.loader.exec_module(obj)
        except Exception:
            sys.modules.pop(name, None)
            raise
        return obj

    def evaluate(self, constraint, response):
        i, args = constraint['instruction_id'], constraint['args']
        result = {'instruction_id': i, 'result': 'UNKNOWN', 'raw': None, 'error': None}
        if i in JUDGE:
            result['error'] = 'LLM_JUDGE_REQUIRED'
            return result
        if i not in self.classes or not args_valid(i, args):
            result['error'] = 'UNSUPPORTED_ID_OR_INVALID_ARGS'
            return result
        old_helper = sys.modules.get('instruction_utils')
        try:
            # Start/end import this exact module by its original top-level name.
            sys.modules['instruction_utils'] = self.helper
            if i == 'length' and args['mode'] == 'sentence':
                self.helper.count_sentences(response)  # propagate missing Punkt, no official fallback
            obj = self.classes[i]()
            obj.args = args
            raw = obj.check_following(response)
            if type(raw) is not bool:
                raise TypeError('Official deterministic checker did not return bool')
            result.update(raw=raw, result='PASS' if raw else 'FAIL')
        except Exception as exc:
            result['error'] = type(exc).__name__ + ': ' + str(exc)
        finally:
            if old_helper is None:
                sys.modules.pop('instruction_utils', None)
            else:
                sys.modules['instruction_utils'] = old_helper
        return result
