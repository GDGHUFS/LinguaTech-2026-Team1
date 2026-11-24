import torch
from torch import nn


class P1Probe(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.head = nn.Linear(hidden_dim, 1)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, u_t, delta, distance):
        return self.head(u_t).squeeze(-1)


class T1Probe(P1Probe):
    def __init__(self, hidden_dim):
        super().__init__(2 * hidden_dim)

    def forward(self, u_t, delta, distance):
        return self.head(torch.cat([u_t, delta], dim=-1)).squeeze(-1)


class ConShiftProbe(T1Probe):
    def __init__(self, hidden_dim):
        super().__init__(hidden_dim)
        self.gate_a = nn.Parameter(torch.zeros(1))
        self.gate_c = nn.Parameter(torch.zeros(1))

    def gate_value(self, distance):
        return torch.sigmoid(self.gate_a * distance + self.gate_c)

    def forward(self, u_t, delta, distance, force_gate=None):
        gate = self.gate_value(distance) if force_gate is None else force_gate
        return self.head(torch.cat([u_t, gate * delta], dim=-1)).squeeze(-1)


PROBES = {'p1': P1Probe, 't1': T1Probe, 'conshift': ConShiftProbe}
