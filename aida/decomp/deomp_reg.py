import torch
import torch.nn as nn
import numpy as np
import torch.distributions as dist

class Decomposer(nn.Module):

    def __init__(self, vars, n_inter):
        super().__init__()
        self.vars = vars
        self.n = len(vars)
        assert self.n == 1
        self.month_vals = nn.Parameter(torch.randn(11))
        self.n_inter = n_inter
        self.p = nn.Parameter(0.5*torch.ones(n_inter))
        self.trend = nn.Parameter

    #month rep - 11 X 0/1 var, 1 for every month except january
    def forward(self, sales_by_var, month_rep, trend_rep):
        assert sales_by_var.size()[0] == self.n_inter
        assert month_rep.size()[0] == self.n_inter



