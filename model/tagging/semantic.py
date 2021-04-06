import torch
import torch.nn as nn

from model.mlp import MLP


class TaggerMLP(nn.Module):
    def __init__(self, v_size, h_sizes, out_size, in_drop_p=.5, drop_ps=[.5,.5], 
                 have_last_bn=False):
        super(TaggerMLP, self).__init__()

        self.mlp = MLP(in_size=v_size, 
                       h_sizes=h_sizes, 
                       out_size=out_size, 
                       in_drop_p=in_drop_p,
                       drop_ps=drop_ps,
                       have_last_bn=have_last_bn)

    def forward(self, v_globals):
        logits = self.mlp(v_globals)
        probs = torch.sigmoid(logits)
        return logits, probs

        