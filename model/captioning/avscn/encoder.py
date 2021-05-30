import torch.nn as nn

from model.tagging.semantic import TaggerMLP


class SCNEncoder(nn.Module):
    def __init__(
        self,
        in_size=6144,
        tagger_h_sizes=[1024, 512],
        tagger_out_size=300,
        tagger_in_dropout_p=0.5,
        tagger_dropout_ps=[0.5, 0.5],
        tagger_have_last_bn=False,
    ):
        super(SCNEncoder, self).__init__()

        self.sem_tagger = TaggerMLP(
            in_size=in_size,
            h_sizes=tagger_h_sizes,
            out_size=tagger_out_size,
            in_dropout_p=tagger_in_dropout_p,
            dropout_ps=tagger_dropout_ps,
            have_last_bn=tagger_have_last_bn,
        )

    def forward(self, v_feats):
        return v_feats, self.sem_tagger(v_feats)
