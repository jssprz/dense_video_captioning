import torch
import torch.nn as nn

from model.mlp import MLP
from model.embeddings.multilevel_enc import VisualMultiLevelEncoding


class TaggerMLP(nn.Module):
    def __init__(
        self, v_size, h_sizes, out_size, in_drop_p=0.5, drop_ps=[0.5, 0.5], have_last_bn=False,
    ):
        super(TaggerMLP, self).__init__()

        self.mlp = MLP(
            in_size=v_size,
            h_sizes=h_sizes,
            out_size=out_size,
            in_drop_p=in_drop_p,
            drop_ps=drop_ps,
            have_last_bn=have_last_bn,
        )

    def forward(self, v_globals):
        logits = self.mlp(v_globals)
        return logits


class TaggerMultilevel(nn.Module):
    def __init__(self, config):
        super(TaggerMultilevel, self).__init__()

        self.ml_enc = VisualMultiLevelEncoding(
            cnn_feats_size=config.cnn_feats_size,
            c3d_feats_size=config.cnn_feats_size,
            global_feat_size=config.global_feat_size,
            out_size=config.out_size,
            norm=config.norm,
            drop_p=config.drop_p,
            mapping_h_sizes=config.mapping_h_sizes,
            mapping_in_drop_p=config.mapping_in_drop_p,
            mapping_h_drop_ps=config.mapping_h_drop_ps,
            have_last_bn=config.have_last_bn,
            pretrained_model_path=config.pretrained_model_path,
        )

    def forward(self, v_feats, feats_count, v_global):
        return self.ml_enc(v_feats[0], v_feats[1], v_global, lengths=feats_count)

