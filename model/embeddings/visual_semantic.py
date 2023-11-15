import torch
import torch.nn as nn

from model.embeddings.multilevel_enc import (
    VisualMultiLevelEncoding,
    TextMultiLevelEncoding,
)


class MultiModal(nn.Module):
    def __init__(self, v_enc_config, t_enc_config, out_size, vocab_size):
        super(MultiModal, self).__init__()

        self.v_model = VisualMultiLevelEncoding(
            cnn_feats_size=v_enc_config.cnn_feats_size,
            c3d_feats_size=v_enc_config.c3d_feats_size,
            global_feat_size=v_enc_config.global_feat_size,
            out_size=out_size,
            norm=v_enc_config.norm,
            rnn_size=v_enc_config.rnn_size,
            cnn_out_channels=v_enc_config.cnn_out_channels,
            cnn_kernel_sizes=v_enc_config.cnn_kernel_sizes,
            mapping_h_sizes=v_enc_config.mapping_h_sizes,
            mapping_in_drop_p=v_enc_config.mapping_in_drop_p,
            mapping_h_drop_ps=v_enc_config.mapping_h_drop_ps,
            concate=v_enc_config.concate,
            have_last_bn=v_enc_config.have_last_bn,
            pretrained_model_path=v_enc_config.pretrained_model_path,
        )

        self.t_model = TextMultiLevelEncoding(
            vocab_size=vocab_size,
            out_size=out_size,
            norm=t_enc_config.norm,
            word_dim=t_enc_config.word_dim,
            rnn_size=t_enc_config.rnn_size,
            mapping_h_sizes=t_enc_config.mapping_h_sizes,
            mapping_in_drop_p=t_enc_config.mapping_in_drop_p,
            mapping_h_drop_ps=t_enc_config.mapping_h_drop_ps,
            concate=t_enc_config.concate,
            cnn_out_channels=t_enc_config.cnn_out_channels,
            cnn_kernel_sizes=t_enc_config.cnn_kernel_sizes,
            have_last_bn=t_enc_config.have_last_bn,
            pretrained_model_path=t_enc_config.pretrained_model_path,
        )

        # self.v_model = nn.Linear(v_enc_config.global_feat_size, out_size)
        # self.t_model = nn.Linear(vocab_size, out_size)

        self.merge_model = nn.Linear(out_size + out_size, out_size)

    def forward(self, v_feats, v_global, cap, cap_len, cap_bow):
        v_enc = self.v_model(v_feats[0], v_feats[1], v_global)
        t_enc = self.t_model(cap, cap_len, cap_bow)

        # v_enc = self.v_model(v_global)
        # t_enc = self.t_model(cap_bow)

        fusion = torch.cat((v_enc, t_enc), dim=1)

        return self.merge_model(fusion)
