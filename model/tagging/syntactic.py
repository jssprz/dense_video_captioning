from model.captioning.avscn.decoder import AVSCNDecoder
import torch
import torch.nn as nn

from model.embeddings.multilevel_enc import VisualMultiLevelEncoding
from model.captioning.sem_syn_an.decoder import SemSynANDecoder


class POSTagger(nn.Module):
    def __init__(self, config, pos_vocab, pretrained_pe, device):
        super(POSTagger, self).__init__()

        self.syn_model = VisualMultiLevelEncoding(
            cnn_feats_size=config.enc_config.cnn_feats_size,
            c3d_feats_size=config.enc_config.c3d_feats_size,
            global_feat_size=config.enc_config.global_feat_size,
            out_size=config.enc_config.out_size,
            norm=config.enc_config.norm,
            drop_p=config.enc_config.drop_p,
            rnn_size=config.enc_config.rnn_size,
            mapping_h_sizes=config.enc_config.mapping_h_sizes,
            mapping_in_drop_p=config.enc_config.mapping_in_drop_p,
            mapping_h_drop_ps=config.enc_config.mapping_h_drop_ps,
            have_last_bn=config.enc_config.have_last_bn,
            pretrained_model_path=config.enc_config.pretrained_model_path,
        )

        if config.dec_config.arch == "SemSynANDecoder":
            self.pos_dec = SemSynANDecoder(
                config=config.dec_config,
                vocab=pos_vocab,
                pretrained_we=pretrained_pe,
                device=device,
                dataset_name="MSVD",
            )
        elif config.dec_config.arch == "AVSCNDecoder":
            self.pos_dec = AVSCNDecoder(
                config=config.dec_config, vocab=pos_vocab, pretrained_we=pretrained_pe, device=device
            )

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, encoding, v_feats, feats_count, teacher_forcing_p, gt_pos=None, max_words=None):
        multilevel_enc = self.syn_model(v_feats[0], v_feats[1], encoding[2], lengths=feats_count)

        if type(self.pos_dec) == SemSynANDecoder:
            encoding = [encoding[0], multilevel_enc, encoding[1], encoding[2]]
        elif type(self.pos_dec) == AVSCNDecoder:
            encoding = [encoding[0], multilevel_enc, encoding[1]]

        return self.pos_dec(
            encoding=encoding, teacher_forcing_p=teacher_forcing_p, gt_captions=gt_pos, max_words=max_words,
        )

