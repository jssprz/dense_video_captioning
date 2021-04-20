import torch
import torch.nn as nn

from model.captioning.avscn.decoder import AVSCNDecoder
from model.captioning.sem_syn_an.decoder import SemSynANDecoder
from model.tagging.semantic import TaggerMLP
from model.embeddings.multilevel_enc import VisualMultiLevelEncoding


class ClipEncoder(nn.Module):
    def __init__(self, v_size, sem_tagger_config, syn_embedd_config):
        super(ClipEncoder, self).__init__()

        self.sem_model = TaggerMLP(v_size=v_size,
                                   out_size=sem_tagger_config.out_size,
                                   h_sizes=sem_tagger_config.h_sizes,
                                   in_drop_p=sem_tagger_config.in_drop_p,
                                   drop_ps=sem_tagger_config.drop_ps,
                                   have_last_bn=sem_tagger_config.have_last_bn)

        self.syn_model = VisualMultiLevelEncoding(cnn_feats_size=syn_embedd_config.v_enc_config.cnn_feats_size,
                                                  c3d_feats_size=syn_embedd_config.v_enc_config.cnn_feats_size,
                                                  global_feat_size=syn_embedd_config.v_enc_config.global_feat_size,
                                                  out_size=syn_embedd_config.v_enc_config.out_size,
                                                  norm=syn_embedd_config.v_enc_config.norm,
                                                  drop_p=syn_embedd_config.v_enc_config.drop_p,
                                                  mapping_h_sizes=syn_embedd_config.v_enc_config.mapping_h_sizes,
                                                  mapping_in_drop_p=syn_embedd_config.v_enc_config.mapping_in_drop_p,
                                                  mapping_h_drop_ps=syn_embedd_config.v_enc_config.mapping_h_drop_ps,
                                                  have_last_bn=syn_embedd_config.v_enc_config.have_last_bn,
                                                  pretrained_model_path=syn_embedd_config.v_enc_config.pretrained_model_path)

    def forward(self, v_feats, v_global):
        v_feats_cat = torch.cat(v_feats, dim=2)
        return v_feats_cat, v_global, self.sem_model(v_global)[1], self.syn_model(v_feats[0], v_feats[1], v_global)


class Ensemble(nn.Module):
    def __init__(self, v_size, sem_tagger_config, syn_embedd_config,
                 avscn_dec_config, semsynan_dec_config, caps_vocab,
                 pretrained_we, device):
        super(Ensemble, self).__init__()

        self.out_size = len(caps_vocab)

        self.encoder = ClipEncoder(v_size=v_size,
                                   sem_tagger_config=sem_tagger_config,
                                   syn_embedd_config=syn_embedd_config)

        self.avscn_dec = AVSCNDecoder(config=avscn_dec_config,
                                      vocab=caps_vocab,
                                      pretrained_we=pretrained_we,
                                      device=device)

        self.semsynan_dec = SemSynANDecoder(config=semsynan_dec_config,
                                            vocab=caps_vocab,
                                            pretrained_we=pretrained_we,
                                            device=device,
                                            dataset_name='MSVD')

        self.decoders = [self.avscn_dec, self.semsynan_dec]

    def forward(self, v_feats, v_global, teacher_forcing_p=.5, gt_captions=None):
        # get encodings from v_feats and v_global
        encoding = self.encoder(v_feats, v_global)

        # ensemble decoders
        logits = torch.zeros(v_global.size(0), gt_captions.size(1), self.out_size).to(v_feats[0].device)
        for dec in self.decoders:
            ls, _ = dec(encoding, teacher_forcing_p, gt_captions)
            logits += ls
        logits /= len(self.decoders)

        return logits
