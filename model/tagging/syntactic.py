import torch
import torch.nn as nn

from model.embeddings.multilevel_enc import VisualMultiLevelEncoding
from model.captioning.sem_syn_an.decoder import SemSynANDecoder


class POSTagger(nn.Module):
    def __init__(self, syn_embedd_config, syn_tagger_config, pos_vocab, pretrained_pe, device):
        super(POSTagger, self).__init__()
        
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

        self.semsynan_dec = SemSynANDecoder(config=syn_tagger_config,
                                            vocab=pos_vocab,
                                            pretrained_we=pretrained_pe,
                                            device=device,
                                            dataset_name='MSVD')

    def forward(self, encoding, v_feats, teacher_forcing_p, gt_pos=None, max_words=None):
        multilevel_enc = self.syn_model(v_feats[0], v_feats[1], encoding[1])
        return self.semsynan_dec(encoding=encoding+[multilevel_enc], 
                                 teacher_forcing_p=teacher_forcing_p, 
                                 gt_captions=gt_pos,
                                 max_words=max_words)