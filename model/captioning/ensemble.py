import torch
import torch.nn as nn

from model.captioning.avscn.decoder import AVSCNDecoder
from model.captioning.sem_syn_an.decoder import SemSynANDecoder
from model.embeddings.multilevel_enc import VisualMultiLevelEncoding
from model.tagging.semantic import TaggerMLP, TaggerMultilevel
from model.tagging.syntactic import POSTagger


class ClipEncoder(nn.Module):
    def __init__(
        self, v_size, v_enc_config, sem_tagger_config, syn_tagger_config, pos_vocab, pretrained_pe, device,
    ):
        super(ClipEncoder, self).__init__()

        # self.sem_model = TaggerMLP(
        #     v_size=v_size,
        #     out_size=sem_tagger_config.out_size,
        #     h_sizes=sem_tagger_config.h_sizes,
        #     in_drop_p=sem_tagger_config.in_drop_p,
        #     drop_ps=sem_tagger_config.drop_ps,
        #     have_last_bn=sem_tagger_config.have_last_bn,
        # )

        self.visual_model = VisualMultiLevelEncoding(
            cnn_feats_size=v_enc_config.cnn_feats_size,
            c3d_feats_size=v_enc_config.c3d_feats_size,
            global_feat_size=v_enc_config.global_feat_size,
            out_size=v_enc_config.out_size,
            norm=v_enc_config.norm,
            drop_p=v_enc_config.drop_p,
            rnn_size=v_enc_config.rnn_size,
            mapping_h_sizes=v_enc_config.mapping_h_sizes,
            mapping_in_drop_p=v_enc_config.mapping_in_drop_p,
            mapping_h_drop_ps=v_enc_config.mapping_h_drop_ps,
            have_last_bn=v_enc_config.have_last_bn,
            pretrained_model_path=v_enc_config.pretrained_model_path,
        )

        self.sem_model = TaggerMultilevel(sem_tagger_config)

        self.syn_model = POSTagger(syn_tagger_config, pos_vocab, pretrained_pe, device)

    def forward(
        self, v_feats, feats_count, v_global, teacher_forcing_p, gt_pos=None, max_words=None,
    ):
        v_feats_cat = torch.cat(v_feats, dim=-1)

        v_enc = self.visual_model(cnn_feats=v_feats[0], c3d_feats=v_feats[1], video_global_feat=v_global, lengths=feats_count)

        # sem_enc = self.sem_model(v_global)
        sem_enc = self.sem_model(v_feats=v_feats, v_global=v_global, feats_count=feats_count)
        sem_enc_no_grad = sem_enc.detach()

        syn_enc = self.syn_model(
            encoding=[v_feats_cat, sem_enc_no_grad, v_global],
            v_feats=v_feats,
            teacher_forcing_p=teacher_forcing_p,
            gt_pos=gt_pos,
            max_words=max_words,
            feats_count=feats_count,
        )
        syn_enc_mean = syn_enc[2].mean(dim=1)
        syn_enc_no_grad = syn_enc_mean.detach()

        return [v_feats_cat, v_enc, sem_enc, syn_enc, syn_enc_mean, sem_enc_no_grad, syn_enc_no_grad]


class Ensemble(nn.Module):
    def __init__(
        self,
        v_size,
        visual_enc_config,
        sem_tagger_config,
        syn_tagger_config,
        avscn_dec_config,
        semsynan_dec_config,
        caps_vocab,
        pretrained_we,
        pos_vocab,
        pretrained_pe,
        device,
    ):
        super(Ensemble, self).__init__()

        self.out_size = len(caps_vocab)

        self.encoder = ClipEncoder(
            v_size=v_size,
            v_enc_config=visual_enc_config,
            sem_tagger_config=sem_tagger_config,
            syn_tagger_config=syn_tagger_config,
            pos_vocab=pos_vocab,
            pretrained_pe=pretrained_pe,
            device=device,
        )

        self.avscn_dec = AVSCNDecoder(
            config=avscn_dec_config, vocab=caps_vocab, pretrained_we=pretrained_we, device=device,
        )

        self.semsynan_dec = SemSynANDecoder(
            config=semsynan_dec_config,
            vocab=caps_vocab,
            pretrained_we=pretrained_we,
            device=device,
            dataset_name="MSVD",
        )

        self.decoders = [self.avscn_dec, self.semsynan_dec]

    def forward(
        self, v_feats, feats_count, v_global, teacher_forcing_p=0.5, gt_captions=None, gt_pos=None, max_words=None,
    ):
        # get encodings from v_feats and v_global
        encoding = self.encoder(
            v_feats=v_feats,
            v_global=v_global,
            teacher_forcing_p=teacher_forcing_p,
            gt_pos=gt_pos,
            max_words=max_words,
            feats_count=feats_count,
        )
        sem_enc, pos_tag_seq_logits = (
            encoding[2],
            encoding[3][0],
        )

        # use the semantic encoding without gradients
        encoding[2] = encoding[5]

        # syntactic encoding 
        # encoding[3] = encoding[4]  # with gradients
        encoding[3] = encoding[6]  # without gradients

        # TODO: evaluate the use of POS tagger as a global controler

        # ensemble decoders
        if self.training:
            logits = torch.zeros(v_global.size(0), gt_captions.size(1), self.out_size).to(v_feats[0].device)
        else:
            logits = torch.zeros(v_global.size(0), max_words, self.out_size).to(v_feats[0].device)

        for dec in self.decoders:
            ls, _, _ = dec(
                encoding=encoding, teacher_forcing_p=teacher_forcing_p, gt_captions=gt_captions, max_words=max_words,
            )
            logits += ls
        logits /= len(self.decoders)

        return logits, sem_enc, pos_tag_seq_logits
