import random

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
            pool_channels=v_enc_config.pool_channels,
            rnn_size=v_enc_config.rnn_size,
            conv_channels=v_enc_config.conv_channels,
            conv_kernel_sizes=v_enc_config.conv_kernel_sizes,
            mapping_h_sizes=v_enc_config.mapping_h_sizes,
            mapping_in_drop_p=v_enc_config.mapping_in_drop_p,
            mapping_h_drop_ps=v_enc_config.mapping_h_drop_ps,
            concate=v_enc_config.concate,
            have_last_bn=v_enc_config.have_last_bn,
            pretrained_model_path=v_enc_config.pretrained_model_path,
        )

        self.sem_model = TaggerMultilevel(sem_tagger_config)

        self.syn_model = POSTagger(syn_tagger_config, pos_vocab, pretrained_pe, device)

    def forward(
        self, v_feats, feats_count, v_global, teacher_forcing_p, gt_pos=None, max_words=None,
    ):
        v_feats_cat = torch.cat(v_feats, dim=-1)

        v_enc = self.visual_model(
            cnn_feats=v_feats[0], c3d_feats=v_feats[1], video_global_feat=v_global, lengths=feats_count
        )

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


class EnsembleDecoder(nn.Module):
    def __init__(
        self,
        config,
        avscn_dec_config,
        semsynan_dec_config,
        caps_vocab,
        with_embedding_layer=True,
        pretrained_we=None,
        device=None,
    ):
        super(EnsembleDecoder, self).__init__()

        self.out_size = len(caps_vocab)
        self.train_sample_max = config.train_sample_max
        self.test_sample_max = config.test_sample_max
        self.embedding_size = config.embedding_size
        self.output_size = len(caps_vocab)

        if with_embedding_layer:
            if pretrained_we is not None:
                self.embedding = nn.Embedding.from_pretrained(pretrained_we)
            else:
                self.embedding = nn.Embedding(self.output_size, self.embedding_size)
            self.embedd_drop = nn.Dropout(config.drop_p)

        self.avscn_dec = AVSCNDecoder(
            config=avscn_dec_config,
            vocab=caps_vocab,
            with_embedding_layer=False,
            pretrained_we=pretrained_we,
            device=device,
        )

        self.semsynan_dec = SemSynANDecoder(
            config=semsynan_dec_config,
            vocab=caps_vocab,
            with_embedding_layer=False,
            pretrained_we=pretrained_we,
            device=device,
            dataset_name="MSVD",
        )

    def reset_internals(self, batch_size):
        self.avscn_dec.reset_internals(batch_size)
        self.semsynan_dec.reset_internals(batch_size)

    def precompute_mats(self, v_pool, s_tags, pos_emb):
        self.avscn_dec.precompute_mats(v_pool, s_tags)
        self.semsynan_dec.precompute_mats(v_pool, s_tags, pos_emb)

    def step(self, v_feats, s_tags, pos_emb, decoder_input):
        ls1 = self.avscn_dec.step(v_feats, decoder_input)
        ls2 = self.semsynan_dec.step(v_feats, s_tags, pos_emb, decoder_input)

        return (ls1 + ls2) / 2

    def forward_fn(self, v_feats, v_pool, s_tags, pos_emb, gt_captions, tf_p, max_words):
        bs = v_pool.size(0)
        self.reset_internals(bs)
        self.precompute_mats(v_pool, s_tags, pos_emb)

        outputs, embedds, words = [], [], []

        # (batch_size x embedding_size)
        decoder_input = torch.zeros(bs, self.embedding_size).to(v_pool.device)

        if not self.training:
            for _ in range(max_words):
                word_logits = self.step(v_feats, s_tags, pos_emb, decoder_input)

                # compute word probs
                if self.test_sample_max:
                    # sample max probailities
                    # (batch_size)
                    word_id = word_logits.max(dim=1)[1]
                else:
                    # sample from distribution
                    # (batch_size)
                    word_id = torch.multinomial(torch.softmax(word_logits, dim=1), 1).squeeze(1)

                # (batch_size) -> (batch_size x embedding_size)
                decoder_input = self.embedding(word_id).squeeze(1)
                # decoder_input = self.embedd_drop(decoder_input)

                embedds.append(decoder_input)
                outputs.append(word_logits)
                words.append(word_id)
        else:
            for seq_pos in range(gt_captions.size(1)):
                word_logits = self.step(v_feats, s_tags, pos_emb, decoder_input)

                use_teacher_forcing = random.random() < tf_p or seq_pos == 0
                if use_teacher_forcing:
                    # use the correct words,
                    # (batch_size)
                    word_id = gt_captions[:, seq_pos]
                elif self.train_sample_max:
                    # select the words ids with the max probability,
                    # (batch_size)
                    word_id = word_logits.max(1)[1]
                else:
                    # sample words from probability distribution
                    # (batch_size)
                    word_id = torch.multinomial(torch.softmax(word_logits, dim=1), 1).squeeze(1)

                # (batch_size) -> (batch_size x embedding_size)
                decoder_input = self.embedding(word_id).squeeze(1)
                embedds.append(decoder_input)

                decoder_input = self.embedd_drop(decoder_input)

                outputs.append(word_logits)
                words.append(word_id)

        return (
            torch.cat([o.unsqueeze(1) for o in outputs], dim=1).contiguous(),
            torch.cat([w.unsqueeze(1) for w in words], dim=1).contiguous(),
            torch.cat([e.unsqueeze(1) for e in embedds], dim=1).contiguous(),
        )

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, encoding, teacher_forcing_p, gt_captions, max_words):
        return self.forward_fn(
            v_feats=encoding[0],
            v_pool=encoding[1],
            s_tags=encoding[2],
            pos_emb=encoding[3],
            tf_p=teacher_forcing_p,
            gt_captions=gt_captions,
            max_words=max_words,
        )


class Ensemble(nn.Module):
    def __init__(
        self,
        v_size,
        visual_enc_config,
        sem_tagger_config,
        syn_tagger_config,
        ensemble_dec_config,
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

        self.decoder = EnsembleDecoder(
            config=ensemble_dec_config,
            avscn_dec_config=avscn_dec_config,
            semsynan_dec_config=semsynan_dec_config,
            caps_vocab=caps_vocab,
            with_embedding_layer=True,
            pretrained_we=pretrained_we,
            device=device,
        )

    def forward(
        self, v_feats, feats_count, v_global, tf_ratios, gt_captions=None, gt_pos=None, max_words=None,
    ):
        # get encodings from v_feats and v_global
        encoding = self.encoder(
            v_feats=v_feats,
            v_global=v_global,
            teacher_forcing_p=tf_ratios["syn_enc"],
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

        logits, cap, _ = self.decoder(
            encoding=encoding, teacher_forcing_p=tf_ratios["cap_dec"], gt_captions=gt_captions, max_words=max_words,
        )

        return logits, cap, sem_enc, pos_tag_seq_logits
