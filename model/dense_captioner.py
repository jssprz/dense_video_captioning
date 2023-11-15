import random

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from utils import get_init_weights, bow_vectors
from model.programmer.proposals import ProposalsEncoder
from model.captioning.ensemble import Ensemble


class VNCLCell(nn.Module):
    def __init__(
        self,
        x_size,
        v_size,
        mm_size,
        vh_size,
        h1_size,
        h2_size,
        drop_p=0.5,
        have_bn=False,
        var_dropout="per-gate",
    ):
        super(VNCLCell, self).__init__()

        self.x_size = x_size
        self.v_size = v_size
        self.mm_size = mm_size
        self.h1_size = h1_size
        self.h2_size = h2_size

        self.drop_p = drop_p
        self.have_bn = have_bn
        self.var_dropout = var_dropout

        # for composing v1 and v2
        self.V_i = get_init_weights((v_size * 2, vh_size))
        self.V_f = get_init_weights((v_size * 2, vh_size))
        self.V_o = get_init_weights((v_size * 2, vh_size))
        self.V_c = get_init_weights((v_size * 2, vh_size))
        # self.V_i_1 = get_init_weights((v_size, vh_size))
        # self.V_f_1 = get_init_weights((v_size, vh_size))
        # self.V_o_1 = get_init_weights((v_size, vh_size))
        # self.V_c_1 = get_init_weights((v_size, vh_size))

        # self.V_i_2 = get_init_weights((v_size, vh_size))
        # self.V_f_2 = get_init_weights((v_size, vh_size))
        # self.V_o_2 = get_init_weights((v_size, vh_size))
        # self.V_c_2 = get_init_weights((v_size, vh_size))

        # for composing v and m
        self.C_i_1 = get_init_weights((vh_size, h1_size))
        self.C_f_1 = get_init_weights((vh_size, h1_size))
        self.C_o_1 = get_init_weights((vh_size, h1_size))
        self.C_c_1 = get_init_weights((vh_size, h1_size))

        self.C_i_2 = get_init_weights((mm_size, h1_size))
        self.C_f_2 = get_init_weights((mm_size, h1_size))
        self.C_o_2 = get_init_weights((mm_size, h1_size))
        self.C_c_2 = get_init_weights((mm_size, h1_size))

        self.C_i_3 = get_init_weights((h1_size, h2_size))
        self.C_f_3 = get_init_weights((h1_size, h2_size))
        self.C_o_3 = get_init_weights((h1_size, h2_size))
        self.C_c_3 = get_init_weights((h1_size, h2_size))

        # for composing x and m
        self.W_i_1 = get_init_weights((x_size, h1_size))
        self.W_f_1 = get_init_weights((x_size, h1_size))
        self.W_o_1 = get_init_weights((x_size, h1_size))
        self.W_c_1 = get_init_weights((x_size, h1_size))

        self.W_i_2 = get_init_weights((mm_size, h1_size))
        self.W_f_2 = get_init_weights((mm_size, h1_size))
        self.W_o_2 = get_init_weights((mm_size, h1_size))
        self.W_c_2 = get_init_weights((mm_size, h1_size))

        self.W_i_3 = get_init_weights((h1_size, h2_size))
        self.W_f_3 = get_init_weights((h1_size, h2_size))
        self.W_o_3 = get_init_weights((h1_size, h2_size))
        self.W_c_3 = get_init_weights((h1_size, h2_size))

        # for composing h and m
        self.U_i_1 = get_init_weights((h2_size, h1_size))
        self.U_f_1 = get_init_weights((h2_size, h1_size))
        self.U_o_1 = get_init_weights((h2_size, h1_size))
        self.U_c_1 = get_init_weights((h2_size, h1_size))

        self.U_i_2 = get_init_weights((mm_size, h1_size))
        self.U_f_2 = get_init_weights((mm_size, h1_size))
        self.U_o_2 = get_init_weights((mm_size, h1_size))
        self.U_c_2 = get_init_weights((mm_size, h1_size))

        self.U_i_3 = get_init_weights((h1_size, h2_size))
        self.U_f_3 = get_init_weights((h1_size, h2_size))
        self.U_o_3 = get_init_weights((h1_size, h2_size))
        self.U_c_3 = get_init_weights((h1_size, h2_size))

        self.b_i = Parameter(torch.zeros(h2_size))
        self.b_f = Parameter(torch.zeros(h2_size))
        self.b_o = Parameter(torch.zeros(h2_size))
        self.b_c = Parameter(torch.zeros(h2_size))

        self.have_bn = have_bn
        if have_bn:
            self.bn = nn.LayerNorm(hidden_size)

        self.__init_layers()

    def __init_layers(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

    def __dropout(self, x, keep_prob, mask_for):
        # if not self.training or keep_prob >= 1.:
        #     return x

        # if mask_for in self.dropM:
        #     mask = self.dropM[mask_for]
        # else:
        #     # op1
        #     # mask = Binomial(probs=keep_prob).sample(x.size()).to(x.device)  # máscara de acuerdo a keep_prob

        #     # op2
        #     mask = x.new_empty(x.size(), requires_grad=False).bernoulli_(keep_prob)

        #     self.dropM[mask_for] = mask

        # assert x.device == mask.device, 'mask and x must be in the same device'

        # return x.masked_fill(mask==0, 0) * (1.0 / keep_prob)
        return x

    def precompute_dots_4_m(self, m, var_drop_p):
        self.dropM = {}

        keep_prob = 1 - var_drop_p
        if self.var_dropout == "per-gate":
            # use a distinct mask for each gate
            m_i = self.__dropout(m, keep_prob, "m_i")
            m_f = self.__dropout(m, keep_prob, "m_f")
            m_o = self.__dropout(m, keep_prob, "m_o")
            m_c = self.__dropout(m, keep_prob, "m_c")
        else:
            # use the same mask for all gates
            m_i = self.__dropout(m, keep_prob, "m")
            m_f = self.__dropout(m, keep_prob, "m")
            m_o = self.__dropout(m, keep_prob, "m")
            m_c = self.__dropout(m, keep_prob, "m")

        # (batch_size x h1_size)
        self.temp2_i = m_i @ self.C_i_2
        self.temp2_f = m_f @ self.C_f_2
        self.temp2_o = m_o @ self.C_o_2
        self.temp2_c = m_c @ self.C_c_2

        # (batch_size x h2_size)
        self.temp4_i = m_i @ self.W_i_2
        self.temp4_f = m_f @ self.W_f_2
        self.temp4_o = m_o @ self.W_o_2
        self.temp4_c = m_c @ self.W_c_2

    def __compute_gate(
        self, activation, temp1, temp2, temp3, temp4, temp5, temp6, Wc, Cc, Uc, b
    ):
        z = (temp1 * temp2) @ Cc
        x = (temp3 * temp4) @ Wc
        h = (temp5 * temp6) @ Uc

        # assert torch.all(torch.tensor([x.device == t.device for t in [v,h,b]])), 'all tensor must be in the same device ({}, {}, {}, {})'.format(x.device, v.device, h.device, b.device)

        logits = z + x + h + b

        if self.have_bn:
            logits = self.bn(logits)

        return activation(logits)

    def forward(self, prev_h, prev_c, x, m, v1, v2, var_drop_p):
        keep_prob = 1 - var_drop_p

        v = torch.cat((v1, v2), dim=-1)

        if self.var_dropout == "per-gate":
            # use a distinct mask for each gate
            h_i = self.__dropout(prev_h, keep_prob, "h_i")
            h_f = self.__dropout(prev_h, keep_prob, "h_f")
            h_o = self.__dropout(prev_h, keep_prob, "h_o")
            h_c = self.__dropout(prev_h, keep_prob, "h_c")

            x_i = self.__dropout(x, keep_prob, "x_i")
            x_f = self.__dropout(x, keep_prob, "x_f")
            x_o = self.__dropout(x, keep_prob, "x_o")
            x_c = self.__dropout(x, keep_prob, "x_c")

            m_i = self.__dropout(m, keep_prob, "m_i")
            m_f = self.__dropout(m, keep_prob, "m_f")
            m_o = self.__dropout(m, keep_prob, "m_o")
            m_c = self.__dropout(m, keep_prob, "m_c")

            v_i = self.__dropout(v, keep_prob, "v_i")
            v_f = self.__dropout(v, keep_prob, "v_f")
            v_o = self.__dropout(v, keep_prob, "v_o")
            v_c = self.__dropout(v, keep_prob, "v_c")

            # v1_i = self.__dropout(v1, keep_prob, "v1_i")
            # v1_f = self.__dropout(v1, keep_prob, "v1_f")
            # v1_o = self.__dropout(v1, keep_prob, "v1_o")
            # v1_c = self.__dropout(v1, keep_prob, "v1_c")

            # v2_i = self.__dropout(v2, keep_prob, "v2_i")
            # v2_f = self.__dropout(v2, keep_prob, "v2_f")
            # v2_o = self.__dropout(v2, keep_prob, "v2_o")
            # v2_c = self.__dropout(v2, keep_prob, "v2_c")
        else:
            # use the same mask for all gates
            h_i = self.__dropout(prev_h, keep_prob, "h")
            h_f = self.__dropout(prev_h, keep_prob, "h")
            h_o = self.__dropout(prev_h, keep_prob, "h")
            h_c = self.__dropout(prev_h, keep_prob, "h")

            x_i = self.__dropout(x, keep_prob, "x")
            x_f = self.__dropout(x, keep_prob, "x")
            x_o = self.__dropout(x, keep_prob, "x")
            x_c = self.__dropout(x, keep_prob, "x")

            m_i = self.__dropout(m, keep_prob, "m")
            m_f = self.__dropout(m, keep_prob, "m")
            m_o = self.__dropout(m, keep_prob, "m")
            m_c = self.__dropout(m, keep_prob, "m")

            v_i = self.__dropout(v, keep_prob, "v")
            v_f = self.__dropout(v, keep_prob, "v")
            v_o = self.__dropout(v, keep_prob, "v")
            v_c = self.__dropout(v, keep_prob, "v")

            # v1_i = self.__dropout(v1, keep_prob, "v1")
            # v1_f = self.__dropout(v1, keep_prob, "v1")
            # v1_o = self.__dropout(v1, keep_prob, "v1")
            # v1_c = self.__dropout(v1, keep_prob, "v1")

            # v2_i = self.__dropout(v2, keep_prob, "v2")
            # v2_f = self.__dropout(v2, keep_prob, "v2")
            # v2_o = self.__dropout(v2, keep_prob, "v2")
            # v2_c = self.__dropout(v2, keep_prob, "v2")

        temp1_i = (v_i @ self.V_i) @ self.C_i_1
        temp1_f = (v_f @ self.V_f) @ self.C_f_1
        temp1_o = (v_o @ self.V_o) @ self.C_o_1
        temp1_c = (v_c @ self.V_c) @ self.C_c_1

        # temp1_i = ((v1_i @ self.V_i_1) * (v2_i @ self.V_i_2)) @ self.C_i_1
        # temp1_f = ((v1_f @ self.V_f_1) * (v2_f @ self.V_f_2)) @ self.C_f_1
        # temp1_o = ((v1_o @ self.V_o_1) * (v2_o @ self.V_o_2)) @ self.C_o_1
        # temp1_c = ((v1_c @ self.V_c_1) * (v2_c @ self.V_c_2)) @ self.C_c_1

        # (batch_size x rnn_hidden_size)
        temp3_i = x_i @ self.W_i_1
        temp3_f = x_f @ self.W_f_1
        temp3_o = x_o @ self.W_o_1
        temp3_c = x_c @ self.W_c_1

        # (batch_size x rnn_hidden_size)
        temp5_i = h_i @ self.U_i_1
        temp5_f = h_f @ self.U_f_1
        temp5_o = h_o @ self.U_o_1
        temp5_c = h_c @ self.U_c_1

        # (batch_size x rnn_hidden_size)
        temp6_i = m_i @ self.U_i_2
        temp6_f = m_f @ self.U_f_2
        temp6_o = m_o @ self.U_o_2
        temp6_c = m_c @ self.U_c_2

        # (batch_size x hidden_size)
        i = self.__compute_gate(
            torch.sigmoid,
            temp1_i,
            self.temp2_i,
            temp3_i,
            self.temp4_i,
            temp5_i,
            temp6_i,
            self.W_i_3,
            self.C_i_3,
            self.U_i_3,
            self.b_i,
        )
        f = self.__compute_gate(
            torch.sigmoid,
            temp1_f,
            self.temp2_f,
            temp3_f,
            self.temp4_f,
            temp5_f,
            temp6_f,
            self.W_f_3,
            self.C_f_3,
            self.U_f_3,
            self.b_f,
        )
        o = self.__compute_gate(
            torch.sigmoid,
            temp1_o,
            self.temp2_o,
            temp3_o,
            self.temp4_o,
            temp5_o,
            temp6_o,
            self.W_o_3,
            self.C_o_3,
            self.U_o_3,
            self.b_o,
        )
        c = self.__compute_gate(
            torch.tanh,
            temp1_c,
            self.temp2_c,
            temp3_c,
            self.temp4_c,
            temp5_c,
            temp6_c,
            self.W_c_3,
            self.C_c_3,
            self.U_c_3,
            self.b_c,
        )

        # (batch_size x hidden_size)
        new_c = f * prev_c + i * c
        new_h = o * torch.tanh(new_c)

        return new_h, new_c


class DenseCaptioner(nn.Module):
    def __init__(
        self,
        config,
        visual_enc_config,
        sem_tagger_config,
        syn_tagger_config,
        ensemble_dec_config,
        avscn_dec_config,
        semsynan_dec_config,
        mm_config,
        vncl_cell_config,
        proposals_tagger_config,
        num_proposals,
        progs_vocab,
        pretrained_ope,
        caps_vocab,
        pretrained_we,
        pos_vocab,
        pretrained_pe,
        device,
    ):
        super(DenseCaptioner, self).__init__()

        self.train_sample_max = config.train_sample_max
        self.test_sample_max = config.test_sample_max
        self.max_clip_len = config.max_clip_len
        self.future_steps = config.future_steps

        self.training_proposals = config.training_proposals
        self.training_programmer = config.training_programmer
        self.training_captioning = config.training_captioning
        self.training_sem_enc = config.training_sem_enc
        self.training_syn_enc = config.training_syn_enc

        self.embedding_size = config.embedding_size
        self.h_size = config.h_size
        self.mm_size = mm_config.out_size

        self.caps_vocab_size = len(caps_vocab)
        self.pos_vocab_size = len(pos_vocab)
        self.sem_enc_size = sem_tagger_config.out_size
        self.v_enc_size = visual_enc_config.out_size

        self.proposals_enc = ProposalsEncoder(
            config, proposals_tagger_config, num_proposals
        )
        if not self.training_proposals:
            # freeze a pre-trained proposals encoder
            for p in self.proposals_enc.parameters():
                p.requires_grad = False

        self.progs_vocab_size = len(progs_vocab)
        self.fc = nn.Linear(proposals_tagger_config.rnn_h_size, self.progs_vocab_size)
        if not self.training_programmer:
            for p in self.fc.parameters():
                p.requires_grad = False

        self.clip_captioner = Ensemble(
            v_size=config.cnn_feats_size + config.c3d_feats_size,
            visual_enc_config=visual_enc_config,
            sem_tagger_config=sem_tagger_config,
            syn_tagger_config=syn_tagger_config,
            ensemble_dec_config=ensemble_dec_config,
            avscn_dec_config=avscn_dec_config,
            semsynan_dec_config=semsynan_dec_config,
            caps_vocab=caps_vocab,
            pretrained_we=pretrained_we,
            pos_vocab=pos_vocab,
            pretrained_pe=pretrained_pe,
            device=device,
        )
        if not self.training_captioning:
            for p in self.clip_captioner.parameters():
                p.requires_grad = False

    def freeze(self, resume_config):
        if resume_config.freeze_captioning:
            for name, p in self.named_parameters():
                if "clip_captioner" in name:
                    p.requires_grad = False
        elif resume_config.random_captioning:
            self.clip_captioner.reset_parameters()

        if resume_config.freeze_programmer:
            self.embedding.requires_grad = False

    def load_state_dict(self, state_dict, resume_config):
        if resume_config.load_cap_v_enc:
            sub_key = "clip_captioner.encoder.visual_model."
            self.clip_captioner.encoder.visual_model.load_state_dict(
                {k[len(sub_key) :]: v for k, v in state_dict.items() if sub_key in k}
            )
        if resume_config.load_cap_sem_enc:
            sub_key = "clip_captioner.encoder.sem_model."
            self.clip_captioner.encoder.sem_model.load_state_dict(
                {k[len(sub_key) :]: v for k, v in state_dict.items() if sub_key in k}
            )
        if resume_config.load_cap_syn_enc:
            sub_key = "clip_captioner.encoder.syn_model."
            self.clip_captioner.encoder.syn_model.load_state_dict(
                {k[len(sub_key) :]: v for k, v in state_dict.items() if sub_key in k}
            )
        if resume_config.load_cap_decoder:
            sub_key = "clip_captioner.decoder."
            self.clip_captioner.decoder.load_state_dict(
                {k[len(sub_key) :]: v for k, v in state_dict.items() if sub_key in k}
            )
        if resume_config.load_programmer:
            self.mm_enc.load_state_dict(
                {k: v for k, v in state_dict.items() if k.startswith("mm_enc.")}
            )
            self.rnn_cell.load_state_dict(
                {k: v for k, v in state_dict.items() if k.startswith("rnn_cell.")}
            )
            self.proposal_enc.load_state_dict(
                {k: v for k, v in state_dict.items() if k.startswith("proposal_enc.")}
            )
            self.fc.load_state_dict(
                {k: v for k, v in state_dict.items() if k.startswith("fc.")}
            )

    def freeze_dict(self, config_dict):
        if "sem_enc" in config_dict and config_dict["sem_enc"]:
            self.clip_captioner.encoder.sem_model.freeze()
        if "syn_enc" in config_dict and config_dict["syn_enc"]:
            self.clip_captioner.encoder.syn_model.freeze()
        if "cap_dec" in config_dict and config_dict["cap_dec"]:
            self.clip_captioner.encoder.visual_model.freeze()
            self.clip_captioner.decoder.freeze()
        if "programmer" in config_dict and config_dict["programmer"]:
            self.fc.requires_grad = False
        elif resume_config.random_programmer:
            self.embedding.reset_parameters()
            self.mm_enc.reset_parameters()
            self.rnn_cell.reset_parameters()
            self.prop_enc.reset_parameters()
            self.fc.reset_parameters()

    def freeze_config(self, config_obj):
        self.freeze_dict(
            {
                "proposals": config_obj.freeze_proposals,
                "programmer": config_obj.freeze_programmer,
                "sem_enc": config_obj.freeze_cap_sem_enc,
                "syn_enc": config_obj.freeze_cap_syn_enc,
                "cap_dec": config_obj.freeze_cap_decoder,
            }
        )

    def unfreeze(self):
        for _, p in self.named_parameters():
            p.requires_grad = True

    def get_clip_feats(self, v_feats, start_idx, end_idx):
        bs = start_idx.size(0)
        feats = [
            torch.zeros(bs, self.max_clip_len, f.size(2)).to(f.device) for f in v_feats
        ]
        pool = torch.zeros(bs, sum([f.size(2) for f in feats])).to(feats[0].device)
        clip_len = torch.zeros(bs, dtype=torch.long).to(feats[0].device)
        for i, (s, e) in enumerate(zip(start_idx, end_idx)):
            if (e - s) > self.max_clip_len:
                indices = torch.linspace(
                    s,
                    min(e, v_feats[0].size(1) - 1),
                    steps=self.max_clip_len,
                    dtype=torch.long,
                )
                f1 = v_feats[0][i, indices, :]
                f2 = v_feats[1][i, indices, :]
                clip_len[i] = len(indices)
            else:
                f1 = v_feats[0][i, s:e, :]
                f2 = v_feats[1][i, s:e, :]
                clip_len[i] = e - s

            feats[0][i, : f1.size(0), :] = f1
            feats[1][i, : f2.size(0), :] = f2

            # pool.append(torch.cat((torch.mean(f1, dim=0), torch.mean(f2, dim=0))))
            pool[i, :] = torch.cat((f1, f2), dim=1).mean(dim=0)

        return pool, feats, clip_len
        # return torch.stack(pool), feats

    # TODO: check if using a moving pool is faster than the pool in each step
    # it is needed to initialize the moving_pool_sum, last_window_sum, and moving_pool to zero,
    # call this function at the begining (with p,q = 0,1), and after each enqueue
    def update_feats_moving_pool(self, v_feats, i):
        feats_count = self.q[i] - self.p[i]
        if feats_count % self.temp_window_pool == 0:
            self.moving_pool_sum[i, :] += (
                self.last_window_sum[i, :]
                + torch.cat(
                    (v_feats[0][i, self.q[i], :], v_feats[0][i, self.q[i], :]), dim=-1
                )
            ) / self.temp_window_pool
            self.last_window_sum[i, :] = torch.zeros(
                v_feats[0].size(-1) + v_feats[1].size(-1)
            ).to(v_feats[0].device)
            self.moving_pool[i, :] = self.moving_pool_sum[i, :] / (
                feats_count / self.temp_window_pool
            )
        else:
            self.last_window_sum[i, :] += torch.cat(
                (v_feats[0][i, self.q[i], :], v_feats[0][i, self.q[i], :]), dim=-1
            )
            last_window_pool = self.last_window_sum[i, :] / (
                feats_count % self.temp_window_pool
            )
            self.moving_pool[i, :] = (self.moving_pool_sum[i, :] + last_window_pool) / (
                feats_count // self.temp_window_pool + 1
            )

    def __step__(self, t, v_feats):
        self.v_p_q_pool, self.v_p_q_feats, self.clip_len = self.get_clip_feats(
            v_feats, self.p, self.q
        )
        # v_q_w_pool, _ = self.get_clip_feats(v_feats, self.q, self.q + self.future_steps)

        # self.h, self.c = self.rnn_cell(
        #     self.h, self.c, self.x, self.prev_match, self.v_p_q_pool, v_q_w_pool, var_drop_p=0.1,
        # )

        # self.a_logits = self.fc(torch.cat((self.h, self.current_proposals), dim=1))

    def compute_captioning_batch(
        self,
        clip_feats,
        clip_len,
        clip_global,
        clip_context_v_enc,
        gt_c,
        gt_p,
        max_cap,
        tf_ratios,
    ):
        # TODO: create batch from lists
        clip_feats = [torch.cat(feats, dim=0) for feats in clip_feats]
        clip_len = torch.cat(clip_len, dim=0)
        clip_global = torch.cat(clip_global, dim=0)
        clip_context_v_enc = torch.cat(clip_context_v_enc, dim=0)
        gt_c = torch.cat(gt_c, dim=0)
        gt_p = torch.cat(gt_p, dim=0)

        # TODO:compute captioning
        (
            cap_logits,
            cap,
            cap_sem_enc,
            pos,
            pos_tag_seq_logits,
            v_enc,
        ) = self.clip_captioner(
            v_feats=clip_feats,
            v_global=clip_global,
            context_v_enc=clip_context_v_enc,
            tf_ratios=tf_ratios,
            gt_captions=gt_c,
            gt_pos=gt_p,
            max_words=max_cap,
            feats_count=clip_len,
        )

        return cap, cap_logits, cap_sem_enc, pos, pos_tag_seq_logits, v_enc

    def forward(
        self,
        v_feats,
        feats_count,
        tf_ratios,
        prog_len=100,
        gt_prog=None,
        gt_caps=None,
        gt_caps_count=None,
        gt_sem_enc=None,
        gt_pos=None,
        gt_intervals=None,
        gt_prop_s=None,
        gt_prop_e=None,
        max_prog=None,
        max_caps=None,
        max_cap=None,
        max_chunks=None,
        max_back_steps=10,
        captioning_batch=10,
    ):
        # initialize
        bs, device = v_feats[0].size(0), v_feats[0].device

        self.p = torch.zeros(bs, dtype=torch.long).to(device)
        self.q = torch.ones(bs, dtype=torch.long).to(device)

        caps_count = torch.zeros(bs, dtype=torch.long).to(device)
        aux_caps_count = torch.zeros(bs, dtype=torch.long).to(device)

        # concat visual features of frames
        v_fcat = torch.cat(v_feats, dim=-1)

        self.proposals_enc.reset_internals(v_fcat)

        # initialize result tensors according to the sizes of ground truth
        prop_logits_s = torch.zeros_like(gt_prop_s)  # the logits of START PROPOSALS
        prop_logits_e = torch.zeros_like(gt_prop_e)  # the logits of END PROPOSALS
        prog_logits = torch.zeros(
            gt_prog.size(0), gt_prog.size(1), self.progs_vocab_size
        ).to(
            device
        )  # the logits of PROGRAM
        caps = torch.zeros_like(gt_caps)
        pos_tags = torch.zeros_like(gt_pos)
        caps_sem_enc = torch.zeros_like(gt_sem_enc)
        intervals = torch.zeros_like(gt_intervals, dtype=torch.float).to(device)
        context_v_enc = torch.zeros((bs, self.v_enc_size)).to(device)
        # else:
        #     # iterate until all pointers reach the end
        #     condition = (
        #         lambda i: i < max_prog
        #         and not torch.all(self.p >= max_chunks - 1)
        #         and not torch.all(caps_count >= max_caps)
        #     )

        #     # initialize result tensors according to the maximum sizes
        #     captions = torch.zeros((bs, max_caps, max_cap), dtype=torch.long).to(device)
        #     caps_sem_enc = torch.zeros(bs, max_caps, self.sem_enc_size).to(device)
        #     pos_tags = torch.zeros(bs, max_caps, max_cap).to(device)
        #     intervals = torch.zeros(bs, max_caps, 2).to(device)

        caps_logits = torch.zeros(
            caps.size(0), caps.size(1), caps.size(2), self.caps_vocab_size
        ).to(device)
        pos_tag_logits = torch.zeros(
            pos_tags.size(0), pos_tags.size(1), pos_tags.size(2), self.pos_vocab_size
        ).to(device)

        clip_feats = [[] for _ in v_feats]
        clip_lens = []
        clip_global = []
        clip_context_v_enc = []
        gt_c = []
        gt_p = []
        vixs = []

        seq_pos = 0

        # STOP CONDITION, according if we are training or testing
        # if self.training:
        # iterate at least prog_len steps, generating at least a caption for each video
        condition = lambda i: i < prog_len and not torch.all(
            caps_count >= gt_caps_count
        )  # or torch.any(caps_count < 1)

        a_id = torch.ones(bs, dtype=torch.long).to(device)
        while condition(seq_pos):
            ##########################################
            ######## PERFORM PROGRAM ACTION ##########
            ##########################################

            # vix_2_dscr, logits_s, logits_e = self.proposals_enc.dense_step(
            #     a_id=a_id,
            #     p=self.p,
            #     q=self.q,
            #     v_feats=v_fcat,
            #     feats_count=feats_count,
            #     max_back_steps=max_back_steps,
            # )

            logits_s, logits_e = self.proposals_enc.train_val_step(
                prop_logits_s=prop_logits_s,
                prop_logits_e=prop_logits_e,
                caps_count=caps_count,
                p=self.p,
                q=self.q,
                a_id=a_id,
                v_feats=v_fcat,
                feats_count=feats_count,
                max_back_steps=max_back_steps,
            )

            #####################################
            ########## GENERATE CAPTION #########
            #####################################

            vix_2_dscr = ((a_id == 2) * (aux_caps_count < gt_caps_count)).nonzero(
                as_tuple=True
            )[0]

            # generate a caption from the current video-clips saved in the sub-batch
            if len(vix_2_dscr) > 0:
                intervals[vix_2_dscr, aux_caps_count[vix_2_dscr], 0] = self.p[
                    vix_2_dscr
                ].float()
                intervals[vix_2_dscr, aux_caps_count[vix_2_dscr], 1] = self.q[
                    vix_2_dscr
                ].float()

                self.__step__(seq_pos, v_feats)
                for i, feats in enumerate(clip_feats):
                    feats.append(self.v_p_q_feats[i][vix_2_dscr, :, :])
                clip_lens.append(self.clip_len[vix_2_dscr])
                clip_global.append(self.v_p_q_pool[vix_2_dscr, :])
                clip_context_v_enc.append(context_v_enc[vix_2_dscr, :])
                gt_c.append(
                    torch.stack(
                        [
                            gt_caps[i][min(gt_caps.size(1) - 1, aux_caps_count[i])]
                            for i in vix_2_dscr
                        ]
                    )
                )
                gt_p.append(
                    torch.stack(
                        [
                            gt_pos[i][min(gt_pos.size(1) - 1, aux_caps_count[i])]
                            for i in vix_2_dscr
                        ]
                    )
                )
                vixs += vix_2_dscr

                aux_caps_count[vix_2_dscr] += 1

                if len(vixs) >= captioning_batch:
                    # compute captioning for batch, considering teacher forcing strategy for cap tensor
                    (
                        cap,
                        cap_logits,
                        cap_sem_enc,
                        pos,
                        pos_tag_seq_logits,
                        v_enc,
                    ) = self.compute_captioning_batch(
                        clip_feats=clip_feats,
                        clip_len=clip_lens,
                        clip_global=clip_global,
                        clip_context_v_enc=clip_context_v_enc,
                        gt_c=gt_c,
                        gt_p=gt_p,
                        max_cap=max_cap,
                        tf_ratios=tf_ratios,
                    )
                    # fill the result tensors according to caps_count and vixs lists
                    for i, vix in enumerate(vixs):
                        caps[vix, caps_count[vix], :] = cap[i]
                        pos_tags[vix, caps_count[vix], :] = pos[i]
                        caps_logits[vix, caps_count[vix], :, :] = cap_logits[i]
                        pos_tag_logits[vix, caps_count[vix], :, :] = pos_tag_seq_logits[
                            i
                        ]
                        caps_sem_enc[vix, caps_count[vix], :] = cap_sem_enc[i]
                        context_v_enc[vix, :] = v_enc[i]

                    caps_count[torch.stack(vixs)] += 1

                    clip_feats = [[] for _ in v_feats]
                    clip_lens = []
                    clip_global = []
                    clip_context_v_enc = []
                    gt_c = []
                    gt_p = []
                    vixs = []

            #####################################
            ### DETERMINE NEXT PROGRAM ACTION ###
            #####################################

            comb_logits = logits_s + logits_e

            # produce an instruction
            a_logits = self.fc(comb_logits)

            # get instructions for next step
            use_teacher_forcing = (
                random.random() < tf_ratios["programmer"] or seq_pos == 0
            )
            if use_teacher_forcing:
                a_id = gt_prog[:, seq_pos]
            else:
                a_id = a_logits.max(1)[1]

            # save logits for loss computation
            prog_logits[:, seq_pos, :] = a_logits

            seq_pos += 1

        return (
            prog_logits,
            caps_logits,
            caps_sem_enc,
            pos_tags,
            pos_tag_logits,
            caps,
            intervals,
            caps_count,
            prop_logits_s,
            prop_logits_e,
            None,  # self.p,
        )
