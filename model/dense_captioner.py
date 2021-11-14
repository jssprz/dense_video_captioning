import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.nn.parameter import Parameter

from utils import get_init_weights, bow_vectors
from model.captioning.attention import Attention
from model.captioning.ensemble import Ensemble
from model.embeddings.visual_semantic import MultiModal
from model.tagging.semantic import TaggerMLP


class VNCLCell(nn.Module):
    def __init__(
        self, x_size, v_size, mm_size, vh_size, h1_size, h2_size, drop_p=0.5, have_bn=False, var_dropout="per-gate",
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

    def __compute_gate(self, activation, temp1, temp2, temp3, temp4, temp5, temp6, Wc, Cc, Uc, b):
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

        self.embedding_size = config.embedding_size
        self.h_size = config.h_size
        self.mm_size = mm_config.out_size
        self.prop_rnn_h_size = proposals_tagger_config.rnn_h_size
        self.num_proposals = num_proposals

        self.progs_vocab_size = len(progs_vocab)
        self.caps_vocab_size = len(caps_vocab)
        self.pos_vocab_size = len(pos_vocab)
        self.sem_enc_size = sem_tagger_config.out_size

        # if pretrained_ope is not None:
        #     self.embedding = nn.Embedding.from_pretrained(pretrained_ope)
        # else:
        #     self.embedding = nn.Embedding(self.progs_vocab_size, self.embedding_size)
        # self.embedd_drop = nn.Dropout(config.drop_p)

        # self.mm_enc = MultiModal(
        #     v_enc_config=mm_config.v_enc_config,
        #     t_enc_config=mm_config.t_enc_config,
        #     out_size=mm_config.out_size,
        #     vocab_size=self.caps_vocab_size,
        # )

        # self.rnn_cell = VNCLCell(
        #     x_size=self.embedding_size,
        #     v_size=config.cnn_feats_size + config.c3d_feats_size,
        #     mm_size=mm_config.out_size,
        #     vh_size=vncl_cell_config.vh_size,
        #     h1_size=vncl_cell_config.h1_size,
        #     h2_size=config.h_size,
        #     drop_p=config.drop_p,
        # )

        # self.prop_fc = nn.Linear(in_features=config.cnn_feats_size+config.c3d_feats_size, out_features=num_proposals)

        # temporal attention modules
        self.v_p_attn = Attention(
            seq_len=self.future_steps, hidden_size=proposals_tagger_config.rnn_h_size, mode="soft"
        )
        self.v_q_attn = Attention(
            seq_len=self.future_steps, hidden_size=proposals_tagger_config.rnn_h_size, mode="soft"
        )

        # start proposals module
        self.prop_s_rnn_0 = nn.LSTMCell(
            input_size=config.cnn_feats_size + config.c3d_feats_size, hidden_size=proposals_tagger_config.rnn_h_size,
        )

        self.prop_s_back_rnn_0 = nn.LSTM(
            input_size=config.cnn_feats_size + config.c3d_feats_size,
            hidden_size=proposals_tagger_config.rnn_h_size,
            batch_first=True,
        )

        self.prop_s_rnn_1 = nn.LSTMCell(
            input_size=proposals_tagger_config.rnn_h_size, hidden_size=proposals_tagger_config.rnn_h_size,
        )

        # end proposals module
        self.prop_e_rnn_0 = nn.LSTMCell(
            input_size=config.cnn_feats_size + config.c3d_feats_size, hidden_size=proposals_tagger_config.rnn_h_size,
        )

        self.prop_e_back_rnn_0 = nn.LSTM(
            input_size=config.cnn_feats_size + config.c3d_feats_size,
            hidden_size=proposals_tagger_config.rnn_h_size,
            batch_first=True,
        )

        self.prop_e_rnn_1 = nn.LSTMCell(
            input_size=proposals_tagger_config.rnn_h_size, hidden_size=proposals_tagger_config.rnn_h_size,
        )

        # self.e_prop_rnn = nn.LSTM(
        #     input_size=config.cnn_feats_size + config.c3d_feats_size,
        #     hidden_size=proposals_tagger_config.rnn_h_size,
        #     batch_first=True,
        # )

        self.prop_enc_s = TaggerMLP(
            v_size=proposals_tagger_config.rnn_h_size * 2,
            out_size=num_proposals,
            h_sizes=proposals_tagger_config.h_sizes,
            in_drop_p=proposals_tagger_config.in_drop_p,
            drop_ps=proposals_tagger_config.drop_ps,
            have_last_bn=proposals_tagger_config.have_last_bn,
        )

        self.prop_enc_e = TaggerMLP(
            v_size=proposals_tagger_config.rnn_h_size * 2,
            out_size=num_proposals,
            h_sizes=proposals_tagger_config.h_sizes,
            in_drop_p=proposals_tagger_config.in_drop_p,
            drop_ps=proposals_tagger_config.drop_ps,
            have_last_bn=proposals_tagger_config.have_last_bn,
        )

        # self.fc = nn.Linear(in_features=config.h_size + num_proposals, out_features=self.progs_vocab_size,)

        # self.clip_captioner = Ensemble(
        #     v_size=config.cnn_feats_size + config.c3d_feats_size,
        #     sem_tagger_config=sem_tagger_config,
        #     syn_embedd_config=syn_embedd_config,
        #     syn_tagger_config=syn_tagger_config,
        #     avscn_dec_config=avscn_dec_config,
        #     semsynan_dec_config=semsynan_dec_config,
        #     caps_vocab=caps_vocab,
        #     pretrained_we=pretrained_we,
        #     pos_vocab=pos_vocab,
        #     pretrained_pe=pretrained_pe,
        #     device=device,
        # )

    def freeze(self, resume_config):
        if resume_config.freeze_captioning:
            for name, p in self.named_parameters():
                if "clip_captioner" in name:
                    p.requires_grad = False
        elif resume_config.random_captioning:
            self.clip_captioner.reset_parameters()

        if resume_config.freeze_programmer:
            self.embedding.requires_grad = False
            self.mm_enc.requires_grad = False
            self.rnn_cell.requires_grad = False
            self.prop_enc.requires_grad = False
            self.fc.requires_grad = False
        elif resume_config.random_programmer:
            self.embedding.reset_parameters()
            self.mm_enc.reset_parameters()
            self.rnn_cell.reset_parameters()
            self.prop_enc.reset_parameters()
            self.fc.reset_parameters()

    def unfreeze(self):
        for _, p in self.named_parameters():
            p.requires_grad = True

    def get_clip_feats(self, v_feats, start_idx, end_idx):
        bs = start_idx.size(0)
        # import ipdb; ipdb.set_trace() # BREAKPOINT
        feats = [torch.zeros(bs, self.max_clip_len, f.size(2)).to(f.device) for f in v_feats]
        pool = torch.zeros(bs, sum([f.size(2) for f in feats])).to(feats[0].device)
        for i, (s, e) in enumerate(zip(start_idx, end_idx)):
            if (e - s) > self.max_clip_len:
                indices = torch.linspace(s, min(e, v_feats[0].size(1) - 1), steps=self.max_clip_len, dtype=torch.long,)
                f1 = v_feats[0][i, indices, :]
                f2 = v_feats[1][i, indices, :]
            else:
                f1 = v_feats[0][i, s:e, :]
                f2 = v_feats[1][i, s:e, :]

            feats[0][i, : f1.size(0), :] = f1
            feats[1][i, : f2.size(0), :] = f2

            # pool.append(torch.cat((torch.mean(f1, dim=0), torch.mean(f2, dim=0))))
            pool[i, :] = torch.cat((f1, f2), dim=1).mean(dim=0)

        return pool, feats
        # return torch.stack(pool), feats

    # TODO: check if using a moving pool is faster than the pool in each step
    # it is needed to initialize the moving_pool_sum, last_window_sum, and moving_pool to zero,
    # call this function at the begining (with p,q = 0,1), and after each enqueue
    def update_feats_moving_pool(self, v_feats, i):
        feats_count = self.q[i] - self.p[i]
        if feats_count % self.temp_window_pool == 0:
            self.moving_pool_sum[i, :] += (
                self.last_window_sum[i, :]
                + torch.cat((v_feats[0][i, self.q[i], :], v_feats[0][i, self.q[i], :]), dim=-1)
            ) / self.temp_window_pool
            self.last_window_sum[i, :] = torch.zeros(v_feats[0].size(-1) + v_feats[1].size(-1)).to(v_feats[0].device)
            self.moving_pool[i, :] = self.moving_pool_sum[i, :] / (feats_count / self.temp_window_pool)
        else:
            self.last_window_sum[i, :] += torch.cat((v_feats[0][i, self.q[i], :], v_feats[0][i, self.q[i], :]), dim=-1)
            last_window_pool = self.last_window_sum[i, :] / (feats_count % self.temp_window_pool)
            self.moving_pool[i, :] = (self.moving_pool_sum[i, :] + last_window_pool) / (
                feats_count // self.temp_window_pool + 1
            )

    def __step__(self, t, v_feats):
        self.v_p_q_pool, self.v_p_q_feats = self.get_clip_feats(v_feats, self.p, self.q)
        v_q_w_pool, _ = self.get_clip_feats(v_feats, self.q, self.q + self.future_steps)

        self.h, self.c = self.rnn_cell(
            self.h, self.c, self.x, self.prev_match, self.v_p_q_pool, v_q_w_pool, var_drop_p=0.1,
        )

        self.a_logits = self.fc(torch.cat((self.h, self.current_proposals), dim=1))

    def forward(
        self,
        v_feats,
        feats_count,
        prog_len=100,
        teacher_forcing_p=0.5,
        gt_program=None,
        gt_captions=None,
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
    ):
        # initialize
        bs, device = v_feats[0].size(0), v_feats[0].device

        caps_count = torch.zeros(bs, dtype=torch.long).to(device)
        self.p = torch.zeros(bs, dtype=torch.long).to(device)
        self.q = torch.ones(bs, dtype=torch.long).to(device)
        # self.x = torch.zeros(bs, self.embedding_size).to(device)

        # concat visual features of frames
        v_fcat = torch.cat(v_feats, dim=-1)

        # initialize the visual features of each pointer, averaging the first frames
        v_p = v_fcat[:, : self.future_steps, :].mean(dim=1)
        v_q = v_fcat[:, 1 : 1 + self.future_steps, :].mean(dim=1)

        # list of indices to be used for fill-in the first step of recurrencies
        idxs = list(range(bs))

        # INITIALIZE TENSORS OF START PROPOSALS MODULE
        # initialize the hidden and cell state of first layer (forward) of START MODULE
        prop_s_h_0 = torch.zeros(bs, self.prop_rnn_h_size).to(device)
        prop_s_c_0 = torch.zeros(bs, self.prop_rnn_h_size).to(device)

        # compute first step of first layer (forward) of START MODULE from the v_p features
        prop_s_h_0[idxs, :], prop_s_c_0[idxs, :] = self.prop_s_rnn_0(v_p, (prop_s_h_0[idxs, :], prop_s_c_0[idxs, :]))

        # initialize the hidden states of recurrent layer to compute backward of START MODULE
        prop_s_back_h_0 = torch.zeros(bs, self.prop_rnn_h_size).to(device)

        # initialize the hidden and cell state of second layer (forward) of START MODULE
        prop_s_h_1 = torch.zeros(bs, self.prop_rnn_h_size).to(device)
        prop_s_c_1 = torch.zeros(bs, self.prop_rnn_h_size).to(device)

        # initialize tensor to store the logits of START PROPOSALS
        prop_logits_s = torch.zeros_like(gt_prop_s)

        # INITIALIZE TENSORS OF END PROPOSALS MODULE
        # initialize the hidden and cell state of first layer (forward) of END MODULE
        prop_e_h_0 = torch.zeros(bs, self.prop_rnn_h_size).to(device)
        prop_e_c_0 = torch.zeros(bs, self.prop_rnn_h_size).to(device)

        # compute first two steps of first layer (forward) of END MODULE from the v_p and v_q features, respectively
        prop_e_h_0[idxs, :], prop_e_c_0[idxs, :] = self.prop_e_rnn_0(v_p, (prop_e_h_0[idxs, :], prop_e_c_0[idxs, :]))
        prop_e_h_0[idxs, :], prop_e_c_0[idxs, :] = self.prop_e_rnn_0(v_q, (prop_e_h_0[idxs, :], prop_e_c_0[idxs, :]))

        # set the greater positions reached by the first layer (forward) of END MODULE similar to self.q positions
        # this tensor is used to avoid the computation of first layar step over the same visual features
        prop_e_0_pos = torch.ones(bs, dtype=torch.long).to(device)

        # initialize the hidden states of recurrent layer to compute backward of END MODULE
        prop_e_back_h_0 = torch.zeros(bs, self.prop_rnn_h_size).to(device)

        # initialize the hidden and cell state of second layer (forward) of END MODULE
        prop_e_h_1 = torch.zeros(bs, self.prop_rnn_h_size).to(device)
        prop_e_c_1 = torch.zeros(bs, self.prop_rnn_h_size).to(device)

        # initialize tensor to store the logits of END PROPOSALS
        prop_logits_e = torch.zeros_like(gt_prop_e)

        # STOP CONDITION, according if we are training or testing
        # if self.training:
        # iterate at least prog_len steps, generating at least a caption for each video
        condition = lambda i: i < prog_len and not torch.all(
            caps_count >= gt_caps_count
        )  # or torch.any(caps_count < 1)

        seq_pos = 0
        while condition(seq_pos):
            # self.__step__(seq_pos, v_feats)
            a_id = gt_program[:, seq_pos]

            # get idxs of videos that require the pointers be skipped
            vix_2_skip = (a_id == 0).nonzero(as_tuple=True)[0]
            # move pointers according to the skip instruction
            self.p[vix_2_skip] += 1
            self.q[vix_2_skip] = self.p[vix_2_skip] + 1

            # get idxs of videos that require new visual features be enqueued
            vix_2_adv = (a_id == 1).nonzero(as_tuple=True)[0]
            # move the q pointers according to the equeue instruction
            self.q[vix_2_adv] += 1

            # refine the idxs of videos to be advanced according to the greated values stored in prop_e_0_pos.
            # Here we also consider the vix_2_skip idxs because skip instuction can produce a greater q than the position stored in prop_e_0_pos
            vix_2_adv = torch.tensor(
                list(
                    set(
                        vix_2_skip[(self.q[vix_2_skip] > prop_e_0_pos[vix_2_skip]).nonzero(as_tuple=True)[0]].tolist()
                        + vix_2_adv[(self.q[vix_2_adv] > prop_e_0_pos[vix_2_adv]).nonzero(as_tuple=True)[0]].tolist()
                    )
                ),
                dtype=torch.long,
            )

            # get idxs of videos that require a new caption be generated
            vix_2_dscr = (a_id == 2).nonzero(as_tuple=True)[0]

            # advance p pointers and reset q pointers for videos that produce skip instruction
            if len(vix_2_skip) > 0:
                # fidx = torch.min(self.p[vix_2_skip], feats_count[vix_2_skip])
                # v_p = v_fcat[vix_2_skip, fidx, :]

                # determine the initial and end frames of windows used to encode the visual features related to pointer p
                fidx_from = torch.min(self.p[vix_2_skip], feats_count[vix_2_skip] - self.future_steps)
                fidx_to = torch.min(self.p[vix_2_skip] + self.future_steps, feats_count[vix_2_skip])

                # tensor to be filled according to the number of frames in windows
                v_p = torch.zeros(len(vix_2_skip), v_fcat.size(-1)).to(device)

                # videos with windows that have at least the minimum num of frames required to compute the attention
                attn_mask = (fidx_from >= 0).nonzero(as_tuple=True)[0]
                if len(attn_mask):
                    vix_2_attn = vix_2_skip[attn_mask]
                    v_p[attn_mask] = self.v_p_attn(
                        torch.stack(
                            [
                                v_fcat[vix, ff:ft, :]
                                for vix, ff, ft in zip(vix_2_attn, fidx_from[attn_mask], fidx_to[attn_mask])
                            ]
                        ),
                        prop_s_h_0[vix_2_attn, :],
                    )

                # videos with windows that have less than the minimum num of frames required to compute the attention
                fill_mask = (fidx_from < 0).nonzero(as_tuple=True)[0]
                if len(fill_mask):
                    vix_2_fill = vix_2_skip[fill_mask]
                    v_p[fill_mask] = v_fcat[vix_2_fill, torch.min(self.p[vix_2_fill], feats_count[vix_2_fill]), :]

                # compute a recurrent step of the first layer (forward) of START MODULE
                prop_s_h_0[vix_2_skip, :], prop_s_c_0[vix_2_skip, :] = self.prop_s_rnn_0(
                    v_p, (prop_s_h_0[vix_2_skip, :], prop_s_c_0[vix_2_skip, :])
                )

            # advance the q pointers for videos that produce enqueue instruction
            if len(vix_2_adv) > 0:
                # update the gratest positions stored in prop_e_0_pos
                prop_e_0_pos[vix_2_adv] = self.q[vix_2_adv]

                # fidx = torch.min(self.q[vix_2_adv], feats_count[vix_2_adv])
                # v_q = v_fcat[vix_2_adv, fidx, :]

                # determine the initial and end frames of windows used to encode the visual features related to pointer q
                fidx_from = torch.min(self.q[vix_2_adv], feats_count[vix_2_adv] - self.future_steps)
                fidx_to = torch.min(self.q[vix_2_adv] + self.future_steps, feats_count[vix_2_adv])

                # tensor to be filled according to the number of frames in windows
                v_q = torch.zeros(len(vix_2_adv), v_fcat.size(-1)).to(device)

                # videos with windows that have at least the minimum num of frames required to compute the attention
                attn_mask = (fidx_from >= 0).nonzero(as_tuple=True)[0]
                if len(attn_mask):
                    vix_2_attn = vix_2_adv[attn_mask]
                    v_q[attn_mask] = self.v_q_attn(
                        torch.stack(
                            [
                                v_fcat[vix, ff:ft, :]
                                for vix, ff, ft in zip(vix_2_attn, fidx_from[attn_mask], fidx_to[attn_mask])
                            ]
                        ),
                        prop_e_h_0[vix_2_attn, :],
                    )

                # videos with windows that have less than the minimum num of frames required to compute the attention
                fill_mask = (fidx_from < 0).nonzero(as_tuple=True)[0]
                if len(fill_mask):
                    vix_2_fill = vix_2_adv[fill_mask]
                    v_q[fill_mask] = v_fcat[vix_2_fill, torch.min(self.q[vix_2_fill], feats_count[vix_2_fill]), :]

                # compute a recurrent step of the first layer (forward) of END MODULE
                prop_e_h_0[vix_2_adv, :], prop_e_c_0[vix_2_adv, :] = self.prop_e_rnn_0(
                    v_q, (prop_e_h_0[vix_2_adv, :], prop_e_c_0[vix_2_adv, :])
                )

            # generate a caption from the current video-clips saved in the sub-batch
            if len(vix_2_dscr) > 0:
                # print(seq_pos, use_teacher_forcing, teacher_forcing_p, vix_2_dscr, caps_count, self.p.data, self.q.data)
                # get sub-batch of video features to be described
                # clip_feats = [feats[vix_2_dscr, :, :] for feats in self.v_p_q_feats]
                # clip_global = self.v_p_q_pool[vix_2_dscr, :]

                # START MODULE
                # compute the recurrency on frames previous to poiter p in backward direction
                ends = torch.min(self.p[vix_2_dscr] + 1, feats_count[vix_2_dscr])
                starts = (ends - max_back_steps).clamp(min=0)

                sub_v_feats_padded = pad_sequence(
                    [v_fcat[v, s:e, :].flip((0,)) for v, s, e in zip(vix_2_dscr, starts, ends)], batch_first=True,
                )
                sub_v_feats_packed = pack_padded_sequence(
                    input=sub_v_feats_padded,
                    lengths=(ends - starts).to("cpu"),
                    batch_first=True,
                    enforce_sorted=False,
                )
                _, (prop_s_back_h_0[vix_2_dscr], _) = self.prop_s_back_rnn_0(sub_v_feats_packed)

                # compute another step of the second layer (forward) of START MOUDULE
                (prop_s_h_1[vix_2_dscr, :], prop_s_c_1[vix_2_dscr, :],) = self.prop_s_rnn_1(
                    prop_s_h_0[vix_2_dscr, :], (prop_s_h_1[vix_2_dscr, :], prop_s_c_1[vix_2_dscr, :],),
                )

                # compute the proposal encoding for start position from the second layer (forward) and the backward on previous frames
                prop_logits_s[vix_2_dscr, caps_count[vix_2_dscr], :] = self.prop_enc_s(
                    torch.cat((prop_s_h_1[vix_2_dscr], prop_s_back_h_0[vix_2_dscr]), dim=-1)
                )[0]

                # END MODULE
                # compute the recurrency on frames previous to poiter q in backward direction
                max_fix = feats_count[vix_2_dscr] - 1
                ends = torch.min(self.q[vix_2_dscr], max_fix) + 1
                starts = torch.max(torch.min(self.p[vix_2_dscr], max_fix), ends - max_back_steps)

                sub_v_feats_padded = pad_sequence(
                    [v_fcat[v, s:e, :].flip((0,)) for v, s, e in zip(vix_2_dscr, starts, ends)], batch_first=True,
                )
                sub_v_feats_packed = pack_padded_sequence(
                    input=sub_v_feats_padded,
                    lengths=(ends - starts).to("cpu"),
                    batch_first=True,
                    enforce_sorted=False,
                )
                _, (prop_e_back_h_0[vix_2_dscr], _) = self.prop_e_back_rnn_0(sub_v_feats_packed)

                # compute another step of the second layer (forward) of END MOUDULE
                (prop_e_h_1[vix_2_dscr, :], prop_e_c_1[vix_2_dscr, :],) = self.prop_e_rnn_1(
                    prop_e_h_0[vix_2_dscr, :], (prop_e_h_1[vix_2_dscr, :], prop_e_c_1[vix_2_dscr, :],),
                )

                # compute the proposal encoding for end position from the second layer (forward) and the backward on previous frames
                prop_logits_e[vix_2_dscr, caps_count[vix_2_dscr], :] = self.prop_enc_e(
                    torch.cat((prop_e_h_1[vix_2_dscr], prop_e_back_h_0[vix_2_dscr]), dim=-1)
                )[0]

                caps_count[vix_2_dscr] += 1

            seq_pos += 1

        # if self.training:
        #     caps_count = torch.min(caps_count, gt_caps_count)

        return (
            None,  # prog_logits,
            None,  # program,
            None,  # caps_logits,
            None,  # caps_sem_enc,
            None,  # pos_tag_logits,
            None,  # captions,
            None,  # intervals,
            caps_count,
            prop_logits_s,
            prop_logits_e,
            None,  # self.p,
        )
