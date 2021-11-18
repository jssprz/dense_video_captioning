import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from utils import get_init_weights, bow_vectors
from model.programmer.proposals import ProposalsEncoder


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
        self.training_proposals = config.training_proposals
        self.training_programmer = config.training_programmer

        self.embedding_size = config.embedding_size
        self.h_size = config.h_size
        self.mm_size = mm_config.out_size

        self.caps_vocab_size = len(caps_vocab)
        self.pos_vocab_size = len(pos_vocab)
        self.sem_enc_size = sem_tagger_config.out_size

        self.proposals_enc = ProposalsEncoder(config, proposals_tagger_config, num_proposals)

        if self.training_programmer:
            # freeze a pre-trained proposals encoder
            for p in self.proposals_enc.parameters():
                p.requires_grad = False

            self.progs_vocab_size = len(progs_vocab)
            self.fc = nn.Linear(proposals_tagger_config.rnn_h_size, self.progs_vocab_size)

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

        self.p = torch.zeros(bs, dtype=torch.long).to(device)
        self.q = torch.ones(bs, dtype=torch.long).to(device)

        caps_count = torch.zeros(bs, dtype=torch.long).to(device)

        # concat visual features of frames
        v_fcat = torch.cat(v_feats, dim=-1)

        self.proposals_enc.reset_internals(v_fcat)

        if self.training_proposals:
            # initialize tensor to store the logits of START PROPOSALS
            prop_logits_s = torch.zeros_like(gt_prop_s)

            # initialize tensor to store the logits of END PROPOSALS
            prop_logits_e = torch.zeros_like(gt_prop_e)

        if self.training_programmer:
            prog_logits = torch.zeros(gt_program.size(0), gt_program.size(1), self.progs_vocab_size).to(device)

        # STOP CONDITION, according if we are training or testing
        # if self.training:
        # iterate at least prog_len steps, generating at least a caption for each video
        condition = lambda i: i < prog_len and not torch.all(
            caps_count >= gt_caps_count
        )  # or torch.any(caps_count < 1)

        seq_pos = 0

        if self.training_proposals:
            # train and validate proposals
            while condition(seq_pos):
                a_id = gt_program[:, seq_pos]
                self.proposals_enc.train_val_step(
                    prop_logits_s, prop_logits_e, caps_count, self.p, self.q, a_id, v_fcat, feats_count, max_back_steps
                )

                seq_pos += 1

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

        if self.training_programmer:
            # train and validate instruction generation
            a_id = torch.ones(bs, dtype=torch.long).to(device)
            while condition(seq_pos):
                logits_s, logits_e = self.proposals_enc.dense_step(
                    a_id, self.p, self.q, v_fcat, feats_count, max_back_steps
                )

                comb_logits = logits_s + logits_e

                # produce an instruction
                a_logits = self.fc(comb_logits)

                # get instructions for next step
                a_id = a_logits.max(1)[1]

                # save logits for loss computation
                prog_logits[:, seq_pos, :] = a_logits

                # excecute a forced skip on videos that produce a generate instructions, to avoid deadlock
                caps_count[a_id == 2] += 1
                a_id[a_id == 2] = 0

                seq_pos += 1

            return (
                prog_logits,
                None,  # program,
                None,  # caps_logits,
                None,  # caps_sem_enc,
                None,  # pos_tag_logits,
                None,  # captions,
                None,  # intervals,
                caps_count,
                None,  # prop_logits_s,
                None,  # prop_logits_e,
                None,  # self.p,
            )

        # if self.training:
        #     caps_count = torch.min(caps_count, gt_caps_count)

