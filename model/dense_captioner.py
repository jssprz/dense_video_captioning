import random

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from utils import get_init_weights, make_bow_vector
from model.captioning.ensemble import Ensemble
from model.embeddings.visual_semantic import MultiModal


class VNCLCell(nn.Module):
    def __init__(self, x_size, v_size, mm_size, vh_size, h1_size, h2_size, drop_p=.5,
                 have_bn=False, var_dropout='per-gate'):
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
        self.V_i_1 = get_init_weights((v_size, vh_size))
        self.V_f_1 = get_init_weights((v_size, vh_size))
        self.V_o_1 = get_init_weights((v_size, vh_size))
        self.V_c_1 = get_init_weights((v_size, vh_size))

        self.V_i_2 = get_init_weights((v_size, vh_size))
        self.V_f_2 = get_init_weights((v_size, vh_size))
        self.V_o_2 = get_init_weights((v_size, vh_size))
        self.V_c_2 = get_init_weights((v_size, vh_size))

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
        if self.var_dropout == 'per-gate':
            # use a distinct mask for each gate
            m_i = self.__dropout(m, keep_prob, 's_i')
            m_f = self.__dropout(m, keep_prob, 's_f')
            m_o = self.__dropout(m, keep_prob, 's_o')
            m_c = self.__dropout(m, keep_prob, 's_c')
        else:
            # use the same mask for all gates
            m_i = self.__dropout(m, keep_prob, 's')
            m_f = self.__dropout(m, keep_prob, 's')
            m_o = self.__dropout(m, keep_prob, 's')
            m_c = self.__dropout(m, keep_prob, 's')

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

    # def step(self, s, rnn_h, rnn_c, decoder_input, encoder_hidden, encoder_outputs, var_drop_p):
    def forward(self, prev_h, prev_c, x, m, v1, v2, var_drop_p):
        keep_prob = 1 - var_drop_p
        if self.var_dropout == 'per-gate':
            # use a distinct mask for each gate
            h_i = self.__dropout(prev_h, keep_prob, 'h_i')
            h_f = self.__dropout(prev_h, keep_prob, 'h_f')
            h_o = self.__dropout(prev_h, keep_prob, 'h_o')
            h_c = self.__dropout(prev_h, keep_prob, 'h_c')

            x_i = self.__dropout(x, keep_prob, 'x_i')
            x_f = self.__dropout(x, keep_prob, 'x_f')
            x_o = self.__dropout(x, keep_prob, 'x_o')
            x_c = self.__dropout(x, keep_prob, 'x_c')

            m_i = self.__dropout(m, keep_prob, 'm_i')
            m_f = self.__dropout(m, keep_prob, 'm_f')
            m_o = self.__dropout(m, keep_prob, 'm_o')
            m_c = self.__dropout(m, keep_prob, 'm_c')

            v1_i = self.__dropout(v1, keep_prob, 'v1_i')
            v1_f = self.__dropout(v1, keep_prob, 'v1_f')
            v1_o = self.__dropout(v1, keep_prob, 'v1_o')
            v1_c = self.__dropout(v1, keep_prob, 'v1_c')

            v2_i = self.__dropout(v2, keep_prob, 'v2_i')
            v2_f = self.__dropout(v2, keep_prob, 'v2_f')
            v2_o = self.__dropout(v2, keep_prob, 'v2_o')
            v2_c = self.__dropout(v2, keep_prob, 'v2_c')
        else:
            # use the same mask for all gates
            h_i = self.__dropout(h, keep_prob, 'h')
            h_f = self.__dropout(h, keep_prob, 'h')
            h_o = self.__dropout(h, keep_prob, 'h')
            h_c = self.__dropout(h, keep_prob, 'h')

            x_i = self.__dropout(x, keep_prob, 'x')
            x_f = self.__dropout(x, keep_prob, 'x')
            x_o = self.__dropout(x, keep_prob, 'x')
            x_c = self.__dropout(x, keep_prob, 'x')

            m_i = self.__dropout(m, keep_prob, 'm')
            m_f = self.__dropout(m, keep_prob, 'm')
            m_o = self.__dropout(m, keep_prob, 'm')
            m_c = self.__dropout(m, keep_prob, 'm')

            v1_i = self.__dropout(v1, keep_prob, 'v1')
            v1_f = self.__dropout(v1, keep_prob, 'v1')
            v1_o = self.__dropout(v1, keep_prob, 'v1')
            v1_c = self.__dropout(v1, keep_prob, 'v1')

            v2_i = self.__dropout(v2, keep_prob, 'v2')
            v2_f = self.__dropout(v2, keep_prob, 'v2')
            v2_o = self.__dropout(v2, keep_prob, 'v2')
            v2_c = self.__dropout(v2, keep_prob, 'v2')

        temp1_i = ((v1 @ self.V_i_1) * (v2 @ self.V_i_2)) @ self.C_i_1
        temp1_f = ((v1 @ self.V_f_1) * (v2 @ self.V_f_2)) @ self.C_f_1
        temp1_o = ((v1 @ self.V_o_1) * (v2 @ self.V_o_2)) @ self.C_o_1
        temp1_c = ((v1 @ self.V_c_1) * (v2 @ self.V_c_2)) @ self.C_c_1

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
        i = self.__compute_gate(torch.sigmoid, temp1_i, self.temp2_i, temp3_i, self.temp4_i, temp5_i, temp6_i, self.W_i_3, self.C_i_3, self.U_i_3, self.b_i)
        f = self.__compute_gate(torch.sigmoid, temp1_f, self.temp2_f, temp3_f, self.temp4_f, temp5_f, temp6_f, self.W_f_3, self.C_f_3, self.U_f_3, self.b_f)
        o = self.__compute_gate(torch.sigmoid, temp1_o, self.temp2_o, temp3_o, self.temp4_o, temp5_o, temp6_o, self.W_o_3, self.C_o_3, self.U_o_3, self.b_o)
        c = self.__compute_gate(torch.tanh, temp1_c, self.temp2_c, temp3_c, self.temp4_c, temp5_c, temp6_c, self.W_c_3, self.C_c_3, self.U_c_3, self.b_c)

        # (batch_size x hidden_size)
        new_c = f * prev_c + i * c
        new_h = o * torch.tanh(prev_c)

        return new_h, new_c


class DenseCaptioner(nn.Module):
    def __init__(self, config, sem_tagger_config, syn_embedd_config, avscn_dec_config,
                 semsynan_dec_config, mm_config, vncl_cell_config, progs_vocab,
                 caps_vocab, pretrained_we, device):
        super(DenseCaptioner, self).__init__()

        self.train_sample_max = config.train_sample_max
        self.test_sample_max = config.test_sample_max
        self.max_clip_len = config.max_clip_len
        self.future_steps = config.future_steps
        self.h_size = config.h_size
        self.mm_size = mm_config.out_size
        self.progs_vocab_size = len(progs_vocab)
        self.caps_vocab_size = len(caps_vocab)

        self.mm_enc = MultiModal(v_enc_config=mm_config.v_enc_config,
                                 t_enc_config=mm_config.t_enc_config,
                                 out_size=mm_config.out_size,
                                 vocab_size=self.caps_vocab_size)

        self.rnn_cell = VNCLCell(x_size=self.progs_vocab_size,
                                 v_size=config.cnn_feats_size+config.c3d_feats_size,
                                 mm_size=mm_config.out_size,
                                 vh_size=vncl_cell_config.vh_size,
                                 h1_size=vncl_cell_config.h1_size,
                                 h2_size=config.h_size,
                                 drop_p=config.drop_p)

        self.fc = nn.Linear(in_features=config.h_size, out_features=self.progs_vocab_size)

        self.clip_captioner = Ensemble(v_size=config.cnn_feats_size+config.c3d_feats_size,
                                       sem_tagger_config=sem_tagger_config,
                                       syn_embedd_config=syn_embedd_config,
                                       avscn_dec_config=avscn_dec_config,
                                       semsynan_dec_config=semsynan_dec_config,
                                       caps_vocab=caps_vocab,
                                       pretrained_we=pretrained_we,
                                       device=device)

    def get_clip_feats(self, v_feats, start_idx, end_idx=None):
        feats = [torch.zeros(f.size(0), self.max_clip_len, f.size(2)).to(f.device) for f in v_feats]
        pool = []
        if end_idx is not None:
            for i, (s, e) in enumerate(zip(start_idx, end_idx)):
                if (e-s) > self.max_clip_len:
                    indices = torch.linspace(s, min(e, v_feats[0].size(1)-1), steps=self.max_clip_len, dtype=torch.long)
                    f1 = v_feats[0][i, indices, :]
                    f2 = v_feats[1][i, indices, :]
                else:
                    f1 = v_feats[0][i, s:e, :]
                    f2 = v_feats[1][i, s:e, :]

                feats[0][i,:f1.size(0)] = f1
                feats[1][i,:f2.size(0)] = f2

                pool.append(torch.cat((torch.mean(f1, dim=0), torch.mean(f2, dim=0))))

        return torch.stack(pool), feats

    def __step__(self, t, video_features):
        self.v_p_q_pool, self.v_p_q_feats = self.get_clip_feats(video_features, self.p, self.q)
        v_q_w_pool, _ = self.get_clip_feats(video_features, self.q, self.q + self.future_steps)

        self.h, self.c = self.rnn_cell(self.h, self.c, self.a_logits, self.prev_match, self.v_p_q_pool, v_q_w_pool, var_drop_p=.1)
        self.a_logits = self.fc(self.h)

    def forward(self, video_features, feats_count, teacher_forcing_p=.5, gt_program=None, gt_captions=None, gt_intervals=None):
        # initialize
        program, bs, device = [], video_features[0].size(0), video_features[0].device
        program, captions, intervals = torch.zeros_like(gt_program), torch.zeros_like(gt_captions), torch.zeros_like(gt_intervals, dtype=torch.float)
        prog_logits = torch.zeros(program.size(0), program.size(1), self.progs_vocab_size).to(device)
        caps_logits = torch.zeros(captions.size(0), captions.size(1), captions.size(2), self.caps_vocab_size).to(device)
        caps_count = torch.zeros(bs, dtype=torch.int8)
        self.p, self.q, self.a_logits = torch.zeros(bs, dtype=torch.int), torch.ones(bs, dtype=torch.int), torch.zeros(bs, self.progs_vocab_size).fill_(-1).to(device)
        self.h, self.c, self.prev_match = torch.zeros(bs, self.h_size).to(device), torch.zeros(bs, self.h_size).to(device), torch.zeros(bs, self.mm_size).to(device)

        # precomputing weights related to the prev_match only
        self.rnn_cell.precompute_dots_4_m(self.prev_match, var_drop_p=.1)

        seq_pos, max_len = 0, gt_program.size(1)
        while seq_pos < max_len:
            use_teacher_forcing = True if random.random() < teacher_forcing_p or seq_pos == 0 else False
            self.__step__(seq_pos, video_features)

            if self.training:
                if use_teacher_forcing:
                    # use the correct instructions,
                    # (batch_size)
                    a_id = gt_program[:, seq_pos]
                elif self.train_sample_max:
                    # select the instruction ids with the max probability,
                    # (batch_size)
                    a_id = self.a_logits.max(1)[1]
                else:
                    # sample instructions from probability distribution
                    # (batch_size)
                    a_id = torch.multinomial(torch.softmax(self.a_logits, dim=1), 1).squeeze(1)
            elif self.test_sample_max:
                # in testing phase select the instruction ids with the max probability,
                # (batch_size)
                a_id = self.a_logits.max(1)[1]
            else:
                # in testing phase sample instructions from probability distribution
                # (batch_size)
                a_id = torch.multinomial(torch.softmax(self.a_logits, dim=1), 1)

            program[:, seq_pos] = a_id
            prog_logits[:, seq_pos, :] = self.a_logits

            # updates the p and q positions for each video, and save sub-batch of video clips to be described
            intervals_to_describe, vidx_to_describe = [], []
            for i, a in enumerate(a_id):
                if a == 2:
                    # skip
                    self.p[i] += 1
                    self.q[i] = self.p[i] + 1
                elif a == 3:
                    # enqueue
                    self.q[i] += 1
                elif a == 4 and caps_count[i] < intervals.size(1):
                    # generate, save interval to describe for constructiong a captioning sub-batch
                    vidx_to_describe.append(i)
                    intervals[i, caps_count[i], :] = torch.tensor([self.p[i], self.q[i]])

            # generate a caption from the current video-clips saved in the sub-batch
            if len(vidx_to_describe) > 0:
                print(seq_pos, use_teacher_forcing, teacher_forcing_p, vidx_to_describe, caps_count, self.p.data, self.q.data)
                # get sub-batch of video features to be described
                clip_feats = [feats[vidx_to_describe, :, :] for feats in self.v_p_q_feats]
                clip_global = self.v_p_q_pool[vidx_to_describe, :]

                # TODO: get ground-truth captions according to the position of p and q and the interval associated to each gt caption

                # get ground-truth captions according to the number of captions that have been generated per video
                gt = torch.stack([gt_captions[i][min(gt_captions.size(1)-1, caps_count[i])] for i in vidx_to_describe])

                # generate captions
                cap_logits = self.clip_captioner(clip_feats, clip_global, teacher_forcing_p, gt)

                if self.training:
                    use_teacher_forcing = True if random.random() < teacher_forcing_p or seq_pos == 0 else False
                    if use_teacher_forcing:
                        cap = gt
                    elif self.train_sample_max:
                        # select the words ids with the max probability,
                        # (sub-batch_size x max-cap-len)
                        cap = cap_logits.max(2)[1]
                    else:
                        # sample words from probability distribution
                        # (sub-batch_size*max-cap-len x caps_vocab_size)
                        cap = cap_logits.view(-1, self.caps_vocab_size)
                        # (sub-batch_size*max-cap-len)
                        cap = torch.multinomial(torch.softmax(cap, dim=1), 1).squeeze(1)
                        # (sub-batch_size x max-cap-len)
                        cap = cap.view(cap_logits.size(0), cap_logits.size(1))
                elif self.test_sample_max:
                    # select the words ids with the max probability,
                    # (sub-batch_size x max-cap-len)
                    cap = cap_logits.max(2)[1]
                else:
                    # sample words from probability distribution
                    # (sub-batch_size*max-cap-len x caps_vocab_size)
                    cap = cap_logits.view(-1, self.caps_vocab_size)
                    # (sub-batch_size*max-cap-len)
                    cap = torch.multinomial(torch.softmax(cap, dim=1), 1).squeeze(1)
                    # (sub-batch_size x max-cap-len)
                    cap = cap.view(cap_logits.size(0), cap_logits.size(1))

                # TODO: sort visual and textual information according to the caption's length

                # TEMP: setting the same len for all captions (the maximum possible len)
                cap_len = torch.IntTensor(cap.size(0)).fill_(cap.size(1))

                # compute caption's bow, the make_bow_vector can also compute the caption len
                cap_bow = torch.stack([make_bow_vector(c, self.caps_vocab_size) for c in cap]).to(device)

                # compute the multimodal representation
                match = self.mm_enc(clip_feats, clip_global, cap, cap_len, cap_bow)

                # save captions in the list of each video that was described in this step
                self.prev_match = torch.clone(self.prev_match)
                for i, c, c_logits, m in zip(vidx_to_describe, cap, cap_logits, match):
                    captions[i, caps_count[i], :] = c
                    caps_logits[i, caps_count[i], :, :] = c_logits
                    caps_count[i] += 1
                    self.prev_match[i, :] = m

                # reset rnn_cel weights, precomputing weights related to the prev_match only
                self.rnn_cell.precompute_dots_4_m(self.prev_match, var_drop_p=.1)

            seq_pos += 1

        return prog_logits, program, caps_logits, captions, intervals, caps_count
