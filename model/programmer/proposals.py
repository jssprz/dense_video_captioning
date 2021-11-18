import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from .attention import Attention
from ..tagging.semantic import TaggerMLP


class ProposalsEncoder(nn.Module):
    def __init__(
        self, config, proposals_tagger_config, num_proposals,
    ):
        super(ProposalsEncoder, self).__init__()

        self.future_steps = config.future_steps
        self.prop_rnn_h_size = proposals_tagger_config.rnn_h_size

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

        self.prop_enc_s = TaggerMLP(
            v_size=proposals_tagger_config.rnn_h_size * 2,
            out_size=num_proposals,
            h_sizes=proposals_tagger_config.mapping_h_sizes,
            in_drop_p=proposals_tagger_config.mapping_in_drop_p,
            drop_ps=proposals_tagger_config.mapping_drop_ps,
            have_last_bn=proposals_tagger_config.have_last_bn,
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

        self.prop_enc_e = TaggerMLP(
            v_size=proposals_tagger_config.rnn_h_size * 2,
            out_size=num_proposals,
            h_sizes=proposals_tagger_config.mapping_h_sizes,
            in_drop_p=proposals_tagger_config.mapping_in_drop_p,
            drop_ps=proposals_tagger_config.mapping_drop_ps,
            have_last_bn=proposals_tagger_config.have_last_bn,
        )

    def reset_internals(self, v_feats):
        bs, self.device = v_feats.size(0), v_feats.device

        # initialize the visual features of each pointer, averaging the first frames
        v_p = v_feats[:, : self.future_steps, :].mean(dim=1)
        v_q = v_feats[:, 1 : 1 + self.future_steps, :].mean(dim=1)

        # list of indices to be used for fill-in the first step of recurrencies
        idxs = list(range(bs))

        # INITIALIZE TENSORS OF START PROPOSALS MODULE
        # initialize the hidden and cell state of first layer (forward) of START MODULE
        self.prop_s_h_0 = torch.zeros(bs, self.prop_rnn_h_size).to(self.device)
        self.prop_s_c_0 = torch.zeros(bs, self.prop_rnn_h_size).to(self.device)

        # compute first step of first layer (forward) of START MODULE from the v_p features
        self.prop_s_h_0[idxs, :], self.prop_s_c_0[idxs, :] = self.prop_s_rnn_0(
            v_p, (self.prop_s_h_0[idxs, :], self.prop_s_c_0[idxs, :])
        )

        # initialize the hidden states of recurrent layer to compute backward of START MODULE
        self.prop_s_back_h_0 = torch.zeros(bs, self.prop_rnn_h_size).to(self.device)

        # initialize the hidden and cell state of second layer (forward) of START MODULE
        self.prop_s_h_1 = torch.zeros(bs, self.prop_rnn_h_size).to(self.device)
        self.prop_s_c_1 = torch.zeros(bs, self.prop_rnn_h_size).to(self.device)

        # INITIALIZE TENSORS OF END PROPOSALS MODULE
        # initialize the hidden and cell state of first layer (forward) of END MODULE
        self.prop_e_h_0 = torch.zeros(bs, self.prop_rnn_h_size).to(self.device)
        self.prop_e_c_0 = torch.zeros(bs, self.prop_rnn_h_size).to(self.device)

        # compute first two steps of first layer (forward) of END MODULE from the v_p and v_q features, respectively
        self.prop_e_h_0[idxs, :], self.prop_e_c_0[idxs, :] = self.prop_e_rnn_0(
            v_p, (self.prop_e_h_0[idxs, :], self.prop_e_c_0[idxs, :])
        )
        self.prop_e_h_0[idxs, :], self.prop_e_c_0[idxs, :] = self.prop_e_rnn_0(
            v_q, (self.prop_e_h_0[idxs, :], self.prop_e_c_0[idxs, :])
        )

        # set the greater positions reached by the first layer (forward) of END MODULE similar to self.q positions
        # this tensor is used to avoid the computation of first layar step over the same visual features
        self.prop_e_0_pos = torch.ones(bs, dtype=torch.long).to(self.device)

        # initialize the hidden states of recurrent layer to compute backward of END MODULE
        self.prop_e_back_h_0 = torch.zeros(bs, self.prop_rnn_h_size).to(self.device)

        # initialize the hidden and cell state of second layer (forward) of END MODULE
        self.prop_e_h_1 = torch.zeros(bs, self.prop_rnn_h_size).to(self.device)
        self.prop_e_c_1 = torch.zeros(bs, self.prop_rnn_h_size).to(self.device)

    def excecute_instruction(self, a_id, p, q):
        # get idxs of videos that require the pointers be skipped
        vix_2_skip = (a_id == 0).nonzero(as_tuple=True)[0]
        # move pointers according to the skip instruction
        p[vix_2_skip] += 1
        q[vix_2_skip] = p[vix_2_skip] + 1

        # get idxs of videos that require new visual features be enqueued
        vix_2_adv = (a_id == 1).nonzero(as_tuple=True)[0]
        # move the q pointers according to the equeue instruction
        q[vix_2_adv] += 1

        # refine the idxs of videos to be advanced according to the greated values stored in prop_e_0_pos.
        # Here we also consider the vix_2_skip idxs because skip instuction can produce a greater q than the position stored in prop_e_0_pos
        vix_2_adv = torch.tensor(
            list(
                set(
                    vix_2_skip[(q[vix_2_skip] > self.prop_e_0_pos[vix_2_skip]).nonzero(as_tuple=True)[0]].tolist()
                    + vix_2_adv[(q[vix_2_adv] > self.prop_e_0_pos[vix_2_adv]).nonzero(as_tuple=True)[0]].tolist()
                )
            ),
            dtype=torch.long,
        )

        # get idxs of videos that require a new caption be generated
        vix_2_dscr = (a_id == 2).nonzero(as_tuple=True)[0]

        return vix_2_skip, vix_2_adv, vix_2_dscr

    def prop_s_rnn_0_step(self, vix_2_skip, p, v_feats, feats_count):
        # determine the initial and end frames of windows used to encode the visual features related to pointer p
        fidx_from = torch.min(p[vix_2_skip], feats_count[vix_2_skip] - self.future_steps)
        fidx_to = torch.min(p[vix_2_skip] + self.future_steps, feats_count[vix_2_skip])

        # tensor to be filled according to the number of frames in windows
        v_p = torch.zeros(len(vix_2_skip), v_feats.size(-1)).to(self.device)

        # videos with windows that have at least the minimum num of frames required to compute the attention
        attn_mask = (fidx_from >= 0).nonzero(as_tuple=True)[0]
        if len(attn_mask):
            vix_2_attn = vix_2_skip[attn_mask]
            v_p[attn_mask] = self.v_p_attn(
                torch.stack(
                    [
                        v_feats[vix, ff:ft, :]
                        for vix, ff, ft in zip(vix_2_attn, fidx_from[attn_mask], fidx_to[attn_mask])
                    ]
                ),
                self.prop_s_h_0[vix_2_attn, :],
            )

        # videos with windows that have less than the minimum num of frames required to compute the attention
        fill_mask = (fidx_from < 0).nonzero(as_tuple=True)[0]
        if len(fill_mask):
            vix_2_fill = vix_2_skip[fill_mask]
            v_p[fill_mask] = v_feats[vix_2_fill, torch.min(p[vix_2_fill], feats_count[vix_2_fill]), :]

        # compute a recurrent step of the first layer (forward) of START MODULE
        self.prop_s_h_0[vix_2_skip, :], self.prop_s_c_0[vix_2_skip, :] = self.prop_s_rnn_0(
            v_p, (self.prop_s_h_0[vix_2_skip, :], self.prop_s_c_0[vix_2_skip, :])
        )

    def prop_s_back_rnn_0_enc(self, vix_2_dscr, p, v_feats, feats_count, max_back_steps):
        # compute the recurrency on frames previous to pointer p in backward direction
        ends = torch.min(p[vix_2_dscr] + 1, feats_count[vix_2_dscr])
        starts = (ends - max_back_steps).clamp(min=0)

        sub_v_feats_padded = pad_sequence(
            [v_feats[v, s:e, :].flip((0,)) for v, s, e in zip(vix_2_dscr, starts, ends)], batch_first=True,
        )
        sub_v_feats_packed = pack_padded_sequence(
            input=sub_v_feats_padded, lengths=(ends - starts).to("cpu"), batch_first=True, enforce_sorted=False,
        )
        _, (self.prop_s_back_h_0[vix_2_dscr], _) = self.prop_s_back_rnn_0(sub_v_feats_packed)

    def prop_s_rnn_1_step(self, vix_2_dscr):
        # compute another step of the second layer (forward) of START MOUDULE
        (self.prop_s_h_1[vix_2_dscr, :], self.prop_s_c_1[vix_2_dscr, :],) = self.prop_s_rnn_1(
            self.prop_s_h_0[vix_2_dscr, :], (self.prop_s_h_1[vix_2_dscr, :], self.prop_s_c_1[vix_2_dscr, :],),
        )

    def prop_e_rnn_0_step(self, vix_2_adv, q, v_feats, feats_count):
        # update the greatest positions stored in prop_e_0_pos
        self.prop_e_0_pos[vix_2_adv] = q[vix_2_adv]

        # determine the initial and end frames of windows used to encode the visual features related to pointer q
        fidx_from = torch.min(q[vix_2_adv], feats_count[vix_2_adv] - self.future_steps)
        fidx_to = torch.min(q[vix_2_adv] + self.future_steps, feats_count[vix_2_adv])

        # tensor to be filled according to the number of frames in windows
        v_q = torch.zeros(len(vix_2_adv), v_feats.size(-1)).to(self.device)

        # videos with windows that have at least the minimum num of frames required to compute the attention
        attn_mask = (fidx_from >= 0).nonzero(as_tuple=True)[0]
        if len(attn_mask):
            vix_2_attn = vix_2_adv[attn_mask]
            v_q[attn_mask] = self.v_q_attn(
                torch.stack(
                    [
                        v_feats[vix, ff:ft, :]
                        for vix, ff, ft in zip(vix_2_attn, fidx_from[attn_mask], fidx_to[attn_mask])
                    ]
                ),
                self.prop_e_h_0[vix_2_attn, :],
            )

        # videos with windows that have less than the minimum num of frames required to compute the attention
        fill_mask = (fidx_from < 0).nonzero(as_tuple=True)[0]
        if len(fill_mask):
            vix_2_fill = vix_2_adv[fill_mask]
            v_q[fill_mask] = v_feats[vix_2_fill, torch.min(q[vix_2_fill], feats_count[vix_2_fill]), :]

        # compute a recurrent step of the first layer (forward) of END MODULE
        self.prop_e_h_0[vix_2_adv, :], self.prop_e_c_0[vix_2_adv, :] = self.prop_e_rnn_0(
            v_q, (self.prop_e_h_0[vix_2_adv, :], self.prop_e_c_0[vix_2_adv, :])
        )

    def prop_e_back_rnn_0_enc(self, vix_2_dscr, p, q, v_feats, feats_count, max_back_steps):
        # compute the recurrency on frames previous to poiter q in backward direction
        max_fix = feats_count[vix_2_dscr] - 1
        ends = torch.min(q[vix_2_dscr], max_fix) + 1
        starts = torch.max(torch.min(p[vix_2_dscr], max_fix), ends - max_back_steps)

        sub_v_feats_padded = pad_sequence(
            [v_feats[v, s:e, :].flip((0,)) for v, s, e in zip(vix_2_dscr, starts, ends)], batch_first=True,
        )
        sub_v_feats_packed = pack_padded_sequence(
            input=sub_v_feats_padded, lengths=(ends - starts).to("cpu"), batch_first=True, enforce_sorted=False,
        )
        _, (self.prop_e_back_h_0[vix_2_dscr], _) = self.prop_e_back_rnn_0(sub_v_feats_packed)

    def prop_e_rnn_1_step(self, vix_2_dscr):
        # compute another step of the second layer (forward) of END MOUDULE
        (self.prop_e_h_1[vix_2_dscr, :], self.prop_e_c_1[vix_2_dscr, :],) = self.prop_e_rnn_1(
            self.prop_e_h_0[vix_2_dscr, :], (self.prop_e_h_1[vix_2_dscr, :], self.prop_e_c_1[vix_2_dscr, :],),
        )

    def train_val_step(
        self, prop_logits_s, prop_logits_e, caps_count, p, q, a_id, v_feats, feats_count, max_back_steps=10
    ):
        vix_2_skip, vix_2_adv, vix_2_dscr = self.excecute_instruction(a_id, p, q)

        if len(vix_2_skip) > 0:
            # START MODULE (1st layer)
            self.prop_s_rnn_0_step(vix_2_skip, p, v_feats, feats_count)

        if len(vix_2_adv) > 0:
            # END MODULE (1st layer)
            self.prop_e_rnn_0_step(vix_2_adv, q, v_feats, feats_count)

        if len(vix_2_dscr) > 0:
            # START MODULE (back direction)
            self.prop_s_back_rnn_0_enc(vix_2_dscr, p, v_feats, feats_count, max_back_steps)
            # START MODULE (2nd layer)
            self.prop_s_rnn_1_step(vix_2_dscr)

            # START MODULE (proposals) from the 2nd layer (forward) and the back direction on previous frames
            prop_logits_s[vix_2_dscr, caps_count[vix_2_dscr], :] = self.prop_enc_s(
                torch.cat((self.prop_s_h_1[vix_2_dscr], self.prop_s_back_h_0[vix_2_dscr]), dim=-1)
            )[0]

            # END MODULE (back direction)
            self.prop_e_back_rnn_0_enc(vix_2_dscr, p, q, v_feats, feats_count, max_back_steps)
            # END MODULE (2nd layer)
            self.prop_e_rnn_1_step(vix_2_dscr)

            # END MODULE (proposals) from the 2nd layer (forward) and the back direction on previous frames
            prop_logits_e[vix_2_dscr, caps_count[vix_2_dscr], :] = self.prop_enc_e(
                torch.cat((self.prop_e_h_1[vix_2_dscr], self.prop_e_back_h_0[vix_2_dscr]), dim=-1)
            )[0]

            caps_count[vix_2_dscr] += 1

    def dense_step(self, a_id, p, q, v_feats, feats_count, max_back_steps=10):
        """ Compute logits always
        """
        vix_2_skip, vix_2_adv, _ = self.excecute_instruction(a_id, p, q)

        if len(vix_2_skip) > 0:
            # START MODULE (1st layer)
            self.prop_s_rnn_0_step(vix_2_skip, p, v_feats, feats_count)

        if len(vix_2_adv) > 0:
            # END MODULE (1st layer)
            self.prop_e_rnn_0_step(vix_2_adv, q, v_feats, feats_count)

        # START MODULE (back direction)
        vix_2_enc = torch.arange(v_feats.size(0))
        self.prop_s_back_rnn_0_enc(vix_2_enc, p, v_feats, feats_count, max_back_steps)
        # START MODULE (2nd layer)
        self.prop_s_rnn_1_step(vix_2_enc)

        # START MODULE (proposals) from the 2nd layer (forward) and the back direction on previous frames
        # prop_logits_s = self.prop_enc_s(
        #     torch.cat((self.prop_s_h_1[vix_2_enc], self.prop_s_back_h_0[vix_2_enc]), dim=-1)
        # )[0]

        # END MODULE (back direction)
        self.prop_e_back_rnn_0_enc(vix_2_enc, p, q, v_feats, feats_count, max_back_steps)
        # END MODULE (2nd layer)
        self.prop_e_rnn_1_step(vix_2_enc)

        # END MODULE (proposals) from the 2nd layer (forward) and the back direction on previous frames
        # prop_logits_e = self.prop_enc_e(
        #     torch.cat((self.prop_e_h_1[vix_2_enc], self.prop_e_back_h_0[vix_2_enc]), dim=-1)
        # )[0]

        # return prop_logits_s, prop_logits_e
        return self.prop_s_h_1[vix_2_enc], self.prop_e_h_1[vix_2_enc]

    def forward():
        pass
