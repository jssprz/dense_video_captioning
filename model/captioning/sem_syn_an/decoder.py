import random

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from utils import get_init_weights
from model.captioning.attention import Attention


class SCNDecoder(nn.Module):
    def __init__(self, in_seq_length, n_feats, n_tags, embedding_size, h_size, rnn_in_size, rnn_h_size, vocab, encoder_num_layers, encoder_bidirectional, 
                 pretrained_we=None, rnn_cell='gru', num_layers=1, drop_p=0.5, beam_size=10, temperature=1.0, train_sample_max=False, test_sample_max=True, beam_search_logic='bfs', have_bn=False, var_dropout='per-gate'):
        super(SCNDecoder, self).__init__()
        self.h_size = h_size
        self.embedding_size = embedding_size
        self.output_size = len(vocab)
        self.in_seq_length = in_seq_length
        self.vocab = vocab
        self.beam_size = beam_size
        self.temperature = temperature
        self.train_sample_max = train_sample_max
        self.test_sample_max = test_sample_max
        self.beam_search_logic = beam_search_logic
        self.drop_p = drop_p
        self.var_dropout = var_dropout

        self.num_layers = num_layers
        self.encoder_num_layers = encoder_num_layers
        self.encoder_num_directions = 2 if encoder_bidirectional else 1
        self.num_directions = 1  # beause this decoder is not bidirectional

        # Components
        if pretrained_we is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_we)
        else:
            self.embedding = nn.Embedding(self.output_size, embedding_size)

        self.Wa_i = get_init_weights((rnn_in_size, rnn_h_size))
        self.Wa_f = get_init_weights((rnn_in_size, rnn_h_size))
        self.Wa_o = get_init_weights((rnn_in_size, rnn_h_size))
        self.Wa_c = get_init_weights((rnn_in_size, rnn_h_size))

        self.Wb_i = get_init_weights((n_tags, rnn_h_size))
        self.Wb_f = get_init_weights((n_tags, rnn_h_size))
        self.Wb_o = get_init_weights((n_tags, rnn_h_size))
        self.Wb_c = get_init_weights((n_tags, rnn_h_size))

        self.Wc_i = get_init_weights((rnn_h_size, h_size))
        self.Wc_f = get_init_weights((rnn_h_size, h_size))
        self.Wc_o = get_init_weights((rnn_h_size, h_size))
        self.Wc_c = get_init_weights((rnn_h_size, h_size))

        self.Ua_i = get_init_weights((h_size, rnn_h_size))
        self.Ua_f = get_init_weights((h_size, rnn_h_size))
        self.Ua_o = get_init_weights((h_size, rnn_h_size))
        self.Ua_c = get_init_weights((h_size, rnn_h_size))

        self.Ub_i = get_init_weights((n_tags, rnn_h_size))
        self.Ub_f = get_init_weights((n_tags, rnn_h_size))
        self.Ub_o = get_init_weights((n_tags, rnn_h_size))
        self.Ub_c = get_init_weights((n_tags, rnn_h_size))

        self.Uc_i = get_init_weights((rnn_h_size, h_size))
        self.Uc_f = get_init_weights((rnn_h_size, h_size))
        self.Uc_o = get_init_weights((rnn_h_size, h_size))
        self.Uc_c = get_init_weights((rnn_h_size, h_size))

        self.Ca_i = get_init_weights((n_feats, rnn_h_size))
        self.Ca_f = get_init_weights((n_feats, rnn_h_size))
        self.Ca_o = get_init_weights((n_feats, rnn_h_size))
        self.Ca_c = get_init_weights((n_feats, rnn_h_size))

        self.Cb_i = get_init_weights((n_tags, rnn_h_size))
        self.Cb_f = get_init_weights((n_tags, rnn_h_size))
        self.Cb_o = get_init_weights((n_tags, rnn_h_size))
        self.Cb_c = get_init_weights((n_tags, rnn_h_size))

        self.Cc_i = get_init_weights((rnn_h_size, h_size))
        self.Cc_f = get_init_weights((rnn_h_size, h_size))
        self.Cc_o = get_init_weights((rnn_h_size, h_size))
        self.Cc_c = get_init_weights((rnn_h_size, h_size))

        self.b_i = Parameter(torch.zeros(h_size))
        self.b_f = Parameter(torch.zeros(h_size))
        self.b_o = Parameter(torch.zeros(h_size))
        self.b_c = Parameter(torch.zeros(h_size))

        self.out = nn.Linear(self.h_size * self.num_directions, self.output_size)

        self.have_bn = have_bn
        if have_bn:
            self.bn = nn.LayerNorm(h_size)

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

    def precompute_mats(self, v, s, var_drop_p):
        self.dropM = {}

        keep_prob = 1 - var_drop_p
        if self.var_dropout == 'per-gate':
            # use a distinct mask for each gate
            s_i = self.__dropout(s, keep_prob, 's_i')
            s_f = self.__dropout(s, keep_prob, 's_f')
            s_o = self.__dropout(s, keep_prob, 's_o')
            s_c = self.__dropout(s, keep_prob, 's_c')

            v_i = self.__dropout(v, keep_prob, 'v_i')
            v_f = self.__dropout(v, keep_prob, 'v_f')
            v_o = self.__dropout(v, keep_prob, 'v_o')
            v_c = self.__dropout(v, keep_prob, 'v_c')
        else:
            # use the same mask for all gates
            s_i = self.__dropout(s, keep_prob, 's')
            s_f = self.__dropout(s, keep_prob, 's')
            s_o = self.__dropout(s, keep_prob, 's')
            s_c = self.__dropout(s, keep_prob, 's')

            v_i = self.__dropout(v, keep_prob, 'v')
            v_f = self.__dropout(v, keep_prob, 'v')
            v_o = self.__dropout(v, keep_prob, 'v')
            v_c = self.__dropout(v, keep_prob, 'v')

        # (batch_size x rnn_h_size)
        self.temp2_i = s_i @ self.Wb_i
        self.temp2_f = s_f @ self.Wb_f
        self.temp2_o = s_o @ self.Wb_o
        self.temp2_c = s_c @ self.Wb_c

        # (batch_size x rnn_h_size)
        self.temp3_i = v_i @ self.Ca_i
        self.temp3_f = v_f @ self.Ca_f
        self.temp3_o = v_o @ self.Ca_o
        self.temp3_c = v_c @ self.Ca_c

        # (batch_size x rnn_h_size)
        self.temp4_i = s_i @ self.Cb_i
        self.temp4_f = s_f @ self.Cb_f
        self.temp4_o = s_o @ self.Cb_o
        self.temp4_c = s_c @ self.Cb_c

    def __compute_gate(self, activation, temp1, temp2, temp3, temp4, temp5, temp6, Wc, Cc, Uc, b):
        x = (temp1 * temp2) @ Wc
        v = (temp3 * temp4) @ Cc
        h = (temp5 * temp6) @ Uc

        assert torch.all(torch.tensor([x.device == t.device for t in [v,h,b]])), 'all tensor must be in the same device ({}, {}, {}, {})'.format(x.device, v.device, h.device, b.device)

        logits = x + v + h + b

        if self.have_bn:
            logits = self.bn(logits)

        return activation(logits)

    def step(self, s, rnn_h, rnn_c, decoder_input, var_drop_p):
        keep_prob = 1 - var_drop_p
        if self.var_dropout == 'per-gate':
            # use a distinct mask for each gate
            s_i = self.__dropout(s, keep_prob, 's_i')
            s_f = self.__dropout(s, keep_prob, 's_f')
            s_o = self.__dropout(s, keep_prob, 's_o')
            s_c = self.__dropout(s, keep_prob, 's_c')

            x_i = self.__dropout(decoder_input, keep_prob, 'x_i')
            x_f = self.__dropout(decoder_input, keep_prob, 'x_f')
            x_o = self.__dropout(decoder_input, keep_prob, 'x_o')
            x_c = self.__dropout(decoder_input, keep_prob, 'x_c')

            h_i = self.__dropout(rnn_h, keep_prob, 'h_i')
            h_f = self.__dropout(rnn_h, keep_prob, 'h_f')
            h_o = self.__dropout(rnn_h, keep_prob, 'h_o')
            h_c = self.__dropout(rnn_h, keep_prob, 'h_c')
        else:
            # use the same mask for all gates
            s_i = self.__dropout(s, keep_prob, 's')
            s_f = self.__dropout(s, keep_prob, 's')
            s_o = self.__dropout(s, keep_prob, 's')
            s_c = self.__dropout(s, keep_prob, 's')

            x_i = self.__dropout(decoder_input, keep_prob, 'x')
            x_f = self.__dropout(decoder_input, keep_prob, 'x')
            x_o = self.__dropout(decoder_input, keep_prob, 'x')
            x_c = self.__dropout(decoder_input, keep_prob, 'x')

            h_i = self.__dropout(rnn_h, keep_prob, 'h')
            h_f = self.__dropout(rnn_h, keep_prob, 'h')
            h_o = self.__dropout(rnn_h, keep_prob, 'h')
            h_c = self.__dropout(rnn_h, keep_prob, 'h')

        # (batch_size x rnn_h_size)
        temp1_i = x_i @ self.Wa_i
        temp1_f = x_f @ self.Wa_f
        temp1_o = x_o @ self.Wa_o
        temp1_c = x_c @ self.Wa_c

        # (batch_size x rnn_h_size)
        temp5_i = s_i @ self.Ub_i
        temp5_f = s_f @ self.Ub_f
        temp5_o = s_o @ self.Ub_o
        temp5_c = s_c @ self.Ub_c

        # (batch_size x rnn_h_size)
        temp6_i = h_i @ self.Ua_i
        temp6_f = h_f @ self.Ua_f
        temp6_o = h_o @ self.Ua_o
        temp6_c = h_c @ self.Ua_c

        # (batch_size x h_size)
        i = self.__compute_gate(torch.sigmoid, temp1_i, self.temp2_i, self.temp3_i, self.temp4_i, temp5_i, temp6_i, self.Wc_i, self.Cc_i, self.Uc_i, self.b_i)
        f = self.__compute_gate(torch.sigmoid, temp1_f, self.temp2_f, self.temp3_f, self.temp4_f, temp5_f, temp6_f, self.Wc_f, self.Cc_f, self.Uc_f, self.b_f)
        o = self.__compute_gate(torch.sigmoid, temp1_o, self.temp2_o, self.temp3_o, self.temp4_o, temp5_o, temp6_o, self.Wc_o, self.Cc_o, self.Uc_o, self.b_o)
        c = self.__compute_gate(torch.tanh, temp1_c, self.temp2_c, self.temp3_c, self.temp4_c, temp5_c, temp6_c, self.Wc_c, self.Cc_c, self.Uc_c, self.b_c)

        # (batch_size x h_size)
        rnn_c = f * rnn_c + i * c
        rnn_h = o * torch.tanh(rnn_c)

        return rnn_h, rnn_c

    def forward_fn(self, v_pool, s_tags, encoder_hidden, encoder_outputs, captions, teacher_forcing_p=0.5):
        batch_size = encoder_outputs.size(0)

        # (batch_size x 1)
        # decoder_input = Variable(torch.LongTensor(batch_size, 1).fill_(self.vocab('<start>'))).to(v_pool.device)

        # (batch_size x embedding_size)
        decoder_input = Variable(torch.Tensor(batch_size, self.embedding_size).fill_(0)).to(v_pool.device)

        if type(encoder_hidden) is tuple:
            # (encoder_n_layers * encoder_num_directions x batch_size x h_size) -> (encoder_n_layers x encoder_num_directions x batch_size x h_size)
            rnn_h = encoder_hidden[0].view(self.encoder_num_layers, self.encoder_num_directions, batch_size, self.h_size)
            rnn_c = encoder_hidden[1].view(self.encoder_num_layers, self.encoder_num_directions, batch_size, self.h_size)

        # get h_n of forward direction of the last num_layers of encoder
        # (n_layers x batch_size x h_size)
        # rnn_h = torch.cat([rnn_h[-i,0,:,:].unsqueeze(0) for i in range(self.num_layers, 0, -1)], dim=0)

        rnn_h = Variable(torch.cat([rnn_h[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(v_pool.device)
        rnn_c = Variable(torch.cat([rnn_c[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(v_pool.device)

        # rnn_h = Variable(torch.zeros(batch_size, self.h_size)).to(v_pool.device)
        # rnn_c = Variable(torch.zeros(batch_size, self.h_size)).to(v_pool.device)

        outputs = []

        s = self.__dropout(s_tags, 1 - self.drop_p)
        v = self.__dropout(v_pool, 1 - self.drop_p)

        self.precompute_mats(v, s)

        if not self.training:
            words = []
            for step in range(self.out_seq_length):
                rnn_h, rnn_c = self.step(s, rnn_h, rnn_c, decoder_input, encoder_hidden, encoder_outputs)

                # compute word_logits
                # (batch_size x output_size)
                word_logits = self.out(rnn_h)

                # compute word probs
                if self.test_sample_max:
                    # sample max probailities
                    # (batch_size x 1), (batch_size x 1)
                    word_id = word_logits.max(dim=1)[1]
                else:
                    # sample from distribution
                    # (batch_size x 1)
                    word_id = torch.multinomial(torch.softmax(word_logits, dim=1), 1)

                # (batch_size x 1) -> (batch_size x embedding_size)
                decoder_input = self.embedding(word_id).squeeze(1)

                outputs.append(word_logits)
                words.append(word_id)

            return torch.cat([o.unsqueeze(1) for o in outputs], dim=1).contiguous(), torch.cat([w.unsqueeze(1) for w in words], dim=1).contiguous()
        else:
            for seq_pos in range(self.out_seq_length):
                rnn_h, rnn_c = self.step(s, rnn_h, rnn_c, decoder_input, encoder_hidden, encoder_outputs)

                # compute word_logits
                # (batch_size x output_size)
                word_logits = self.out(rnn_h)

                use_teacher_forcing = True if random.random() < teacher_forcing_p or seq_pos == 0 else False
                if use_teacher_forcing:
                    decoder_input = captions[:, seq_pos]  # use the correct words, (batch_size x 1)
                elif self.train_sample_max:
                    # select the words ids with the max probability,
                    # (batch_size x 1)
                    decoder_input = word_logits.max(1)[1]
                else:
                    # sample words from probability distribution
                    # (batch_size x 1)
                    decoder_input = torch.multinomial(torch.softmax(word_logits, dim=1), 1)

                # (batch_size x 1) -> (batch_size x embedding_size)
                decoder_input = self.embedding(decoder_input).squeeze(1)
                decoder_input = self.__dropout(decoder_input, 1 - self.drop_p)

                outputs.append(word_logits)

            # (batch_size x out_seq_length x output_size), none
            return torch.cat([o.unsqueeze(1) for o in outputs], dim=1).contiguous(), None

    def forward(self, videos_encodes, teacher_forcing_p=.5, gt_captions=None):
        return self.forward_fn(v_pool=videos_encodes[3],
                               s_tags=videos_encodes[2],
                               encoder_hidden=videos_encodes[1],
                               encoder_outputs=videos_encodes[0],
                               captions=captions,
                               teacher_forcing_p=teacher_forcing_p)

    def sample(self, videos_encodes):
        return self.forward(videos_encodes, None, teacher_forcing_p=0.0)


class SemSynANDecoder(nn.Module):
    def __init__(self, config, vocab, pretrained_we=None, device='gpu', dataset_name='MSVD'):
        super(SemSynANDecoder, self).__init__()

        self.h_size = config.h_size
        self.embedding_size = config.embedding_size
        self.output_size = len(vocab)
        self.in_seq_length = config.in_seq_length
        self.vocab = vocab
        self.device = device
        self.beam_size = config.beam_size
        self.temperature = config.temperature
        self.train_sample_max = config.train_sample_max
        self.test_sample_max = config.test_sample_max
        self.beam_search_logic = config.beam_search_logic

        self.num_layers = config.num_layers
        self.encoder_num_layers = config.encoder_num_layers
        self.encoder_num_directions = 2 if config.encoder_bidirectional else 1
        self.num_directions = 1  # because this decoder is not bidirectional

        # Components

        if pretrained_we is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_we)
        else:
            self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.embedd_drop = nn.Dropout(config.drop_p)

        self.v_sem_layer = SCNDecoder(config.in_seq_length,
                                      config.n_feats,
                                      config.n_tags,
                                      config.embedding_size,
                                      config.h_size,
                                      config.rnn_in_size,
                                      config.rnn_h_size,
                                      vocab,
                                      config.encoder_num_layers,
                                      config.encoder_bidirectional,
                                      pretrained_we,
                                      config.rnn_cell,
                                      config.num_layers,
                                      config.drop_p,
                                      config.beam_size,
                                      config.temperature,
                                      config.train_sample_max,
                                      config.test_sample_max,
                                      config.beam_search_logic,
                                      have_bn=False)

        self.v_syn_layer = SCNDecoder(config.in_seq_length,
                                      config.n_feats,
                                      config.posemb_size,
                                      config.embedding_size,
                                      config.h_size,
                                      config.rnn_in_size,
                                      config.rnn_h_size,
                                      vocab,
                                      config.encoder_num_layers,
                                      config.encoder_bidirectional,
                                      pretrained_we,
                                      config.rnn_cell,
                                      config.num_layers,
                                      config.drop_p,
                                      config.beam_size,
                                      config.temperature,
                                      config.train_sample_max,
                                      config.test_sample_max,
                                      config.beam_search_logic,
                                      have_bn=False)

        self.se_sy_layer = SCNDecoder(config.in_seq_length,
                                      config.n_tags,
                                      config.posemb_size,
                                      config.embedding_size,
                                      config.h_size,
                                      config.rnn_in_size,
                                      config.rnn_h_size,
                                      vocab,
                                      config.encoder_num_layers,
                                      config.encoder_bidirectional,
                                      pretrained_we,
                                      config.rnn_cell,
                                      config.num_layers,
                                      config.drop_p,
                                      config.beam_size,
                                      config.temperature,
                                      config.train_sample_max,
                                      config.test_sample_max,
                                      config.beam_search_logic,
                                      have_bn=False)

        self.merge1 = nn.Linear(self.h_size + config.n_feats, self.h_size)
        self.merge2 = nn.Linear(self.h_size + config.n_feats, self.h_size)
        self.out = nn.Linear(self.h_size, self.output_size)

        self.dataset_name = dataset_name
        if dataset_name == 'MSVD':
            self.v_sem_attn = Attention(self.in_seq_length, self.embedding_size, self.h_size, self.num_layers, self.num_directions, mode='soft')
            self.v_syn_attn = Attention(self.in_seq_length, self.embedding_size, self.h_size, self.num_layers, self.num_directions, mode='soft')
            self.se_sy_attn = Attention(self.in_seq_length, self.embedding_size, self.h_size, self.num_layers, self.num_directions, mode='soft')
        elif dataset_name == 'MSR-VTT':
            self.v_attn = Attention(self.in_seq_length, self.embedding_size, self.h_size*3, self.num_layers, self.num_directions, mode='soft')
            self.s_attn = Attention(self.in_seq_length, self.embedding_size, self.h_size*3, self.num_layers, self.num_directions, mode='soft')

        self.__init_layers()

    def __init_layers(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

    def __adaptive_merge(self, rnn_h, v_attn, v_sem_h, v_syn_h, sem_syn_h):
        h = torch.cat((rnn_h, v_attn), dim=1)
        beta1 = torch.sigmoid(self.merge1(h))
        beta2 = torch.sigmoid(self.merge2(h))
        aa1 = beta1 * v_sem_h + (1 - beta1) * v_syn_h
        return beta2 * aa1 + (1 - beta2) * sem_syn_h

    def forward_fn(self, v_feats, v_pool, s_tags, pos_emb, teacher_forcing_p=.5, gt_captions=None, max_words=None):
        batch_size = v_pool.size(0)

        # (batch_size x embedding_size)
        decoder_input = torch.zeros(batch_size, self.embedding_size).to(self.device)
        # decoder_input = Variable(torch.Tensor(batch_size, self.embedding_size).fill_(0)).to(self.device)

        # if type(enc_hidden) is tuple:
        #     # (encoder_n_layers * encoder_num_directions x batch_size x h_size) -> (encoder_n_layers x encoder_num_directions x batch_size x h_size)
        #     rnn_h = enc_hidden[0].view(self.encoder_num_layers, self.encoder_num_directions, batch_size, self.h_size)
        #     rnn_c = enc_hidden[1].view(self.encoder_num_layers, self.encoder_num_directions, batch_size, self.h_size)

        # v_sem_h = Variable(torch.cat([rnn_h[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)
        # v_sem_c = Variable(torch.cat([rnn_c[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)

        # v_syn_h = Variable(torch.cat([rnn_h[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)
        # v_syn_c = Variable(torch.cat([rnn_c[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)

        # se_sy_h = Variable(torch.cat([rnn_h[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)
        # se_sy_c = Variable(torch.cat([rnn_c[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)

        # rnn_h = Variable(torch.cat([rnn_h[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)

        v_sem_h = torch.zeros(batch_size, self.h_size).to(self.device)
        v_sem_c = torch.zeros(batch_size, self.h_size).to(self.device)

        v_syn_h = torch.zeros(batch_size, self.h_size).to(self.device)
        v_syn_c = torch.zeros(batch_size, self.h_size).to(self.device)

        se_sy_h = torch.zeros(batch_size, self.h_size).to(self.device)
        se_sy_c = torch.zeros(batch_size, self.h_size).to(self.device)

        rnn_h = torch.zeros(batch_size, self.h_size).to(self.device)

        outputs, embedds = [], []

        self.v_sem_layer.precompute_mats(v_pool, s_tags, var_drop_p=0.1)
        self.v_syn_layer.precompute_mats(v_pool, pos_emb, var_drop_p=0.1)
        self.se_sy_layer.precompute_mats(s_tags, pos_emb, var_drop_p=0.1)

        if not self.training:
            words = []
            for step in range(max_words):
                v_sem_h, v_sem_c = self.v_sem_layer.step(s_tags, v_sem_h, v_sem_c, decoder_input, var_drop_p=.1)
                v_syn_h, v_syn_c = self.v_syn_layer.step(pos_emb, v_syn_h, v_syn_c, decoder_input, var_drop_p=.1)
                se_sy_h, se_sy_c = self.se_sy_layer.step(pos_emb, se_sy_h, se_sy_c, decoder_input, var_drop_p=.1)

                if self.dataset_name == 'MSVD':
                    v_attn1 = self.v_sem_attn(v_feats, v_sem_h)
                    v_attn2 = self.v_syn_attn(v_feats, v_syn_h)
                    v_attn3 = self.se_sy_attn(v_feats, se_sy_h)
                    v_attn = (v_attn1 + v_attn2 + v_attn3) / 3
                elif self.dataset_name == 'MSR-VTT':
                    h = torch.cat((v_sem_h,v_syn_h,se_sy_h),dim=1)
                    v_attn = self.v_attn(v_feats, h)

                rnn_h = self.__adaptive_merge(rnn_h, v_attn, v_sem_h, v_syn_h, se_sy_h)

                # compute word_logits
                # (batch_size x output_size)
                word_logits = self.out(rnn_h)

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

            return (
                torch.cat([o.unsqueeze(1) for o in outputs], dim=1).contiguous(), 
                torch.cat([w.unsqueeze(1) for w in words], dim=1).contiguous(), 
                torch.cat([e.unsqueeze(1) for e in embedds], dim=1).contiguous()
            )
        else:
            words = []
            for seq_pos in range(gt_captions.size(1)):
                v_sem_h, v_sem_c = self.v_sem_layer.step(s_tags, v_sem_h, v_sem_c, decoder_input, var_drop_p=.1)
                v_syn_h, v_syn_c = self.v_syn_layer.step(pos_emb, v_syn_h, v_syn_c, decoder_input, var_drop_p=.1)
                se_sy_h, se_sy_c = self.se_sy_layer.step(pos_emb, se_sy_h, se_sy_c, decoder_input, var_drop_p=.1)

                if self.dataset_name == 'MSVD':
                    v_attn1 = self.v_sem_attn(v_feats, v_sem_h)
                    v_attn2 = self.v_syn_attn(v_feats, v_syn_h)
                    v_attn3 = self.se_sy_attn(v_feats, se_sy_h)
                    v_attn = (v_attn1 + v_attn2 + v_attn3) / 3
                elif self.dataset_name == 'MSR-VTT':
                    h = torch.cat((v_sem_h,v_syn_h,se_sy_h),dim=1)
                    v_attn = self.v_attn(v_feats, h)

                rnn_h = self.__adaptive_merge(rnn_h, v_attn, v_sem_h, v_syn_h, se_sy_h)

                # compute word_logits
                # (batch_size x output_size)
                word_logits = self.out(rnn_h)

                use_teacher_forcing = True if random.random() < teacher_forcing_p or seq_pos == 0 else False
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
                torch.cat([e.unsqueeze(1) for e in embedds], dim=1).contiguous()
            )

    def forward(self, encoding, teacher_forcing_p=.5, gt_captions=None, max_words=None):
        return self.forward_fn(v_feats=encoding[0],
                               v_pool=encoding[1],
                               s_tags=encoding[2],
                               pos_emb=encoding[3],
                               gt_captions=gt_captions,
                               teacher_forcing_p=teacher_forcing_p,
                               max_words=max_words)

    def sample(self, videos_encodes):
        return self.forward(videos_encodes, None, teacher_forcing_p=0.0)
