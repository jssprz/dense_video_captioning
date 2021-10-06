import random

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import get_init_weights
from model.captioning.attention import Attention


class SCNAttnDecoder(nn.Module):
    def __init__(
        self,
        in_seq_length,
        n_feats,
        n_tags,
        embedding_size,
        h_size,
        rnn_in_size,
        rnn_h_size,
        vocab,
        device,
        encoder_num_layers,
        encoder_bidirectional,
        pretrained_embedding=None,
        rnn_cell="gru",
        num_layers=1,
        drop_p=0.5,
        beam_size=10,
        temperature=1.0,
        train_sample_max=False,
        test_sample_max=True,
        beam_search_logic="bfs",
    ):
        super(SCNAttnDecoder, self).__init__()
        self.h_size = h_size
        self.embedding_size = embedding_size
        self.output_size = len(vocab)
        self.in_seq_length = in_seq_length
        self.vocab = vocab
        self.device = device
        self.beam_size = beam_size
        self.temperature = temperature
        self.train_sample_max = train_sample_max
        self.test_sample_max = test_sample_max
        self.beam_search_logic = beam_search_logic

        self.num_layers = num_layers
        self.encoder_num_layers = encoder_num_layers
        self.encoder_num_directions = 2 if encoder_bidirectional else 1
        self.num_directions = 1

        # Components
        self.in_v_drop = nn.Dropout(drop_p)
        self.in_s_drop = nn.Dropout(drop_p)

        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding)
        else:
            self.embedding = nn.Embedding(self.output_size, embedding_size)
        self.embedd_drop = nn.Dropout(drop_p)

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

        self.b_i = nn.Parameter(torch.zeros(h_size))
        self.b_f = nn.Parameter(torch.zeros(h_size))
        self.b_o = nn.Parameter(torch.zeros(h_size))
        self.b_c = nn.Parameter(torch.zeros(h_size))

        self.out = nn.Linear(self.h_size * self.num_directions, self.output_size)

        self.__init_layers()

    def __init_layers(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

    def precompute_mats(self, v, s):
        # (batch_size x rnn_h_size)
        self.temp2_i = s @ self.Wb_i
        self.temp2_f = s @ self.Wb_f
        self.temp2_o = s @ self.Wb_o
        self.temp2_c = s @ self.Wb_c

        # (batch_size x rnn_h_size)
        self.temp5_i = s @ self.Ub_i
        self.temp5_f = s @ self.Ub_f
        self.temp5_o = s @ self.Ub_o
        self.temp5_c = s @ self.Ub_c

    def __compute_gate(self, activation, temp1, temp2, temp5, temp6, Wc, Uc, b):
        x = (temp1 * temp2) @ Wc
        h = (temp5 * temp6) @ Uc
        return activation(x + h + b)

    def step(self, rnn_h, rnn_c, decoder_input):
        # (batch_size x rnn_h_size)
        temp1_i = decoder_input @ self.Wa_i
        temp1_f = decoder_input @ self.Wa_f
        temp1_o = decoder_input @ self.Wa_o
        temp1_c = decoder_input @ self.Wa_c

        # (batch_size x rnn_h_size)
        temp6_i = rnn_h @ self.Ua_i
        temp6_f = rnn_h @ self.Ua_f
        temp6_o = rnn_h @ self.Ua_o
        temp6_c = rnn_h @ self.Ua_c

        # (batch_size x h_size)
        i = self.__compute_gate(
            torch.sigmoid, temp1_i, self.temp2_i, self.temp5_i, temp6_i, self.Wc_i, self.Uc_i, self.b_i,
        )
        f = self.__compute_gate(
            torch.sigmoid, temp1_f, self.temp2_f, self.temp5_f, temp6_f, self.Wc_f, self.Uc_f, self.b_f,
        )
        o = self.__compute_gate(
            torch.sigmoid, temp1_o, self.temp2_o, self.temp5_o, temp6_o, self.Wc_o, self.Uc_o, self.b_o,
        )
        c = self.__compute_gate(
            torch.tanh, temp1_c, self.temp2_c, self.temp5_c, temp6_c, self.Wc_c, self.Uc_c, self.b_c,
        )

        # (batch_size x h_size)
        rnn_c = f * rnn_c + i * c
        rnn_h = o * torch.tanh(rnn_c)

        return rnn_h, rnn_c

    def forward_fn(
        self, v_pool, s_tags, encoder_h, encoder_outputs, captions, teacher_forcing_p=0.5,
    ):
        batch_size = encoder_outputs.size(0)

        # (batch_size x embedding_size)
        decoder_input = Variable(torch.Tensor(batch_size, self.embedding_size).fill_(0)).to(self.device)

        if type(encoder_h) is tuple:
            # (encoder_n_layers * encoder_num_directions x batch_size x h_size) -> (encoder_n_layers x encoder_num_directions x batch_size x h_size)
            rnn_h = encoder_h[0].view(self.encoder_num_layers, self.encoder_num_directions, batch_size, self.h_size,)
            rnn_c = encoder_h[1].view(self.encoder_num_layers, self.encoder_num_directions, batch_size, self.h_size,)

        rnn_h = Variable(torch.cat([rnn_h[-i, 0, :, :] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)
        rnn_c = Variable(torch.cat([rnn_c[-i, 0, :, :] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)

        outputs = []

        s = self.in_s_drop(s_tags)
        v = self.in_v_drop(v_pool)

        self.precompute_mats(v, s)

        if not self.training:
            next_nodes = [([decoder_input], [], 0.0, rnn_h)]
            for step in range(self.out_seq_length):
                temp = []
                for tokens_seqs, word_logits_seqs, log_probs, rnn_h in next_nodes:
                    decoder_input = tokens_seqs[-1]

                    if step:
                        # (batch_size x 1) -> (batch_size x embedding_size)
                        decoder_input = self.embedding(decoder_input).squeeze(1)
                        decoder_input = self.embedd_drop(decoder_input)

                    rnn_h, rnn_c = self.step(rnn_h, rnn_c, decoder_input, encoder_h, encoder_outputs)

                    # compute word_logits
                    # (batch_size x output_size)
                    word_logits = self.out(rnn_h)

                    # compute word probs
                    word_log_probs = torch.log_softmax(word_logits, dim=1)

                    if self.test_sample_max:
                        # sample max probailities
                        # (batch_size x beam_size), (batch_size x beam_size)
                        sample_log_probs, sample_ids = word_log_probs.topk(k=self.beam_size, dim=1)
                    else:
                        # sample from distribution
                        word_probs = torch.exp(torch.div(word_log_probs, self.temperature))
                        # (batch_size x beam_size)
                        sample_ids = torch.multinomial(word_probs, self.beam_size).to(self.device)
                        sample_log_probs = word_log_probs.gather(dim=1, index=sample_ids)

                    for j in range(self.beam_size):
                        temp.append(
                            (
                                tokens_seqs + [sample_ids[:, j].unsqueeze(1)],
                                word_logits_seqs + [word_logits],
                                log_probs + torch.mean(sample_log_probs[:, j]).item() / self.out_seq_length,
                                rnn_h.clone(),
                            )
                        )

                next_nodes = sorted(temp, reverse=True, key=lambda x: x[2])[: self.beam_size]

            best_seqs, best_word_logits_seq, max_avg_prob, _ = next_nodes[0]
            return (
                torch.cat([o.unsqueeze(1) for o in best_word_logits_seq], dim=1).contiguous(),
                torch.cat([t for t in best_seqs[1:]], dim=1).contiguous(),
            )
        else:
            for seq_pos in range(self.out_seq_length):
                rnn_h, rnn_c = self.step(rnn_h, rnn_c, decoder_input, encoder_h, encoder_outputs)

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
                decoder_input = self.embedd_drop(decoder_input)

                outputs.append(word_logits)

            # (batch_size x out_seq_length x output_size), none
            return (
                torch.cat([o.unsqueeze(1) for o in outputs], dim=1).contiguous(),
                None,
            )

    def forward(self, encoding, teacher_forcing_p=0.5, gt_captions=None):
        return self.forward_fn(
            v_pool=videos_encodes[3],
            s_tags=videos_encodes[2],
            encoder_h=videos_encodes[1],
            encoder_outputs=videos_encodes[0],
            captions=captions,
            teacher_forcing_p=teacher_forcing_p,
        )

    def sample(self, videos_encodes):
        return self.forward(videos_encodes, None, teacher_forcing_p=0.0)


class AVSCNDecoder(nn.Module):
    def __init__(self, config, vocab, with_embedding_layer=True, pretrained_we=None, device=None):
        super(AVSCNDecoder, self).__init__()
        self.vocab = vocab
        self.device = device

        self.h_size = config.h_size
        self.embedding_size = config.embedding_size
        self.output_size = len(vocab)
        self.in_seq_length = config.in_seq_length
        self.beam_size = config.beam_size
        self.temperature = config.temperature
        self.train_sample_max = config.train_sample_max
        self.test_sample_max = config.test_sample_max
        self.beam_search_logic = config.beam_search_logic

        self.num_layers = config.num_layers
        self.encoder_num_layers = config.encoder_num_layers
        self.encoder_num_directions = 2 if config.encoder_bidirectional else 1
        self.num_directions = 1  # beause this decoder is not bidirectional

        # Components
        self.in_v_drop = nn.Dropout(config.drop_p)
        self.in_s_drop = nn.Dropout(config.drop_p)

        if with_embedding_layer:
            if pretrained_we is not None:
                self.embedding = nn.Embedding.from_pretrained(pretrained_we)
            else:
                self.embedding = nn.Embedding(self.output_size, self.embedding_size)
            self.embedd_drop = nn.Dropout(config.drop_p)

        self.semantic_layer = SCNAttnDecoder(
            config.in_seq_length,
            config.v_enc_size,
            config.sem_enc_size,
            config.embedding_size,
            config.h_size,
            config.rnn_in_size,
            config.rnn_h_size,
            vocab,
            device,
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
        )

        # change n_feats and n_tags only
        self.visual_layer = SCNAttnDecoder(
            config.in_seq_length,
            config.sem_enc_size,
            config.v_enc_size,
            config.embedding_size,
            config.h_size,
            config.rnn_in_size,
            config.rnn_h_size,
            vocab,
            device,
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
        )

        self.attn1 = Attention(
            self.in_seq_length, self.embedding_size, self.h_size, self.num_layers, self.num_directions, mode="soft",
        )
        self.attn2 = Attention(
            self.in_seq_length, self.embedding_size, self.h_size, self.num_layers, self.num_directions, mode="soft",
        )

        self.v_fc = nn.Linear(self.h_size, self.h_size)
        self.s_fc = nn.Linear(self.h_size, self.h_size)

        self.merge1 = nn.Linear(self.h_size + config.n_feats, self.h_size)
        self.merge2 = nn.Linear(self.h_size * 2, self.h_size)

        self.out = nn.Linear(self.h_size, self.output_size)

        self.__init_layers()

    def __init_layers(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

    def __dropout(self, x, keep_prob, mask_for):
        if not self.training or keep_prob >= 1.0:
            return x

        if mask_for in self.dropM:
            mask = self.dropM[mask_for]
        else:
            mask = x.new_empty(x.size(), requires_grad=False).bernoulli_(keep_prob)
            self.dropM[mask_for] = mask

        return x.masked_fill(mask == 0, 0) * (1.0 / keep_prob)

    def __adaptive_attn(self, v_attn, rnn_h, semantic_h, visual_h):
        rnn_h = self.__dropout(rnn_h, 0.8, "rnn_h")
        v_attn = self.__dropout(v_attn, 0.5, "v_attn")
        beta = torch.sigmoid(self.merge1(torch.cat((rnn_h, v_attn), dim=1)))

        visual_h = self.__dropout(visual_h, 0.8, "visual_h")
        semantic_h = self.__dropout(semantic_h, 0.8, "semantic_h")

        v_h = (torch.relu(self.v_fc(visual_h)) * semantic_h) + (beta * visual_h)
        s_h = (torch.relu(self.s_fc(semantic_h)) * visual_h) + ((1 - beta) * semantic_h)

        h = torch.cat((v_h, s_h), dim=1)
        return torch.relu(self.merge2(h))

    def reset_internals(self, batch_size):
        self.dropM = {}

        self.semantic_h = torch.zeros(batch_size, self.h_size).to(self.device)
        self.semantic_c = torch.zeros(batch_size, self.h_size).to(self.device)

        self.visual_h = torch.zeros(batch_size, self.h_size).to(self.device)
        self.visual_c = torch.zeros(batch_size, self.h_size).to(self.device)

        self.rnn_h = torch.zeros(batch_size, self.h_size).to(self.device)
        # rnn_c = torch.zeros(batch_size, self.h_size)

    def precompute_mats(self, v_pool, s_tags):
        s = self.in_s_drop(s_tags)
        v = self.in_v_drop(v_pool)

        self.semantic_layer.precompute_mats(v, s)
        self.visual_layer.precompute_mats(s, v)

    def step(self, v_feats, decoder_input):
        self.visual_h, self.visual_c = self.visual_layer.step(self.visual_h, self.visual_c, decoder_input)
        self.semantic_h, self.semantic_c = self.semantic_layer.step(
            self.semantic_h, self.semantic_c, decoder_input
        )
        visual_attn1 = self.attn1(v_feats, self.visual_h)
        visual_attn2 = self.attn2(v_feats, self.semantic_h)
        self.rnn_h = self.__adaptive_attn(
            (visual_attn1 + visual_attn2) / 2, self.rnn_h, self.semantic_h, self.visual_h
        )

        # compute word_logits
        # (batch_size x output_size)
        word_logits = self.out(self.rnn_h)

        return word_logits

    def forward_fn(
        self, v_feats, v_pool, s_tags, teacher_forcing_p=0.5, gt_captions=None, max_words=None,
    ):
        bs = v_pool.size(0)
        self.reset_internals(bs)
        self.precompute_mats(v_pool, s_tags)

        outputs, embedds = [], []

        # (batch_size x embedding_size)
        decoder_input = torch.zeros(bs, self.embedding_size).to(self.device)

        if not self.training:
            words = []
            for _ in range(max_words):
                word_logits = self.step(v_feats, decoder_input)

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
                embedds.append(decoder_input)

                # decoder_input = self.embedd_drop(decoder_input)

                outputs.append(word_logits)
                words.append(word_id)

            return (
                torch.cat([o.unsqueeze(1) for o in outputs], dim=1).contiguous(),
                torch.cat([w.unsqueeze(1) for w in words], dim=1).contiguous(),
                torch.cat([e.unsqueeze(1) for e in embedds], dim=1).contiguous(),
            )
        else:
            for seq_pos in range(gt_captions.size(1)):
                word_logits = self.step(v_feats, decoder_input)

                use_teacher_forcing = random.random() < teacher_forcing_p or seq_pos == 0
                if use_teacher_forcing:
                    # use the correct words,
                    # (batch_size)
                    decoder_input = gt_captions[:, seq_pos]
                elif self.train_sample_max:
                    # select the words ids with the max probability,
                    # (batch_size)
                    decoder_input = word_logits.max(1)[1]
                else:
                    # sample words from probability distribution
                    # (batch_size)
                    decoder_input = torch.multinomial(torch.softmax(word_logits, dim=1), 1).squeeze(1)

                # (batch_size) -> (batch_size x embedding_size)
                decoder_input = self.embedding(decoder_input).squeeze(1)
                embedds.append(decoder_input)

                decoder_input = self.embedd_drop(decoder_input)

                outputs.append(word_logits)

            # (batch_size x out_seq_length x output_size), none
            return (
                torch.cat([o.unsqueeze(1) for o in outputs], dim=1).contiguous(),
                None,
                torch.cat([e.unsqueeze(1) for e in embedds], dim=1).contiguous(),
            )

    def forward(self, encoding, teacher_forcing_p=0.5, gt_captions=None, max_words=None):
        return self.forward_fn(
            v_feats=encoding[0],
            v_pool=encoding[1],
            s_tags=encoding[2],
            teacher_forcing_p=teacher_forcing_p,
            gt_captions=gt_captions,
            max_words=max_words,
        )

    def sample(self, videos_encodes):
        return self.forward(videos_encodes, None, teacher_forcing_p=0.0)
