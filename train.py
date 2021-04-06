import os
import sys
import argparse
import pickle
import json
import datetime 
import logging
import time
from multiprocessing import Pool

import h5py
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from utils import decode_from_tokens, load_texts, evaluate_from_tokens, get_trainer_str, get_sem_tagger_str, get_syn_embedd_str, get_avscn_decoder_str, get_semsynan_decoder_str, get_mm_str, get_vncl_cell_str
from vocabulary import Vocabulary
from configuration_dict import ConfigDict
from loader import get_dense_loader
from model.dense_captioner import DenseCaptioner
from loss import DenseCaptioningLoss


class Trainer:
    def __init__(self, trainer_config, dense_captioner_config, modules_config, device):
        self.trainer_config = trainer_config
        self.modules_config = modules_config

        self.exp_name = f'({trainer_config.str})'
        for config in modules_config.values():
            self.exp_name += f' ({config.str})'

        # self.encoder_name = '{}-{}{}-drop{}'.format(self.config.encoder_num_layers, 
        #                                             'bi' if self.config.encoder_bidirectional else '', 
        #                                             self.config.encoder_rnn_cell, self.config.encoder_drop_p)
        # self.decoder_name = '{}-{}{}-drop{}-bs{}-{}-{}-{}'.format(self.config.decoder_num_layers,  
        #                                                        self.config.decoder_rnn_cell, 
        #                                                        '-attn' if self.config.decoder_attn else '', 
        #                                                        self.config.decoder_drop_p, self.config.decoder_beam_size,
        #                                                        self.config.decoder_beam_search_logic, 
        #                                                        'train-max' if self.config.decoder_train_sample_max else 'train-dist',
        #                                                        'test-max' if self.config.decoder_test_sample_max else 'test-dist' )
        # self.loss_name = '{}-{}'.format(self.config.criterion_name, self.config.criterion_reduction)
        
        # self.exp_name = '{} ({}) ({}) ({}) batch{} {}'.format(self.config.dataset_name, 
        #                                                       self.encoder_name, 
        #                                                       self.decoder_name,
        #                                                       self.loss_name,
        #                                                       self.config.batch_size,
        #                                                       self.config.learning_rate)
        
        self.datetime_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.writer = SummaryWriter(log_dir=os.path.join('./log/runs/', f'{self.datetime_str} {trainer_config.str}'))
        logging.basicConfig(filename='./log/output_{}'.format(self.datetime_str),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        self.logger = logging.getLogger('{}'.format(self.exp_name))

        print('Experiment: {}'.format(self.datetime_str), '\n')
        # print(self.exp_name, '\n')
        print('Process id {}'.format(os.getpid()), '\n')
        
        if device == 'gpu' and torch.cuda.is_available():
            freer_gpu_id = get_freer_gpu()
            self.device = torch.device('cuda:{}'.format(freer_gpu_id))
            torch.cuda.empty_cache()
            self.logger.info('Running on cuda:{} device'.format(freer_gpu_id))
            print('Running on cuda:{} device'.format(freer_gpu_id))
        else:
            self.device = torch.device('cpu')
            self.logger.info('Running on cpu device')
            print('Running on cpu device')


class DenseVideo2TextTrainer(Trainer):
    def __init__(self, trainer_config, dense_captioner_config, modules_config, dataset_folder, result_folder, device):
        super(DenseVideo2TextTrainer, self).__init__(trainer_config, dense_captioner_config, modules_config, device)

        self.dataset_folder = dataset_folder

        # max_frames = 20 #20 30
        # # self.max_words = 30
        # cnn_feature_size = 2048
        # c3d_feature_size = 4096
        # i3d_feature_size = 400
        # eco_feature_size = 1536
        # res_eco_features_size = 3584
        # cnn_global_size = 512
        # projected_size = 512
        # hidden_size = 1024  # Number of hidden layer units of the cyclic network
        # mid_size = 128  # The middle of the boundary detection layer represents the dimension
        
        # n_tags = 300
        # global_tagger_hidden_size = 1024
        # specific_tagger_hidden_size = 128
        # hidden_size = 1024
        # embedding_size = 300  #1024
        # rnn_in_size = 300  #1024
        # rnn_hidden_size = 1024
        
        # self.__load_ground_truth_captions()
        # self.__load_fusion_ground_truth_captions()

        # load vocabulary
        with open(os.path.join(dataset_folder, 'new_dense_corpus2.pkl'), "rb") as f:
            self.corpus = pickle.load(f)
            idx2op_dict = self.corpus[4] 
            idx2word_dict = self.corpus[6]
            idx2pos_dict = self.corpus[8]
            idx2upos_dict = self.corpus[9]

        self.programs_vocab = Vocabulary.from_idx2word_dict(idx2op_dict, False)
        print('Size of programs_vocab: {}'.format(len(self.programs_vocab)))
        
        self.caps_vocab = Vocabulary.from_idx2word_dict(idx2word_dict, False)
        print('Size of caps_vocab: {}'.format(len(self.caps_vocab)))
        
        self.pos_vocab = Vocabulary.from_idx2word_dict(idx2pos_dict, False)
        print('Size of pos_vocab: {}'.format(len(self.pos_vocab)))
        
        self.upos_vocab = Vocabulary.from_idx2word_dict(idx2upos_dict, False)
        print('Size of upos_vocab: {}'.format(len(self.upos_vocab)), '\n')
        
        # Pretrained Embedding
        pretrained_we = torch.Tensor(self.corpus[7])
        
        # Initialize data loaders
        self.__init_dense_loader()

        # Load ground-truth for computing evaluation metrics
        # self.__load_ground_truth_programs()
        # self.__load_ground_truth_captions()

        # Model
        print('\nInitializing the Model...')
        self.dense_captioner = DenseCaptioner(dense_captioner_config,
                                              self.modules_config['sem_tagger_config'],
                                              self.modules_config['syn_embedd_config'],
                                              self.modules_config['avscn_dec_config'],
                                              self.modules_config['semsynan_dec_config'],
                                              self.modules_config['mm_config'],
                                              self.modules_config['vncl_cell_config'],
                                              progs_vocab=self.programs_vocab,
                                              caps_vocab=self.caps_vocab,
                                              pretrained_we=pretrained_we,
                                              device=device)

        # Optimizer
        print('\nInitializing the Optimizer...')
        if self.trainer_config.optimizer_name == 'Adagrad':
            self.optimizer = optim.Adagrad([{'params': self.dense_captioner.mm_enc.parameters()}, 
                                            {'params': self.dense_captioner.rnn_cell.parameters()}, 
                                            {'params': self.dense_captioner.fc.parameters()}, 
                                            {'params': self.dense_captioner.clip_captioner.parameters()}], 
                                           lr=self.trainer_config.learning_rate)
        else:
            self.optimizer = optim.Adam([{'params': self.dense_captioner.mm_enc.parameters()}, 
                                         {'params': self.dense_captioner.rnn_cell.parameters()}, 
                                         {'params': self.dense_captioner.fc.parameters()}, 
                                         {'params': self.dense_captioner.clip_captioner.parameters()}], 
                                        lr=self.trainer_config.learning_rate) #, weight_decay=.0001)

        # learning-rate decay scheduler
        lambda1 = lambda epoch: self.trainer_config.lr_decay_factor ** (epoch // 40)
        lambda2 = lambda epoch: self.trainer_config.lr_decay_factor ** (epoch // 40)
        lambda3 = lambda epoch: self.trainer_config.lr_decay_factor ** (epoch // 40)
        lambda4 = lambda epoch: self.trainer_config.lr_decay_factor ** (epoch // 40)        
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, 
                                                        lr_lambda=[lambda1, lambda2, lambda3, lambda4])

        # Loss function
        self.criterion = DenseCaptioningLoss(config=trainer_config.criterion_config,
                                             c_max_len=self.max_words, 
                                             p_max_len=self.max_prog)

        print('\n****We are ready to start the training process****\n')

    def __init_vocab(self, corpus):
        self.vocab = Vocabulary.from_words(['<pad>', '<start>', '<end>', '<unk>'])
        self.vocab.add_sentences(corpus)
        print('Vocabulary has {} words.'.format(len(self.vocab)))

    def __load_ground_truth_programs(self):
        # this is the ground truth captions
        self.ref_programs = {'val_1': {}}
        
        ref_txt_path = {'val_1': os.path.join(self.dataset_folder,'val_1_ref_captions.txt')}
        
        for phase in ['val_1']:
            self.ref_programs[phase] = load_texts(ref_txt_path[phase])

    def __load_ground_truth_captions(self):
        # this is the ground truth captions
        self.ref_captions = {'val_1': {}}
        
        ref_txt_path = {'val_1': os.path.join(self.dataset_folder,'val_1_ref_programs.txt')}
        
        for phase in ['val_1']:
            self.ref_captions[phase] = load_texts(ref_txt_path[phase])

    def __load_fusion_ground_truth_captions(self):
        configs = [#ConfigurationFile('config.ini', 'MSVD-sem-syn-cn-max'), 
                   #ConfigurationFile('config.ini', 'MSR-VTT-sem-syn-cn-max'),
                   #ConfigurationFile('config.ini', 'VATEX-sem-syn-cn-max'), 
                   ConfigurationFile('config.ini', 'TRECVID-sem-syn-cn-max')]
        
        # this is the ground truth captions
        self.ground_truth = {'valid': {}}
        for config in configs:
            reference_txt_path = {'valid': 'results/{}_val_references.txt'.format(config.dataset_name)}
        
            for phase in ['valid']:
                current_len = len(self.ground_truth[phase])
                for line in list(open(reference_txt_path[phase], 'r')):
                    row = line.split('\t')
                    idx = current_len + int(row[0])
                    sentence = row[1].strip()
                    if idx in self.ground_truth[phase]:
                        self.ground_truth[phase][idx].append(sentence)
                    else:
                        self.ground_truth[phase][idx] = [sentence]

    def __extract_split_data_from_corpus(self, split=0):
        split_data = self.corpus[split]
        vidxs, cidxs, intervals, fps, progs, prog_lens, caps, pos, upos, cap_lens = split_data[0], split_data[1], split_data[2], split_data[3], split_data[4], [len(p) for p in split_data[4]], split_data[5], split_data[6], split_data[7], [[len(c) for c in caps] for caps in split_data[6]]
        max_prog = max(prog_lens)
        caps_count_t = torch.tensor([len(v_caps) for v_caps in caps], dtype=torch.int8)
        max_caps = torch.max(caps_count_t)
        max_words = max([l for v_lens in cap_lens for l in v_lens])
        
        caps_t = torch.LongTensor(len(caps), max_caps, max_words).fill_(0)
        cap_lens_t = torch.LongTensor(len(caps), max_caps).fill_(0)
        cidxs_t = torch.LongTensor(len(caps), max_caps).fill_(0)
        for i, (v_cidxs, v_caps) in enumerate(zip(cidxs, caps)):
            v_caps_t = torch.LongTensor(max_caps, max_words).fill_(0)
            cidxs_t[i, :len(v_cidxs)] = torch.tensor(v_cidxs)
            for j, (cidx, c) in enumerate(zip(v_cidxs, v_caps)):
                v_caps_t[j, :len(c)] = torch.tensor(c)
                cap_lens_t[i, j] = len(c)
            caps_t[i] = v_caps_t

        pos_t = torch.zeros((len(caps), max_caps, max_words), dtype=torch.int8)
        for i, v_pos in enumerate(pos):
            v_pos_t = torch.zeros((max_caps, max_words), dtype=torch.int8)
            for j, c in enumerate(v_pos):
                v_pos_t[j, :len(c)] = torch.tensor(c)
            pos_t[i] = v_pos_t

        upos_t = torch.zeros((len(caps), max_caps, max_words), dtype=torch.int8)
        for i, v_upos in enumerate(upos):
            v_upos_t = torch.zeros((max_caps, max_words), dtype=torch.int8)
            for j, c in enumerate(v_upos):
                v_upos_t[j, :len(c)] = torch.tensor(c)
            upos_t[i] = v_upos_t

        intervals_t = torch.zeros((len(caps), max_caps, 2))
        for i, v_intervals in enumerate(intervals):
            intervals_t[i, :len(v_intervals)] = torch.Tensor([[s,e] for s, e in v_intervals])

        progs_t = torch.zeros((len(caps), max_prog), dtype=torch.long)
        for i, v_prog in enumerate(progs):
            progs_t[i, :len(v_prog)] = torch.LongTensor(v_prog)
        
        return vidxs, cidxs_t, intervals_t, caps_count_t, fps, progs_t, prog_lens, caps_t, pos_t, upos_t, cap_lens_t

    def __init_dense_loader(self):
        print('Initializing data loaders...')

        # open the h5 file with visual features
        h5_path='data/test4_train.h5' # os.path.join(self.dataset_folder, self.trainer_config.features_filename)
        h5 = h5py.File(h5_path, 'r')
        dataset = h5[self.trainer_config.dataset_name]

        # get train split data
        print(' initializing train split data loader...')
        vidxs, cidxs, intervals_t, caps_count_t, fps, progs_t, prog_lens, caps_t, pos_t, upos_t, cap_lens_t = self.__extract_split_data_from_corpus(split=0)
        self.max_prog = progs_t.size(1)
        self.max_caps = caps_t.size(1)
        self.max_words = caps_t.size(2)
        self.max_interval = torch.max(intervals_t.view(-1, 2)[:,1]-intervals_t.view(-1, 2)[:,0])
        self.last_interval_end = torch.max(intervals_t.view(-1, 2)[:,1])

        # get train loader
        train_loader = get_dense_loader(h5_dataset=dataset, vidxs=vidxs, cidxs=cidxs, intervals=intervals_t, caps_count=caps_count_t, 
                                        captions=caps_t, pos=pos_t, upos=upos_t, cap_lens=cap_lens_t, progs=progs_t, 
                                        prog_lens=prog_lens, batch_size=self.trainer_config.batch_size, train=True)
        
        # get valid split data
        print(' initializing valid split data loader...')
        vidxs, cidxs, intervals_t, caps_count_t, fps, progs_t, prog_lens, caps_t, pos_t, upos_t, cap_lens_t = self.__extract_split_data_from_corpus(split=1)
        # self.max_prog = max(self.max_prog, progs_t.size(1))
        # self.max_caps = max(self.max_caps, caps_t.size(1))
        # self.max_words = max(self.max_words, caps_t.size(2))
        # self.max_interval = max(self.max_interval, torch.max(intervals_t.view(-1, 2)[:,1]-intervals_t.view(-1, 2)[:,0]))
        # self.last_interval_end = max(self.last_interval_end, torch.max(intervals_t.view(-1, 2)[:,1]))

        # get valid loader
        val_loader = get_dense_loader(h5_dataset=dataset, vidxs=vidxs, cidxs=cidxs, intervals=intervals_t, caps_count=caps_count_t, 
                                      captions=caps_t, pos=pos_t, upos=upos_t, cap_lens=cap_lens_t, progs=progs_t, 
                                      prog_lens=prog_lens, batch_size=self.trainer_config.batch_size*2, train=False)

        print(' Max program len:', self.max_prog)
        print(' Max caption len:', self.max_words)
        print(' Max intervals count:', self.max_caps)
        print(' Max interval len:', int(self.max_interval))
        print(' Last interval end:', int(self.last_interval_end))

        self.loaders = {'train': train_loader, 'val_1': val_loader}

    def __decode_from_tokens(self, tokens):
        words = []
        for token in tokens:
            if token.item() == self.vocab('<eos>'):
                break
            words.append(self.vocab.idx_to_word(token.item()))
        return ' '.join(words)

    def __get_sentences(self, all_outputs, all_video_ids):
        predicted_sentences = {}
        for outputs, video_ids in zip(all_outputs, all_video_ids):
            for predicted_tokens, vid in zip(outputs, video_ids):
                predicted_sentences[vid] = [self.__decode_from_tokens(predicted_tokens)]
        return predicted_sentences

    def __process_batch(self, video_feats, feats_count, gt_intervals, gt_caps_count, gt_captions, gt_pos, gt_upos, gt_cap_lens, gt_program, gt_prog_len, 
                        teacher_forcing_ratio=.5, phase='train', use_rl=False):
        bsz = video_feats[0].size(0)

        # Move all tensors to device
        for i,f in enumerate(video_feats):
            video_feats[i] = f.to(self.device)
        gt_intervals = gt_intervals.to(self.device)
        gt_captions = gt_captions.to(self.device)
        gt_pos = gt_pos.to(self.device)
        gt_upos = gt_upos.to(self.device)
        gt_cap_lens = gt_cap_lens.to(self.device)
        gt_program = gt_program.to(self.device)
        gt_prog_len = gt_prog_len.to(self.device)

        # Constructing the mini batch's Variables
        # cnn_feats = Variable(cnn_feats).to(self.device)
        # c3d_feats = Variable(c3d_feats).to(self.device)
        # i3d_feats = Variable(i3d_feats).to(self.device)
        # eco_feats = Variable(eco_feats).to(self.device)
        # eco_sem_feats = Variable(eco_sem_feats).to(self.device)
        # tsm_sem_feats = Variable(tsm_sem_feats).to(self.device)
        # cnn_globals = Variable(cnn_globals).to(self.device)
        # cnn_sem_globals = Variable(cnn_sem_globals).to(self.device)
        # tags_globals = Variable(tags_globals).to(self.device)
        # res_eco_globals = Variable(res_eco_globals).to(self.device)
        # targets = Variable(targets).to(self.device)
        # target_lens = Variable(target_lens).to(self.device)

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            program, captions, intervals, caps_count = self.dense_captioner(video_feats, feats_count, teacher_forcing_ratio, gt_program, gt_captions, gt_intervals)
            # video_encoded = self.encoder(cnn_feats, c3d_feats, i3d_feats, eco_feats, eco_sem_feats, tsm_sem_feats, cnn_globals, cnn_sem_globals, tags_globals, res_eco_globals)
            
            # outputs, tokens = self.decoder(video_encoded, targets if phase == 'train' else None, teacher_forcing_ratio)
            
            # Straighten the output (removing the part of the pad) and then straighten it
            # outputs = torch.cat([decode[j][:target_lens[j]] for j in range(bsz)], dim=0)

            # Straighten the target (remove the part of the pad) and straighten it
            # targets = torch.cat([targets[j][:target_lens[j]] for j in range(bsz)], dim=0)

            # Evaluate the loss function
            loss = self.criterion(gt_captions, gt_cap_lens, captions, gt_program, gt_prog_len, program, gt_intervals, intervals, gt_caps_count, caps_count)

            # if not use_rl:
            #     if type(self.criterion) is SentenceLengthLoss:
            #         loss = self.criterion(outputs.view(-1, len(self.vocab)), targets.view(-1), target_lens)
            #     else:
            #         loss = self.criterion(outputs.view(-1, len(self.vocab)), targets.view(-1))
            # elif self.reinforce_type == 'syntax':
            #     # compute cap_wids, lengths and cap_bows from tokens
            #     corpus, taggeds, lengths = [], [], []
            #     for gen in tokens:
            #         s = self.__decode_from_tokens(gen)
            #         tokenized = nltk.word_tokenize(s)
            #         tagged = nltk.pos_tag(tokenized)
            #         if not len(tagged):
            #             tagged = ['UNK']
            #         taggeds.append(tagged)
            #         lengths.append(len(tagged))
            #         template = ' '.join([t[1] for t in tagged])
            #         corpus.append(template)            
                
            #     with open("./video_tagging/pos_vectorizer.pk", "rb") as f:
            #         pos_vectorizer = pickle.load(f)
            #         pos_vocab = ['EOS', 'UNK'] + pos_vectorizer.get_feature_names()
            #         eos_idx = 0
            #         unk_idx = 1
            #     x = pos_vectorizer.transform(corpus)
            #     max_length = max(lengths)
                
            #     s_feats = torch.from_numpy(x.toarray() / np.c_[lengths]).float()
            #     s_wids = torch.LongTensor([[pos_vocab.index(w[1]) if w[1] in pos_vocab else unk_idx for w in s] + [eos_idx for _ in range(max_length - len(s))] for s in taggeds])
            #     s_lens = torch.LongTensor(lengths)
                
            #     data = list(zip(s_wids, s_lens, s_feats))
            #     data.sort(key=lambda x: x[1], reverse=True)
            #     s_wids, s_lens, s_feats = zip(*data)
                
            #     s_wids = torch.stack(s_wids, 0).to(self.device)
            #     s_lens = torch.stack(s_lens, 0).to(self.device)
            #     s_feats = torch.stack(s_feats, 0).to(self.device)
                
            #     with torch.no_grad():
            #         # compute visual encoding
            #         #v_logits = self.visual_model(res_eco_feats)
            #         v_logits = self.visual_model(cnn_feats, c3d_feats, cnn_globals, None)

            #         # compute syntax encoding
            #         s_logits = self.syntax_model(s_wids, s_lens, s_feats)

            #         # compute reward
            #         rewards = self.reward(s_logits, v_logits)
                
            #     if type(self.criterion) is SentenceLengthLoss:
            #         loss = self.criterion(outputs.view(-1, len(self.vocab)), targets.view(-1), target_lens, rewards=rewards)
            #     else:
            #         loss = self.criterion(outputs.view(-1, len(self.vocab)), targets.view(-1))
            # else:
            #     loss = RewardCriterion(outputs.log(), targets, reward)

        if phase == 'train':
            loss.backward()
            self.optimizer.step()
        
        return loss, program, captions
        
    def __evaluate(self, predicted_sentences, phase):
        scores = score(self.ground_truth[phase], predicted_sentences)
        weights = {'Bleu_1':0., 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 1.4, 'CIDEr': 1.17, 'METEOR': 2., 'ROUGE_L':1.}
        scores['All_Metrics'] = sum([scores[k] * weights[k] for k in scores.keys()])
        return scores

    def __report_results(self, metrics_results, predicted_sentences, phase, epoch, save_checkpoints_dir):
        self.logger.info('{} set metrics: {}'.format(phase, metrics_results))
        for name, result in metrics_results.items():
            self.writer.add_scalar('data/end2end/{}-{}'.format(phase, name), result, epoch)
            if self.best_metrics[phase][name][1] < result:
                self.best_metrics[phase][name] = (epoch, result)
                if name in ['Bleu_4','METEOR', 'ROUGE_L', 'CIDEr', 'All_Metrics']:
                    self.early_stop = 0
                if name == 'METEOR' and phase == 'valid':
                    torch.save(obj={'epoch': epoch,
                                    'dense_captioner': self.dense_captioner.state_dict(),
                                    'optimizer': self.optimizer.state_dict(),
                                    'prog_best_metrics': self.prog_best_metrics,
                                    'cap_best_metrics': self.cap_best_metrics},
                               f=os.path.join(save_checkpoints_dir, f'captioning_chkpt_{epoch}.pkl'.format))

                    with open(os.path.join(save_checkpoints_dir, f'captioning_chkpt_{epoch}_output.json'), 'w') as f:
                        json.dump(predicted_sentences, f)

                    # remove previously saved
                    if self.last_saved_epoch != -1:
                        os.remove(os.path.join(save_checkpoints_dir, f'captioning_chkpt_{self.last_saved_epoch}.pkl'))
                    self.last_saved_epoch = epoch

    def train_model(self, resume=False, checkpoint_path=None, min_num_epochs=50, early_stop_limit=10):
        parallel_pool = Pool()
        # self.logger.info('Training captioning model on [{}] dataset with [{}] encoder and [{}] decoder'
        #                  .format(self.config.dataset_name, self.encoder_name, self.decoder_name))

        save_checkpoints_dir = os.path.join('./models', self.trainer_config.str, self.datetime_str)
        if not os.path.exists(save_checkpoints_dir):
            os.makedirs(save_checkpoints_dir)

        if resume and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            begin_epoch = checkpoint['epoch']
            self.prog_best_metrics = checkpoint['prog_best_metrics']
            self.cap_best_metrics = checkpoint['cap_best_metrics']
            self.dense_captioner.load_state_dict(checkpoint['dense_captioner'])
        else:
            begin_epoch = 0
            self.prog_best_metrics, self.cap_best_metrics = {}, {}
            for p in ['val_1']:
                self.prog_best_metrics[p] = {'Bleu_1': (0, 0), 'Bleu_2': (0, 0), 'Bleu_3': (0, 0), 'Bleu_4': (0, 0),
                                             'METEOR': (0, 0), 'ROUGE_L': (0, 0), 'CIDEr': (0, 0), 'SPICE': (0, 0), 'All_Metrics': (0, 0)}
            for p in ['val_1']:
                self.cap_best_metrics[p] = {'Bleu_1': (0, 0), 'Bleu_2': (0, 0), 'Bleu_3': (0, 0), 'Bleu_4': (0, 0),
                                            'METEOR': (0, 0), 'ROUGE_L': (0, 0), 'CIDEr': (0, 0), 'SPICE': (0, 0), 'All_Metrics': (0, 0)}

        self.dense_captioner.to(self.device)
        print('\nParameters of Dense Captioner model:\n')
        total_size = 0
        for n, p in self.dense_captioner.named_parameters():
            print(n, p.size(), p.device)
            total_size += torch.numel(p)
        print(' total size: ', (total_size*8)/(1024**3), '\n')        

        # Start training process
        self.early_stop, self.last_saved_epoch = 0, -1
        time_phases = {'train': 0, 'val_1': 0}
        cap_metrics_results, prog_metrics_results = None, None
        for epoch in range(begin_epoch, 1000):
            time_start_epoch = time.perf_counter()
                        
            k = self.trainer_config.convergence_speed_factor
            teacher_forcing_ratio = max(.6, k / (k + np.exp(epoch / k)))  # inverse sigmoid decay
            self.writer.add_scalar('data/end2end/teacher_forcing_ratio', teacher_forcing_ratio, epoch)

            loss_phases = {'train': 0, 'val_1': 0}
            for phase in ['train', 'val_1']:
                if phase == 'train':
                    self.dense_captioner.train()
                else:
                    self.dense_captioner.eval()

                # predicted_sentences = {}
                loss_count = 0
                all_programs, all_captions, all_video_ids = [], [], []
                for i, (vidx, cidxs, cnn, c3d, feats_count, intervals, caps_count, captions, pos, upos, cap_lens, progs, prog_lens) in enumerate(self.loaders[phase], start=1):
                    video_feats = [cnn, c3d]
                    use_rl = False
                    loss, program, captions = self.__process_batch(video_feats, feats_count, intervals, caps_count, captions, pos, upos, cap_lens, progs, prog_lens, 
                                                                        teacher_forcing_ratio, phase, use_rl=use_rl)
                    loss_count += loss.item()

                    self.writer.add_scalar('data/end2end/{}-iters-loss'.format(phase), loss, epoch * len(self.loaders[phase]) + i)                

                    lrs = self.lr_scheduler.get_last_lr()
                    sys.stdout.write('\rEpoch:{0:03d} Phase:{1:6s} Iter:{2:04d}/{3:04d} Loss:{4:10.4f} lr:{5:.6f}'.format(epoch, phase, i, len(self.loaders[phase]), loss.item(), lrs[0]))
                    
                    if phase != 'train':
                        all_programs.append(program.to('cpu'))
                        all_prog_ids.append(vidx)

                        all_captions.append(captions.to('cpu'))
                        all_caps_ids.append(cidxs)
                        
                        # for predicted_tokens, vid in zip(outputs, video_ids):
                        #     predicted_sentences[vid] = [self.__decode_from_tokens(predicted_tokens)]
                        
                        # logging sample sentences of prediction and target
                        # self.logger.info('[vid:{}]'.format(video_ids[0]))
                        # self.logger.info('\nWE: {}\nGT: {}'.format(predicted_sentences[video_ids[0]],
                        #                                            self.__decode_from_tokens(captions[0].squeeze())))

                avg_loss = loss_count/len(self.loaders[phase])
                loss_phases[phase] = avg_loss
                self.writer.add_scalar('data/end2end/{}-epochs-avg-loss'.format(phase), avg_loss, epoch)

                if phase != 'train':
                    self.early_stop += 1
                    # predicted_sentences = pool.apply_async(self.__get_sentences, [all_outputs, all_video_ids])

                    if cap_metrics_results is not None:
                        cap_metrics_results, pred_caps = cap_metrics_results.get()
                        prog_metrics_results, pred_progs = prog_metrics_results.get()

                        self.__report_results(cap_metrics_results, pred_caps, prog_metrics_results, pred_progs, phase, epoch-1, save_checkpoints_dir)

                        if self.early_stop == 0:
                            log_msg = f'\n IMPROVEMENT ON {phase} e{epoch} !\n'
                            log_msg += '\t captioning metrics:'
                            for k, (e, v) in self.cap_best_metrics[phase].items(): 
                                log_msg += '\t\t{0}: ({1:03d}, {2:.3f}) \n'.format(k, e, v)
                            for k, (e, v) in self.cap_best_metrics[phase].items(): 
                                log_msg += '\t\t{0}: ({1:03d}, {2:.3f}) \n'.format(k, e, v)
                            print(log_msg, '\n')
                            self.logger.info(log_msg)
                    
                    prog_metrics_results = parallel_pool.apply_async(evaluate_from_tokens, [self.programs_vocab, all_programs, all_prog_ids, self.ref_programs[phase]])
                    cap_metrics_results = parallel_pool.apply_async(evaluate_from_tokens, [self.caps_vocab, all_captions, all_caps_ids, self.ref_captions[phase]])
                    # metrics_results, predicted_sentences = evaluate(all_outputs, all_video_ids)
                                                     
                time_phases[phase] += time.clock() - time_start_epoch
                
            log_msg = '\n'
            for k, v in loss_phases.items(): 
                log_msg += ' {0} Avg-Loss:{1:10.4f}'.format(k, v)
            for k, v in time_phases.items(): 
                log_msg += ' {0} Time:{1:10.3f}s'.format(k, v/(epoch+1))
            log_msg += ' teacher_forcing_ratio:{0:.3f} enc-lr:{1:.6f} dec-lr:{1:.6f}'.format(teacher_forcing_ratio, lrs[0], lrs[1])
            # vid = video_ids[0]
            # log_msg += '\n[vid {}]:\nWE: {}\nGT: {}'.format(vid, predicted_sentences[vid], self.ground_truth['valid'][vid])
            sys.stdout.write(log_msg+'\n')
                
            if epoch >= min_num_epochs and self.early_stop >= early_stop_limit * 2:
                metrics_results, predicted_sentences = metrics_results.get()
                self.__report_results(metrics_results, predicted_sentences, phase, epoch-1, save_checkpoints_dir)
                msg = '----early stopped at epoch {} after {} without any improvement-----'.format(epoch, early_stop_limit)
                self.logger.debug(msg)
                print(msg)
                break
                
            self.writer.add_scalar('data/end2end/learning-rate', self.optimizer.param_groups[0]['lr'], epoch)                
            self.lr_scheduler.step()
            
        self.logger.info('Best results: {}'.format(str(self.best_metrics)))
        return self.best_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate captions por test samples')
    parser.add_argument('-chckpt', '--checkpoint_path', type=str, default='pretrain/chckpt.pt',
                    help='Set the path to pre-trained model (default is pretrain/chckpt.pt).')
    parser.add_argument('-data', '--dataset_folder', type=str, default='data/MSVD',
                    help='Set the path to dataset folder (default is data/MSVD).')
    parser.add_argument('-out', '--output_folder', type=str, default='results/MSVD',
                    help='Set the path to output folder (default is results/MSVD).')

    args = parser.parse_args()

    # load hiper-parameters
    config_path = os.path.join(args.dataset_folder, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    trainer_config = ConfigDict(config['trainer_config'])
    trainer_config.str = get_trainer_str(trainer_config)
    print(trainer_config.str)

    dense_captioner_config = ConfigDict(config['dense_captioner_config'])
    # trainer_config.str = get_trainer_str(trainer_config)
    # print(dense_captioner_config.str)

    sem_tagger_config = ConfigDict(config['sem_tagger_config'])
    sem_tagger_config.str = get_sem_tagger_str(sem_tagger_config)
    print(sem_tagger_config.str)

    syn_embedd_config = ConfigDict(config['syn_embedd_config'])
    syn_embedd_config.str = get_syn_embedd_str(syn_embedd_config)
    print(syn_embedd_config.str)
    
    avscn_dec_config = ConfigDict(config['avscn_decoder_config'])
    avscn_dec_config.str = get_avscn_decoder_str(avscn_dec_config)
    print(avscn_dec_config.str)
    
    semsynan_dec_config = ConfigDict(config['semsynan_decoder_config'])
    semsynan_dec_config.str = get_semsynan_decoder_str(semsynan_dec_config)
    print(semsynan_dec_config.str)

    mm_config = ConfigDict(config['multimodal_config'])
    mm_config.str = get_mm_str(mm_config)
    print(mm_config.str)
    
    vncl_cell_config = ConfigDict(config['vncl_cell_config'])
    vncl_cell_config.str = get_vncl_cell_str(vncl_cell_config)
    print(vncl_cell_config.str, '\n')
    
    print('Initializing the experiment.........')
    modules_config = {'sem_tagger_config': sem_tagger_config, 
                      'syn_embedd_config': syn_embedd_config, 
                      'avscn_dec_config': avscn_dec_config, 
                      'semsynan_dec_config': semsynan_dec_config, 
                      'mm_config': mm_config,
                      'vncl_cell_config': vncl_cell_config}
    # modules_config = [sem_tagger_config, syn_embedd_config, avscn_dec_config, semsynan_dec_config, vncl_cell_config]
    trainer = DenseVideo2TextTrainer(trainer_config, dense_captioner_config, modules_config, args.dataset_folder, args.output_folder, 'cpu')    

    print('Training.........')
    best_results = trainer.train_model(resume=False, 
                                       checkpoint_path='',
                                       early_stop_limit=10)

    print('Best results in the test set: {}'.format(str(best_results)))