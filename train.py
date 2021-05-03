import os
import sys
import argparse
import pickle
import json
import datetime
import logging
import time
from multiprocessing import Pool
import heapq
from shutil import copyfile

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from utils import get_freer_gpu, decode_from_tokens, load_texts, evaluate_from_tokens, densecap_evaluate_from_tokens, get_trainer_str, get_dense_captioner_str, get_sem_tagger_str, get_syn_embedd_str, get_syn_tagger_str, get_avscn_decoder_str, get_semsynan_decoder_str, get_mm_str, get_vncl_cell_str
from vocabulary import Vocabulary
from configuration_dict import ConfigDict
from loader import extract_split_data_from_corpus, data2tensors, get_dense_loader
from model.dense_captioner import DenseCaptioner
from loss import DenseCaptioningLoss


class Trainer:
    def __init__(self, trainer_config, modules_config, out_folder):
        self.trainer_config = trainer_config
        self.modules_config = modules_config

        self.exp_name = f'({trainer_config.str})'
        for config in modules_config.values():
            self.exp_name += f' ({config.str})'

        self.out_folder = out_folder

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
        self.writer = SummaryWriter(log_dir=os.path.join(self.out_folder, 'log/runs/', f'{self.datetime_str} {trainer_config.str}'))
        logging.basicConfig(filename=os.path.join(self.out_folder, f'log/output_{self.datetime_str}'),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
        self.logger = logging.getLogger(f'{self.exp_name}')
        self.logger.info(f'Experiment: {self.exp_name}')

        print('Experiment: {}'.format(self.datetime_str), '\n')
        # print(self.exp_name, '\n')
        print('Process id {}'.format(os.getpid()), '\n')

        if trainer_config.device == 'gpu' and torch.cuda.is_available():
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
    def __init__(self, trainer_config, modules_config, dataset_folder, out_folder):
        super(DenseVideo2TextTrainer, self).__init__(trainer_config, modules_config, out_folder)

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

        # load vocabularies
        with open(os.path.join(dataset_folder, 'dense_corpus2.pkl'), "rb") as f:
            self.corpus = pickle.load(f)
            idx2op_dict = self.corpus[4]
            idx2word_dict = self.corpus[6]
            idx2pos_dict = self.corpus[8]
            idx2upos_dict = self.corpus[9]

        self.programs_vocab = Vocabulary.from_idx2word_dict(idx2op_dict, False)
        self.logger.info('Size of programs_vocab: {}'.format(len(self.programs_vocab)))
        print('Size of programs_vocab: {}'.format(len(self.programs_vocab)))

        self.caps_vocab = Vocabulary.from_idx2word_dict(idx2word_dict, False)
        self.logger.info('Size of caps_vocab: {}'.format(len(self.caps_vocab)))
        print('Size of caps_vocab: {}'.format(len(self.caps_vocab)))

        self.pos_vocab = Vocabulary.from_idx2word_dict(idx2pos_dict, False)
        self.logger.info('Size of pos_vocab: {}'.format(len(self.pos_vocab)))
        print('Size of pos_vocab: {}'.format(len(self.pos_vocab)))

        self.upos_vocab = Vocabulary.from_idx2word_dict(idx2upos_dict, False)
        self.logger.info('Size of upos_vocab: {}'.format(len(self.upos_vocab)))
        print('Size of upos_vocab: {}'.format(len(self.upos_vocab)), '\n')

        # Pretrained Embeddings
        pretrained_ope = None
        pretrained_we = torch.Tensor(self.corpus[7])
        pretrained_pe = None

        # Initialize data loaders
        self.__init_dense_loader()

        # Load ground-truth for computing evaluation metrics
        print('\nLoading ground truth...')
        self.__load_ground_truth()

        # Model
        print('\nInitializing the Model...')
        self.dense_captioner = DenseCaptioner(self.modules_config['dense_captioner_config'],
                                              self.modules_config['sem_tagger_config'],
                                              self.modules_config['syn_embedd_config'],
                                              self.modules_config['syn_tagger_config'],
                                              self.modules_config['avscn_dec_config'],
                                              self.modules_config['semsynan_dec_config'],
                                              self.modules_config['mm_config'],
                                              self.modules_config['vncl_cell_config'],
                                              progs_vocab=self.programs_vocab,
                                              pretrained_ope=pretrained_ope,
                                              caps_vocab=self.caps_vocab,
                                              pretrained_we=pretrained_we,
                                              pos_vocab=self.pos_vocab,
                                              pretrained_pe=pretrained_pe,
                                              device=self.device)

        # Optimizer
        print('\nInitializing the Optimizer...')
        if self.trainer_config.optimizer_config.optimizer_name == 'Adagrad':
            self.optimizer = optim.Adagrad([{'params': self.dense_captioner.mm_enc.parameters()},
                                            {'params': self.dense_captioner.rnn_cell.parameters()},
                                            {'params': self.dense_captioner.fc.parameters()},
                                            {'params': self.dense_captioner.clip_captioner.parameters()}],
                                           lr=self.trainer_config.learning_rate)
        else:
            self.optimizer = optim.Adam([{'params': self.dense_captioner.mm_enc.parameters()},
                                         {'params': self.dense_captioner.rnn_cell.parameters(), 'lr': self.trainer_config.optimizer_config.programmer_lr},
                                         {'params': self.dense_captioner.fc.parameters(), 'lr': self.trainer_config.optimizer_config.programmer_lr},
                                         {'params': self.dense_captioner.clip_captioner.avscn_dec.parameters(), 'lr': self.trainer_config.optimizer_config.captioning_lr},
                                         {'params': self.dense_captioner.clip_captioner.semsynan_dec.parameters(), 'lr': self.trainer_config.optimizer_config.captioning_lr},
                                         {'params': self.dense_captioner.clip_captioner.encoder.sem_model.parameters(), 'lr': self.trainer_config.optimizer_config.sem_enc_lr},
                                         {'params': self.dense_captioner.clip_captioner.encoder.syn_model.parameters(), 'lr': self.trainer_config.optimizer_config.syn_enc_lr}],
                                        lr=self.trainer_config.optimizer_config.learning_rate) #, weight_decay=.0001)

        # learning-rate decay scheduler
        lambda1 = lambda epoch: self.trainer_config.lr_decay_factor ** (epoch // 40)
        lambda2 = lambda epoch: self.trainer_config.lr_decay_factor ** (epoch // 40)
        lambda3 = lambda epoch: self.trainer_config.lr_decay_factor ** (epoch // 40)
        lambda4 = lambda epoch: self.trainer_config.lr_decay_factor ** (epoch // 40)
        lambda5 = lambda epoch: self.trainer_config.lr_decay_factor ** (epoch // 40)
        lambda6 = lambda epoch: self.trainer_config.lr_decay_factor ** (epoch // 40)
        lambda7 = lambda epoch: self.trainer_config.lr_decay_factor ** (epoch // 40)
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                        lr_lambda=[lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7])

        # Loss function
        self.criterion = DenseCaptioningLoss(config=trainer_config.criterion_config,
                                             c_max_len=self.max_words,
                                             p_max_len=self.max_prog,
                                             device=self.device)

        print('\n****We are ready to start the training process****\n')

    def __init_vocab(self, corpus):
        self.vocab = Vocabulary.from_words(['<pad>', '<start>', '<end>', '<unk>'])
        self.vocab.add_sentences(self.corpus)
        print('Vocabulary has {} words.'.format(len(self.vocab)))

    def __load_ground_truth(self):
        self.ref_programs, self.ref_captions, self.ref_densecaps = {}, {}, {}

        ref_progams_txt_path = {'val_1': os.path.join(self.dataset_folder,'val_1_ref_programs.txt')}
        ref_captions_txt_path = {'val_1': os.path.join(self.dataset_folder,'val_1_ref_captions.txt')}
        ref_densecap_json_path = {'val_1': os.path.join(self.dataset_folder,'val_1_ref_densecap.json')}

        ref_vidxs_blacklists = {'val_1': self.trainer_config.valid_blacklist}
        ref_cidxs_blacklists = {'val_1': [cidx for vidx in ref_vidxs_blacklists['val_1'] for cidx in self.corpus[1][1][self.corpus[1][0].index(vidx)]]}

        for phase in ['val_1']:
            self.ref_programs[phase] = load_texts(ref_progams_txt_path[phase], blacklist=ref_vidxs_blacklists[phase])
            self.ref_captions[phase] = load_texts(ref_captions_txt_path[phase], blacklist=ref_cidxs_blacklists[phase])
            with open(ref_densecap_json_path[phase], 'r') as f:
                self.ref_densecaps[phase] = json.load(f)
                for vidx in ref_vidxs_blacklists[phase]:
                    del self.ref_densecaps[phase][str(vidx)]

            print(f' For phase {phase} were sekiped:\n  vidxs: {ref_vidxs_blacklists[phase]}\n  cidxs: {ref_cidxs_blacklists[phase]}')

            # self.ref_programs[phase] = {k:self.ref_programs[phase][k] for k in range(3)}
            # self.ref_captions[phase] = {k:self.ref_captions[phase][k] for k in range(33844, 33844+8)}
            # self.ref_densecaps[phase] = {str(k):self.ref_densecaps[phase][str(k)] for k in range(3)}

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

    def __get_most_freq_words(self, caps, caps_upos, postags=['NOUN', 'ADJ', 'VERB'], words_to_discard=['<unk>']):
        widx2count, uidxs_to_use = {}, [self.upos_vocab(tag) for tag in postags]

        for v_caps, v_caps_upos in zip(caps, caps_upos):
            for cap, upos in zip(v_caps, v_caps_upos):
                for widx, uidx in zip(cap, upos):
                    if uidx in uidxs_to_use:
                        if widx not in widx2count.keys():
                            widx2count[widx] = 1
                        else:
                            widx2count[widx] += 1

        for w in words_to_discard:
            del widx2count[self.caps_vocab(w)]

        freq_words = heapq.nlargest(self.modules_config['sem_tagger_config'].out_size, widx2count, key=widx2count.get)
        self.logger.info(f'TAGs-IDXs: {freq_words}')
        self.logger.info(f'TAGs-words:' + ' '.join([self.caps_vocab.idx_to_word(idx) for idx in freq_words]))
        self.logger.info(f'TAGs-freq: {[widx2count[idx] for idx in freq_words]}')
        self.logger.info(f'TAGs-total freq of tags: {sum([widx2count[idx] for idx in freq_words])}')
        self.logger.info(f'TAGs-mean freq of tags: {np.mean([widx2count[idx] for idx in freq_words])}')

        return freq_words

    def __get_sem_enc(self, freq_words, caps, caps_upos, postags=['NOUN', 'ADJ', 'VERB']):
        uidxs_to_use = [self.upos_vocab(tag) for tag in postags]

        X = torch.zeros(len(caps), self.max_caps, self.modules_config['sem_tagger_config'].out_size)
        for i, (v_caps, v_caps_upos) in enumerate(zip(caps, caps_upos)):
            for j, (cap, upos) in enumerate(zip(v_caps, v_caps_upos)):
                for widx, uidx in zip(cap, upos):
                    if uidx in uidxs_to_use and widx in freq_words:
                        X[i, j, freq_words.index(widx)] = 1

        return X

    def __init_dense_loader(self):
        print('Initializing data loaders...')

        # get train split data
        print(' initializing train split data loader...')
        vidxs, cidxs, intervals, fps, progs, prog_lens, caps, pos, upos, cap_lens = extract_split_data_from_corpus(self.corpus, split=0)
        cidxs_t, intervals_t, caps_count_t, progs_t, caps_t, pos_t, upos_t, cap_lens_t = data2tensors(cidxs, intervals, progs, prog_lens, caps, pos, upos, cap_lens)
        self.max_prog = progs_t.size(1)
        self.max_caps = caps_t.size(1)
        self.max_words = caps_t.size(2)
        self.max_interval = torch.max(intervals_t.view(-1, 2)[:,1]-intervals_t.view(-1, 2)[:,0])
        self.last_interval_end = torch.max(intervals_t.view(-1, 2)[:,1])

        # determine the K most frequent words for semantic encodings from the train split
        freq_words = self.__get_most_freq_words(caps, upos, postags=self.modules_config['sem_tagger_config'].upos_tags, words_to_discard=self.modules_config['sem_tagger_config'].words_to_discard)

        # determine the ground truth for semantic enconding
        caps_sem_enc_t = self.__get_sem_enc(freq_words, caps, upos)

        # get train loader
        # h5_path = os.path.join(self.dataset_folder, self.trainer_config.features_filename)
        # self.h5_train = h5py.File(self.trainer_config.train_h5_file_path, 'r')
        # train_dataset = self.h5_train[self.trainer_config.h5_file_group_name]
        # train_loader = get_dense_loader(h5_dataset=train_dataset, vidxs=vidxs, vidxs_blcklist=self.trainer_config.valid_blacklist, cidxs=cidxs_t, 
        #                                 intervals=intervals_t, caps_count=caps_count_t, captions=caps_t, caps_sem_enc=caps_sem_enc_t, pos=pos_t, 
        #                                 upos=upos_t, cap_lens=cap_lens_t, progs=progs_t, prog_lens=prog_lens, batch_size=self.trainer_config.batch_size, 
        #                                 train=True, num_workers=trainer_config.loader_num_workers, pin_memory=trainer_config.loader_pin_memory)

        train_loader = get_dense_loader(h5_file_path=self.trainer_config.train_h5_file_path, h5_file_group_name=self.trainer_config.h5_file_group_name,
                                        vidxs=vidxs, vidxs_blcklist=self.trainer_config.valid_blacklist, cidxs=cidxs_t, 
                                        intervals=intervals_t, caps_count=caps_count_t, captions=caps_t, caps_sem_enc=caps_sem_enc_t, pos=pos_t, 
                                        upos=upos_t, cap_lens=cap_lens_t, progs=progs_t, prog_lens=prog_lens, batch_size=self.trainer_config.batch_size, 
                                        train=True, num_workers=trainer_config.loader_num_workers, pin_memory=trainer_config.loader_pin_memory)

        # get valid split data
        print(' initializing valid split data loader...')
        vidxs, cidxs, intervals, fps, progs, prog_lens, caps, pos, upos, cap_lens = extract_split_data_from_corpus(self.corpus, split=1)
        cidxs_t, intervals_t, caps_count_t, progs_t, caps_t, pos_t, upos_t, cap_lens_t = data2tensors(cidxs, intervals, progs, prog_lens, caps, pos, upos, cap_lens,
                                                                                                      self.max_prog, self.max_caps, self.max_words)
        # self.max_prog = max(self.max_prog, progs_t.size(1))
        # self.max_caps = max(self.max_caps, caps_t.size(1))
        # self.max_words = max(self.max_words, caps_t.size(2))
        # self.max_interval = max(self.max_interval, torch.max(intervals_t.view(-1, 2)[:,1]-intervals_t.view(-1, 2)[:,0]))
        # self.last_interval_end = max(self.last_interval_end, torch.max(intervals_t.view(-1, 2)[:,1]))

        # determine the ground truth for semantic enconding
        caps_sem_enc_t = self.__get_sem_enc(freq_words, caps, upos)

        # get valid loader
        # self.h5_val = h5py.File(self.trainer_config.valid_h5_file_path, 'r')
        # val_dataset = self.h5_val[self.trainer_config.h5_file_group_name]
        # val_loader = get_dense_loader(h5_dataset=val_dataset, vidxs=vidxs, vidxs_blcklist=self.trainer_config.valid_blacklist, cidxs=cidxs_t, 
        #                               intervals=intervals_t, caps_count=caps_count_t, captions=caps_t, caps_sem_enc=caps_sem_enc_t, pos=pos_t, 
        #                               upos=upos_t, cap_lens=cap_lens_t, progs=progs_t, prog_lens=prog_lens, batch_size=self.trainer_config.batch_size*3, 
        #                               train=False, num_workers=trainer_config.loader_num_workers, pin_memory=trainer_config.loader_pin_memory)

        val_loader = get_dense_loader(h5_file_path=self.trainer_config.valid_h5_file_path, h5_file_group_name=self.trainer_config.h5_file_group_name,
                                      vidxs=vidxs, vidxs_blcklist=self.trainer_config.valid_blacklist, cidxs=cidxs_t, 
                                      intervals=intervals_t, caps_count=caps_count_t, captions=caps_t, caps_sem_enc=caps_sem_enc_t, pos=pos_t, 
                                      upos=upos_t, cap_lens=cap_lens_t, progs=progs_t, prog_lens=prog_lens, batch_size=self.trainer_config.batch_size*3, 
                                      train=False, num_workers=trainer_config.loader_num_workers, pin_memory=trainer_config.loader_pin_memory)

        self.logger.info(f'Max program len: {self.max_prog}')
        self.logger.info(f'Max caption len: {self.max_words}')
        self.logger.info(f'Max intervals count: {self.max_caps}')
        self.logger.info(f'Max interval len: {int(self.max_interval)}')
        self.logger.info(f'Last interval end: {int(self.last_interval_end)}')

        print(f' Max program len: {self.max_prog}')
        print(f' Max caption len: {self.max_words}')
        print(f' Max intervals count: {self.max_caps}')
        print(f' Max interval len: {int(self.max_interval)}')
        print(f' Last interval end: {int(self.last_interval_end)}')

        self.loaders = {'train': train_loader, 'val_1': val_loader}

    def __get_sentences(self, all_outputs, all_video_ids):
        predicted_sentences = {}
        for outputs, video_ids in zip(all_outputs, all_video_ids):
            for predicted_tokens, vid in zip(outputs, video_ids):
                predicted_sentences[vid] = [self.__decode_from_tokens(predicted_tokens)]
        return predicted_sentences

    def __process_batch(self, video_feats, feats_count, gt_intervals, gt_caps_count, gt_captions, gt_caps_sem_enc, gt_pos, gt_upos, gt_cap_lens, gt_program, gt_prog_len,
                        teacher_forcing_ratio=.5, phase='train', use_rl=False):
        bsz = video_feats[0].size(0)

        # Move all tensors to device
        video_feats = [f.to(self.device) for f in video_feats]
        gt_intervals = gt_intervals.to(self.device)
        gt_captions = gt_captions.to(self.device)
        gt_pos = gt_pos.to(self.device)
        gt_upos = gt_upos.to(self.device)
        gt_cap_lens = gt_cap_lens.to(self.device)
        gt_program = gt_program.to(self.device)
        gt_prog_len = gt_prog_len.to(self.device)
        gt_caps_sem_enc = gt_caps_sem_enc.to(self.device)

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            # truncate at least minimum program length steps, considering at least a caption for each video
            if phase=='train':
                truncate_prog_at = int(max(torch.min(gt_prog_len), torch.max((gt_intervals[:,0,1])*2 - gt_intervals[:,0,0] + 1)))
                self.logger.info(f'the gt programs of len {gt_prog_len} will be truncated around {truncate_prog_at}')
                
                temp_prog_pos = torch.zeros(gt_intervals.size(0), gt_intervals.size(1)).to(gt_intervals.device)
                for i in range(gt_intervals.size(1)):
                    if i == 0:
                        temp_prog_pos[:,i] = gt_intervals[:,i,1]*2 - gt_intervals[:,i,0] + 1
                    else:
                        temp_prog_pos[:,i] = temp_prog_pos[:,i-1] + (gt_intervals[:,i,0] - gt_intervals[:,i-1,1]) + (gt_intervals[:,i,1] - gt_intervals[:,i,0])*2 + 1

                self.logger.info(f'gt caps count: {gt_caps_count}')
                gt_caps_count = torch.sum((gt_intervals[:,:,1] > 0) * (temp_prog_pos < truncate_prog_at), dim=1)
                self.logger.info(f'tuncated gt caps count: {gt_caps_count}')
            else:
                truncate_prog_at = None

            prog_logits, program, caps_logits, caps_sem_enc, pos_tags_logits, captions, intervals, caps_count = self.dense_captioner(video_features=video_feats,
                                                                                                                                     feats_count=feats_count,
                                                                                                                                     prog_len=truncate_prog_at,
                                                                                                                                     teacher_forcing_p=teacher_forcing_ratio,
                                                                                                                                     gt_program=gt_program,
                                                                                                                                     gt_captions=gt_captions,
                                                                                                                                     gt_caps_count=gt_caps_count,
                                                                                                                                     gt_sem_enc=gt_caps_sem_enc,
                                                                                                                                     gt_pos=gt_pos,
                                                                                                                                     gt_intervals=gt_intervals,
                                                                                                                                     max_prog=self.max_prog,
                                                                                                                                     max_caps=self.max_caps,
                                                                                                                                     max_cap=self.max_words)
            # video_encoded = self.encoder(cnn_feats, c3d_feats, i3d_feats, eco_feats, eco_sem_feats, tsm_sem_feats, cnn_globals, cnn_sem_globals, tags_globals, res_eco_globals)

            # outputs, tokens = self.decoder(video_encoded, targets if phase == 'train' else None, teacher_forcing_ratio)

            # Straighten the output (removing the part of the pad) and then straighten it
            # outputs = torch.cat([decode[j][:target_lens[j]] for j in range(bsz)], dim=0)

            # Straighten the target (remove the part of the pad) and straighten it
            # targets = torch.cat([targets[j][:target_lens[j]] for j in range(bsz)], dim=0)

            # Evaluate the loss function
            loss, prog_loss, cap_loss, sem_enc_loss, pos_loss, iou_loss = self.criterion(gt_captions, gt_cap_lens, caps_logits, gt_caps_sem_enc,
                                                                                         caps_sem_enc, gt_pos, pos_tags_logits, gt_program,
                                                                                         gt_prog_len, prog_logits, gt_intervals, intervals,
                                                                                         gt_caps_count, caps_count, truncate_prog_at)
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
            # compute backward pass for somputing the gradients
            self.logger.info('loss backward....')
            loss.backward()

            # clip gradients to prevent NaNs in the prog-loss
            # nn.utils.clip_grad_norm_(self.dense_captioner.rnn_cell.parameters(), 0.5) 

            # update the parameters
            self.logger.info('optimizer step...')
            self.optimizer.step()

        return loss, prog_loss, cap_loss, sem_enc_loss, pos_loss, iou_loss, program, captions, intervals, caps_count

    def __evaluate(self, predicted_sentences, phase):
        scores = score(self.ground_truth[phase], predicted_sentences)
        weights = {'Bleu_1':0., 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 1.4, 'CIDEr': 1.17, 'METEOR': 2., 'ROUGE_L':1.}
        scores['All_Metrics'] = sum([scores[k] * weights[k] for k in scores.keys()])
        return scores

    def __save_checkpoint(self, epoch, save_checkpoints_dir, new_best=False):
        if new_best:
            chkpt_filename = f'best_chkpt_{epoch}.pt'
        else:
            chkpt_filename = f'chkpt_{epoch}.pt'

        if self.last_saved_epoch == epoch:
            # the checkpoint was saved after training at this epoch, we don't need to save it again, we copy it only
            copyfile(os.path.join(save_checkpoints_dir, f'chkpt_{epoch}.pt'), os.path.join(save_checkpoints_dir, chkpt_filename))
            print(' the checkpoint was copied only')
        else:
            # save the new checkpoint (best or not)
            torch.save(obj={'epoch': epoch,
                            'trainer_config': self.trainer_config,
                            'modules_config': self.modules_config,
                            'dense_captioner': self.dense_captioner.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'best_metrics': self.best_metrics},
                        f=os.path.join(save_checkpoints_dir, chkpt_filename))
            print(' saved')

        # remove previously saved
        if new_best and self.last_best_saved_epoch != -1:
            os.remove(os.path.join(save_checkpoints_dir, f'best_chkpt_{self.last_best_saved_epoch}.pt'))
        elif not new_best and self.last_saved_epoch != -1:
            os.remove(os.path.join(save_checkpoints_dir, f'chkpt_{self.last_saved_epoch}.pt'))

        # update the last saved epoch
        if new_best:
            self.last_best_saved_epoch = epoch
        else:
            self.last_saved_epoch = epoch

    def __process_results(self, metrics_results, prediction, phase, epoch, save_checkpoints_dir, component):
        self.logger.info(f'{phase} set metrics for {component}: {metrics_results}')
        min_metrics = ['Recall']
        for name, result in metrics_results.items():
            self.writer.add_scalar(f'end2end/{phase}-{component}-{name}', result, epoch)
            if name in self.best_metrics[component][phase] and ((name in min_metrics and result > self.best_metrics[component][phase][name][1]) or (name not in min_metrics and result < self.best_metrics[component][phase][name][1])):
                self.best_metrics[component][phase][name] = (epoch, result)
                if name in ['Bleu_4','METEOR', 'ROUGE_L', 'CIDEr', 'All_Metrics']:
                    self.early_stop = 0
                    torch.save(prediction, os.path.join(save_checkpoints_dir, f'chkpt_{epoch}_{component}_output.json'))
                    # with open(os.path.join(save_checkpoints_dir, f'chkpt_{epoch}_{component}_output.json'), 'w') as f:
                    #    json.dump(prediction, f)
                if component=='densecap' and name == 'METEOR' and phase == 'val_1':
                    print('saving best checkpoint...')
                    self.__save_checkpoint(epoch, save_checkpoints_dir, True)

    def train_model(self, resume=False, checkpoint_path=None, min_num_epochs=50, early_stop_limit=10):
        parallel_pool = Pool()
        # self.logger.info('Training captioning model on [{}] dataset with [{}] encoder and [{}] decoder'
        #                  .format(self.config.dataset_name, self.encoder_name, self.decoder_name))

        save_checkpoints_dir = os.path.join(self.out_folder, 'models', self.trainer_config.str, self.datetime_str)
        if not os.path.exists(save_checkpoints_dir):
            os.makedirs(save_checkpoints_dir)

        if resume and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            begin_epoch = checkpoint['epoch']
            self.best_metrics = checkpoint['best_metrics']
        else:
            begin_epoch = 0
            self.best_metrics = {'programmer': {}, 'captioning': {}, 'densecap': {}}
            for p in ['val_1']:
                # self.best_metrics['programmer'][p] = {'Bleu_1': (0, 0), 'Bleu_2': (0, 0), 'Bleu_3': (0, 0), 'Bleu_4': (0, 0),
                #                              'METEOR': (0, 0), 'ROUGE_L': (0, 0), 'CIDEr': (0, 0), 'SPICE': (0, 0), 'All_Metrics': (0, 0)}
                self.best_metrics['captioning'][p] = {'Bleu_1': (0, 0), 'Bleu_2': (0, 0), 'Bleu_3': (0, 0), 'Bleu_4': (0, 0),
                                                      'METEOR': (0, 0), 'ROUGE_L': (0, 0), 'CIDEr': (0, 0), 'SPICE': (0, 0),
                                                      'All_Metrics': (0, 0)}
                self.best_metrics['densecap'][p] = {'Bleu_1': (0, 0), 'Bleu_2': (0, 0), 'Bleu_3': (0, 0), 'Bleu_4': (0, 0),
                                                    'METEOR': (0, 0), 'ROUGE_L': (0, 0), 'CIDEr': (0, 0), 'SPICE': (0, 0),
                                                    'Recall': (0, float('inf')), 'Precision': (0, 0), 'All_Metrics': (0, 0)}

        self.dense_captioner.to(self.device)
        print('\nParameters of Dense Captioner model:\n')
        total_size = 0
        for n, p in self.dense_captioner.named_parameters():
             print(n, p.size(), p.device)
             total_size += torch.numel(p)
        print(' total size: ', (total_size*8)/(1024**3), '\n')

        # Start training process
        self.early_stop, self.last_saved_epoch, self.last_best_saved_epoch = 0, -1, -1
        time_phases = {'train': 0, 'val_1': 0}
        prog_metrics_results, cap_metrics_results, densecap_metrics_results = None, None, None
        for epoch in range(begin_epoch, 1000):
            k = self.trainer_config.convergence_speed_factor
            teacher_forcing_ratio = max(.6, k / (k + np.exp(epoch / k)))  # inverse sigmoid decay
            self.writer.add_scalar('end2end/teacher_forcing_ratio', teacher_forcing_ratio, epoch)

            loss_phases = {'train': 0, 'val_1': 0}
            for phase in ['train', 'val_1']:
                if phase == 'train':
                    self.dense_captioner.train()
                else:
                    self.dense_captioner.eval()

                # predicted_sentences = {}
                time_start_epoch = time.perf_counter()
                loss_count = 0
                all_programs, all_captions, all_prog_ids, all_caps_ids, all_intervals, all_tstamps = [], [], [], [], [], []
                for i, (vidx, cidxs, cnn, c3d, feats_count, tstamps, gt_intervals, gt_caps_count, gt_caps, gt_caps_sem_enc, gt_pos, gt_upos, gt_cap_lens, gt_prog, gt_prog_len) in enumerate(self.loaders[phase], start=1):
                    video_feats = [cnn, c3d]
                    use_rl = False
                    loss, prog_loss, cap_loss, sem_enc_loss, pos_loss, iou_loss, program, captions, intervals, caps_count = self.__process_batch(video_feats, feats_count, gt_intervals, gt_caps_count, gt_caps, gt_caps_sem_enc, gt_pos, gt_upos,
                                                                                                                                       gt_cap_lens, gt_prog, gt_prog_len, teacher_forcing_ratio, phase, use_rl=use_rl)
                    loss_count += loss.item()

                    self.writer.add_scalar('end2end/{}-iters-loss'.format(phase), loss, epoch * len(self.loaders[phase]) + i)
                    self.writer.add_scalar('end2end/{}-iters-prog_loss'.format(phase), prog_loss, epoch * len(self.loaders[phase]) + i)
                    self.writer.add_scalar('end2end/{}-iters-cap_loss'.format(phase), cap_loss, epoch * len(self.loaders[phase]) + i)
                    self.writer.add_scalar('end2end/{}-iters-sem_enc_loss'.format(phase), sem_enc_loss, epoch * len(self.loaders[phase]) + i)
                    self.writer.add_scalar('end2end/{}-iters-pos_tag_loss'.format(phase), pos_loss, epoch * len(self.loaders[phase]) + i)
                    self.writer.add_scalar('end2end/{}-iters-iou_loss'.format(phase), iou_loss, epoch * len(self.loaders[phase]) + i)

                    lrs = self.lr_scheduler.get_last_lr()
                    log_msg = '\rEpoch:{0:03d} Phase:{1:6s} Iter:{2:04d}/{3:04d} lr:{4:.6f} Loss:{5:9.4f} [cap-loss:{6:9.4f} prog-loss:{7:9.4f} sem-enc-loss:{8:9.4f} pos-tag-loss:{9:9.4f} iou-loss:{10:9.4f}]'.format(epoch, phase, i, 
                                                                                                                                                                                                            len(self.loaders[phase]), 
                                                                                                                                                                                                            lrs[0], loss.item(), 
                                                                                                                                                                                                            cap_loss.item(), 
                                                                                                                                                                                                            prog_loss.item(), 
                                                                                                                                                                                                            sem_enc_loss.item(),
                                                                                                                                                                                                            pos_loss.item(),
                                                                                                                                                                                                            iou_loss.item())
                    self.logger.info(log_msg)
                    sys.stdout.write(log_msg)

                    pred = decode_from_tokens(self.programs_vocab, program[0], until_eos=False, max_length=gt_prog_len[0])
                    gt = decode_from_tokens(self.programs_vocab, gt_prog[0], until_eos=False, max_length=gt_prog_len[0])
                    # print('\nPRED PROG:', pred)
                    # print('\nPRED INTERV:', intervals[0, :gt_caps_count[0]])
                    # print('\nGT PROG:', gt)
                    # print('\nGT INTERV:', gt_intervals[0, :gt_caps_count[0]])
                    sample_diff = sum([s1!=s2 for s1, s2 in zip(pred.split(' '), gt.split(' '))])
                    self.logger.info(f'sample-pred-prog-diff: {sample_diff}')

                    if phase != 'train':
                        # save programs and the videos' idx for computing evaluation metrics
                        all_programs.append(program.to('cpu'))
                        all_prog_ids.append(vidx)

                        # save captions and the captions' idx for computing evaluation metrics (only the first caps_count captions are evaluated)
                        all_captions.append((captions.to('cpu'), caps_count.to('cpu'), gt_caps_count.to('cpu')))
                        all_caps_ids.append(cidxs)

                        # save intervals for computing evaluation metrics
                        all_intervals.append(intervals.to('cpu'))
                        all_tstamps.append(tstamps)

                        # for predicted_tokens, vid in zip(outputs, video_ids):
                        #     predicted_sentences[vid] = [self.__decode_from_tokens(predicted_tokens)]

                        # logging sample sentences of prediction and target
                        # self.logger.info('[vid:{}]'.format(video_ids[0]))
                        # self.logger.info('\nWE: {}\nGT: {}'.format(predicted_sentences[video_ids[0]],
                        #                                            self.__decode_from_tokens(captions[0].squeeze())))

                # for sanity, replace the last checkpoint when epoch is power of 2
                if phase == 'train' and (epoch == 0 or not(epoch & (epoch-1))):
                    print('saving checkpoint...')
                    self.__save_checkpoint(epoch, save_checkpoints_dir, False)

                avg_loss = loss_count/len(self.loaders[phase])
                loss_phases[phase] = avg_loss
                self.writer.add_scalar('end2end/{}-epochs-avg-loss'.format(phase), avg_loss, epoch)

                if phase != 'train':
                    self.early_stop += 1
                    # predicted_sentences = pool.apply_async(self.__get_sentences, [all_outputs, all_video_ids])

                    #if cap_metrics_results is not None:
                        # get async results
                        # cap_metrics_results, pred_caps = cap_metrics_results.get()
                        # prog_metrics_results, pred_progs = prog_metrics_results.get()
                        # densecap_metrics_results, pred_intervals = densecap_metrics_results.get()

                    # print('evaluating progs...')
                    # prog_metrics_results, pred_progs = evaluate_from_tokens(self.programs_vocab, all_programs, all_prog_ids, self.ref_programs[phase], False)
                    print('evaluating captions (basic)...')
                    cap_metrics_results, pred_caps = evaluate_from_tokens(self.caps_vocab, all_captions, all_caps_ids, self.ref_captions[phase])
                    print('evaluating captions (dense)...')
                    densecap_metrics_results, pred_intervals = densecap_evaluate_from_tokens(self.caps_vocab, all_prog_ids, all_tstamps, all_intervals, all_captions, self.ref_densecaps[phase])

                    # process results, saving the checkpoint if any improvement occurs
                    # self.__process_results(prog_metrics_results, pred_progs, phase, epoch-1, save_checkpoints_dir, 'programmer')
                    self.__process_results(cap_metrics_results, pred_caps, phase, epoch, save_checkpoints_dir, 'captioning')
                    self.__process_results(densecap_metrics_results, pred_intervals, phase, epoch, save_checkpoints_dir, 'densecap')

                    # report results if any improvement occurs
                    if self.early_stop == 0:
                        log_msg = f'\n IMPROVEMENT ON {phase} at epoch {epoch} !'

                        # log_msg += '\n Programmer metrics: \n   '
                        # log_msg += '\t'.join([f'{k}:({e:03d}, {v:.3f})' for k, (e, v) in self.best_metrics['programmer'][phase].items()])

                        log_msg += '\n  Captioning metrics: \n   '
                        log_msg += '\t'.join([f'{k}:({e:03d}, {v:.3f})' for k, (e, v) in self.best_metrics['captioning'][phase].items()])

                        log_msg += '\n  DenseCaptioning metrics: \n   '
                        log_msg += '\t'.join([f'{k}:({e:03d}, {v:.3f})' for k, (e, v) in self.best_metrics['densecap'][phase].items()])

                        print(log_msg, '\n')
                        self.logger.info(log_msg)

                    # prog_metrics_results = parallel_pool.apply_async(evaluate_from_tokens, [self.programs_vocab, all_programs, all_prog_ids, self.ref_programs[phase], False])
                    # cap_metrics_results = parallel_pool.apply_async(evaluate_from_tokens, [self.caps_vocab, all_captions, all_caps_ids, self.ref_captions[phase]])
                    # densecap_metrics_results = parallel_pool.apply_async(densecap_evaluate_from_tokens, [self.caps_vocab, all_intervals, all_captions, all_caps_ids, self.ref_densecaps[phase]])

                time_phases[phase] += time.perf_counter() - time_start_epoch

            log_msg = '\n'
            for k, v in loss_phases.items():
                log_msg += ' {0}-avg-loss:{1:9.4f}'.format(k, v)
            for k, v in time_phases.items():
                log_msg += ' {0}-avg-lime:{1:9.3f}h'.format(k, (v/3600)/(epoch+1))
            log_msg += ' teacher_forcing_ratio:{0:.3f} enc-lr:{1:.6f} dec-lr:{1:.6f}'.format(teacher_forcing_ratio, lrs[0], lrs[1])
            # vid = video_ids[0]
            # log_msg += '\n[vid {}]:\nWE: {}\nGT: {}'.format(vid, predicted_sentences[vid], self.ground_truth['valid'][vid])
            
            self.logger.info(log_msg)
            sys.stdout.write(log_msg+'\n')

            # check if the training must be early sopped
            if epoch >= min_num_epochs and self.early_stop >= early_stop_limit * 2:
                # get async results
                # cap_metrics_results, pred_caps = cap_metrics_results.get()
                # prog_metrics_results, pred_progs = prog_metrics_results.get()
                # densecap_metrics_results, pred_intervals = densecap_metrics_results.get()

                # self.__process_results(prog_metrics_results, pred_caps, phase, epoch-1, save_checkpoints_dir, 'programmer')
                self.__process_results(cap_metrics_results, pred_progs, phase, epoch-1, save_checkpoints_dir, 'captioning')
                self.__process_results(densecap_metrics_results, pred_intervals, phase, epoch-1, save_checkpoints_dir, 'densecap')

                msg = '----early stopped at epoch {} after {} without any improvement-----'.format(epoch, early_stop_limit)
                self.logger.debug(msg)
                print(msg)
                break

            self.writer.add_scalar('end2end/learning-rate', self.optimizer.param_groups[0]['lr'], epoch)
            self.lr_scheduler.step()

        # close h5 files
        # self.h5_train.close()
        # self.h5_val.close()
        for loader in self.loaders.values():
            loader.close_h5_file()

        # log best results
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
    print('Loading configuration file...')
    config_path = os.path.join(args.dataset_folder, 'train_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    trainer_config = ConfigDict(config['trainer_config'])
    trainer_config.str = get_trainer_str(trainer_config)
    print(trainer_config.str)

    dense_captioner_config = ConfigDict(config['dense_captioner_config'])
    dense_captioner_config.str = get_dense_captioner_str(dense_captioner_config)
    print(dense_captioner_config.str)

    sem_tagger_config = ConfigDict(config['sem_tagger_config'])
    sem_tagger_config.str = get_sem_tagger_str(sem_tagger_config)
    print(sem_tagger_config.str)

    syn_embedd_config = ConfigDict(config['syn_embedd_config'])
    syn_embedd_config.str = get_syn_embedd_str(syn_embedd_config)
    print(syn_embedd_config.str)

    syn_tagger_config = ConfigDict(config['syn_tagger_config'])
    syn_tagger_config.str = get_syn_tagger_str(syn_tagger_config)
    print(syn_tagger_config.str)

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

    print('Initializing the experiment.......')
    modules_config = {'dense_captioner_config': dense_captioner_config,
                      'sem_tagger_config': sem_tagger_config,
                      'syn_embedd_config': syn_embedd_config,
                      'syn_tagger_config': syn_tagger_config,
                      'avscn_dec_config': avscn_dec_config,
                      'semsynan_dec_config': semsynan_dec_config,
                      'mm_config': mm_config,
                      'vncl_cell_config': vncl_cell_config}
    # modules_config = [sem_tagger_config, syn_embedd_config, avscn_dec_config, semsynan_dec_config, vncl_cell_config]
    trainer = DenseVideo2TextTrainer(trainer_config, modules_config, args.dataset_folder, args.output_folder)

    print('Training.........')
    # try:
    best_results = trainer.train_model(resume=False,
                                       checkpoint_path='',
                                       early_stop_limit=trainer_config.early_stop_limit)
    print('Best results in the test set: {}'.format(str(best_results)))
    # except Exception as e:
    #     print(f'An error occurred during training/validation process: {e}')
    #     trainer.h5_train.close()
    #     trainer.h5_val.close()

    print('--- END ---')
