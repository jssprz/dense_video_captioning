from operator import mul
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

import numpy as np
from numpy import linspace
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from utils import (
    get_freer_gpu,
    get_gpu_temps,
    decode_from_tokens,
    load_texts,
    evaluate_from_tokens,
    densecap_evaluate_from_tokens,
    multilabel_evaluate_from_logits,
    get_tf_ratio,
    get_trainer_str,
    get_dense_captioner_str,
    get_sem_tagger_str,
    get_syn_embedd_str,
    get_syn_tagger_str,
    get_avscn_decoder_str,
    get_semsynan_decoder_str,
    get_mm_str,
    get_vncl_cell_str,
    get_proposals_tagger_str,
)
from vocabulary import Vocabulary
from configuration_dict import ConfigDict
from loader import extract_split_data_from_corpus, data2tensors, get_dense_loader
from model.dense_captioner import DenseCaptioner
from loss import DenseCaptioningLoss

if os.name == "posix":
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


class Trainer:
    def __init__(self, trainer_config, modules_config, out_folder):
        self.trainer_config = trainer_config
        self.modules_config = modules_config

        self.exp_name = f"({trainer_config.str})"
        for config in modules_config.values():
            self.exp_name += f" ({config.str})"

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
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.out_folder, "log/runs/", f"{self.datetime_str} {trainer_config.str}",)
        )
        logging.basicConfig(
            filename=os.path.join(self.out_folder, f"log/output_{self.datetime_str}"),
            filemode="a",
            format="%(asctime)s,%(msecs)d %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.INFO,
        )
        self.logger = logging.getLogger(f"{self.exp_name}")
        self.logger.info(f"Experiment: {self.exp_name}")

        print("Experiment: {}".format(self.datetime_str), "\n")
        # print(self.exp_name, '\n')
        print("Process id {}".format(os.getpid()), "\n")

        if trainer_config.device == "gpu" and torch.cuda.is_available():
            freer_gpu_id = get_freer_gpu()
            self.device = torch.device("cuda:{}".format(freer_gpu_id))
            torch.cuda.empty_cache()
            self.logger.info("Running on cuda:{} device".format(freer_gpu_id))
            print("Running on cuda:{} device".format(freer_gpu_id))
        else:
            self.device = torch.device("cpu")
            self.logger.info("Running on cpu device")
            print("Running on cpu device")


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
        with open(os.path.join(dataset_folder, "dense_corpus3.pkl"), "rb") as f:
            self.corpus = pickle.load(f)
            idx2op_dict = self.corpus[4]
            idx2word_dict = self.corpus[6]
            idx2pos_dict = self.corpus[8]
            idx2upos_dict = self.corpus[9]

        self.programs_vocab = Vocabulary.from_idx2word_dict(idx2op_dict, False)
        self.logger.info("Size of programs_vocab: {}".format(len(self.programs_vocab)))
        print("Size of programs_vocab: {}".format(len(self.programs_vocab)))

        self.caps_vocab = Vocabulary.from_idx2word_dict(idx2word_dict, False)
        self.logger.info("Size of caps_vocab: {}".format(len(self.caps_vocab)))
        print("Size of caps_vocab: {}".format(len(self.caps_vocab)))

        self.pos_vocab = Vocabulary.from_idx2word_dict(idx2pos_dict, False)
        self.logger.info("Size of pos_vocab: {}".format(len(self.pos_vocab)))
        print("Size of pos_vocab: {}".format(len(self.pos_vocab)))

        self.upos_vocab = Vocabulary.from_idx2word_dict(idx2upos_dict, False)
        self.logger.info("Size of upos_vocab: {}".format(len(self.upos_vocab)))
        print("Size of upos_vocab: {}".format(len(self.upos_vocab)), "\n")

        # Pretrained Embeddings
        pretrained_ope = None
        pretrained_we = torch.Tensor(self.corpus[7])
        pretrained_pe = None

        # Initialize data loaders
        self.__init_dense_loader()

        # Load ground-truth for computing evaluation metrics
        print("\nLoading ground truth...")
        self.__load_ground_truth()

        # Model
        print("\nInitializing the Model...")
        self.dense_captioner = DenseCaptioner(
            self.modules_config["dense_captioner_config"],
            self.modules_config["sem_tagger_config"],
            self.modules_config["syn_embedd_config"],
            self.modules_config["syn_tagger_config"],
            self.modules_config["avscn_dec_config"],
            self.modules_config["semsynan_dec_config"],
            self.modules_config["mm_config"],
            self.modules_config["vncl_cell_config"],
            self.modules_config["proposals_tagger_config"],
            num_proposals=self.num_proposals,
            progs_vocab=self.programs_vocab,
            pretrained_ope=pretrained_ope,
            caps_vocab=self.caps_vocab,
            pretrained_we=pretrained_we,
            pos_vocab=self.pos_vocab,
            pretrained_pe=pretrained_pe,
            device=self.device,
        )

        # Optimizer
        print("\nInitializing the Optimizer...")
        opt_conf = self.trainer_config.optimizer_config
        if opt_conf.optimizer_name == "Adagrad":
            self.optimizer = optim.Adagrad(
                [
                    {"params": self.dense_captioner.mm_enc.parameters()},
                    {"params": self.dense_captioner.proposal_enc.parameters()},
                    {"params": self.dense_captioner.rnn_cell.parameters()},
                    {"params": self.dense_captioner.fc.parameters()},
                    {"params": self.dense_captioner.clip_captioner.parameters()},
                ],
                lr=self.trainer_config.learning_rate,
            )
        else:
            self.optimizer = optim.Adam(
                [
                    # {"params": self.dense_captioner.mm_enc.parameters(), "lr": opt_conf.programmer_lr,},
                    {"params": self.dense_captioner.parameters(), "lr": opt_conf.proposals_lr,},
                    # {"params": self.dense_captioner.embedding.parameters(), "lr": opt_conf.programmer_lr,},
                    # {"params": self.dense_captioner.rnn_cell.parameters(), "lr": opt_conf.programmer_lr,},
                    # {"params": self.dense_captioner.fc.parameters(), "lr": opt_conf.programmer_lr,},
                    # {
                    #     "params": self.dense_captioner.clip_captioner.avscn_dec.parameters(),
                    #     "lr": opt_conf.captioning_lr,
                    # },
                    # {
                    #     "params": self.dense_captioner.clip_captioner.semsynan_dec.parameters(),
                    #     "lr": opt_conf.captioning_lr,
                    # },
                    # {
                    #     "params": self.dense_captioner.clip_captioner.encoder.sem_model.parameters(),
                    #     "lr": opt_conf.sem_enc_lr,
                    # },
                    # {
                    #     "params": self.dense_captioner.clip_captioner.encoder.syn_model.parameters(),
                    #     "lr": opt_conf.syn_enc_lr,
                    # },
                ],
                lr=opt_conf.learning_rate,
            )  # , weight_decay=.0001)

        # learning-rate decay scheduler
        lambda1 = lambda epoch: opt_conf.lr_decay_factor ** (epoch // opt_conf.programmer_lr_decay_epochs)
        # lambda2 = lambda epoch: opt_conf.lr_decay_factor ** (epoch // opt_conf.programmer_lr_decay_epochs)
        # lambda3 = lambda epoch: opt_conf.lr_decay_factor ** (epoch // opt_conf.programmer_lr_decay_epochs)
        # lambda4 = lambda epoch: opt_conf.lr_decay_factor ** (epoch // opt_conf.programmer_lr_decay_epochs)
        # lambda5 = lambda epoch: opt_conf.lr_decay_factor ** (epoch // opt_conf.programmer_lr_decay_epochs)
        # lambda6 = lambda epoch: self.trainer_config.lr_decay_factor ** (epoch // 40)
        # lambda7 = lambda epoch: self.trainer_config.lr_decay_factor ** (epoch // 40)
        # lambda8 = lambda epoch: self.trainer_config.lr_decay_factor ** (epoch // 40)
        # lambda9 = lambda epoch: self.trainer_config.lr_decay_factor ** (epoch // 40)

        self.lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=[lambda1],)

        # Loss function
        self.criterion = DenseCaptioningLoss(
            config=trainer_config.criterion_config,
            c_max_len=self.max_words,
            p_max_len=self.max_prog,
            s_prop_pos_weights=self.s_prop_pos_weights,
            e_prop_pos_weights=self.e_prop_pos_weights,
            device=self.device,
        )

        print("\n****We are ready to start the training process****\n")

    def __load_ground_truth(self):
        self.ref_programs, self.ref_captions, self.ref_densecaps = {}, {}, {}

        ref_progams_txt_path = {"val_1": os.path.join(self.dataset_folder, "val_1_ref_programs.txt")}
        ref_captions_txt_path = {"val_1": os.path.join(self.dataset_folder, "val_1_ref_captions.txt")}
        ref_densecap_json_path = {"val_1": os.path.join(self.dataset_folder, "val_1_ref_densecap.json")}

        ref_vidxs_blacklists = {"val_1": self.trainer_config.valid_blacklist}
        ref_cidxs_blacklists = {
            "val_1": [
                cidx
                for vidx in ref_vidxs_blacklists["val_1"]
                for cidx in self.corpus[1][1][self.corpus[1][0].index(vidx)]
            ]
        }

        for phase in ["val_1"]:
            self.ref_programs[phase] = load_texts(ref_progams_txt_path[phase], blacklist=ref_vidxs_blacklists[phase])
            self.ref_captions[phase] = load_texts(ref_captions_txt_path[phase], blacklist=ref_cidxs_blacklists[phase])
            with open(ref_densecap_json_path[phase], "r") as f:
                self.ref_densecaps[phase] = json.load(f)
                for vidx in ref_vidxs_blacklists[phase]:
                    del self.ref_densecaps[phase][str(vidx)]

            print(
                f" For phase {phase} were skiped:\n  vidxs: {ref_vidxs_blacklists[phase]}\n  cidxs: {ref_cidxs_blacklists[phase]}"
            )

            # self.ref_programs[phase] = {k:self.ref_programs[phase][k] for k in range(3)}
            # self.ref_captions[phase] = {k:self.ref_captions[phase][k] for k in range(33844, 33844+8)}
            # self.ref_densecaps[phase] = {str(k):self.ref_densecaps[phase][str(k)] for k in range(3)}

    def __get_most_freq_words(self, caps, caps_upos, postags=["NOUN", "ADJ", "VERB"], words_to_discard=["<unk>"]):
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

        freq_words = heapq.nlargest(self.modules_config["sem_tagger_config"].out_size, widx2count, key=widx2count.get,)
        self.logger.info(f"TAGs-IDXs: {freq_words}")
        self.logger.info(f"TAGs-words:" + " ".join([self.caps_vocab.idx_to_word(idx) for idx in freq_words]))
        self.logger.info(f"TAGs-freq: {[widx2count[idx] for idx in freq_words]}")
        self.logger.info(f"TAGs-total freq of tags: {sum([widx2count[idx] for idx in freq_words])}")
        self.logger.info(f"TAGs-mean freq of tags: {np.mean([widx2count[idx] for idx in freq_words])}")

        return freq_words

    def __get_sem_enc(self, freq_words, caps, caps_upos, postags=["NOUN", "ADJ", "VERB"]):
        uidxs_to_use = [self.upos_vocab(tag) for tag in postags]

        total_num_caps = 0
        X = torch.zeros(len(caps), self.max_caps, len(freq_words))
        for i, (v_caps, v_caps_upos) in enumerate(zip(caps, caps_upos)):
            total_num_caps += len(v_caps)
            for j, (cap, upos) in enumerate(zip(v_caps, v_caps_upos)):
                for widx, uidx in zip(cap, upos):
                    if uidx in uidxs_to_use and widx in freq_words:
                        X[i, j, freq_words.index(widx)] = 1

        # total of activations for each tag
        pos_samples = X.sum(dim=0).sum(dim=0)  # (len(freq_words), )

        # total number of deactivations for each tag
        neg_samples = torch.tensor(total_num_caps).repeat(len(freq_words)) - pos_samples

        pos_weights = neg_samples / pos_samples

        return X, pos_weights

    def __get_interval_mask(
        self, intervals, caps_count, max_num_chunks, proposals=None, num_estimates=128, min_count_per_proposal=20
    ):
        # compute the length of all intervals, including padding region
        aux = intervals[:, :, 1] - intervals[:, :, 0]

        # filter the length of real intervals only, discarding the padding region that can affect clustering
        data = torch.cat([aux[i, :c] for i, c in enumerate(caps_count)])
        # data = (aux[aux>0]).view(-1)

        # for determining the masks of validation split we use the proposals determined from training split
        if proposals is None:
            # determine clusters according to intervals length
            print("computing event-proposals by the KernelDensity algorithm ")
            kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(data.unsqueeze(1).numpy())
            s = linspace(0, self.max_interval, num=num_estimates)
            e = kde.score_samples(s.reshape(-1, 1))
            proposals = s[argrelextrema(e, np.less)[0]]
            self.logger.info(f"PROPOSALS: Number of event-proposals: {len(proposals)}")
            self.logger.info(f"PROPOSALS: Event-proposals: {proposals}")
            print(f"PROPOSALS: Number of event-proposals: {len(proposals)}")

            # filter proposals with less than min_count_per_proposal of events
            filter_proposals, filter_proposals_count = [], []
            import ipdb; ipdb.set_trace()

            def append_porposal(p, count):
                filter_proposals.append(p)
                filter_proposals_count.append(count)

            current_sum = (data < proposals[0]).sum()
            if current_sum >= min_count_per_proposal:
                append_porposal(proposals[0], current_sum)
                current_sum = 0

            for i, p in enumerate(proposals[1:], start=1):
                current_sum += ((data >= proposals[i - 1]) * (data < p)).sum()
                if current_sum >= min_count_per_proposal:
                    append_porposal(p, current_sum)
                    current_sum = 0

            current_sum += (data >= proposals[-1]).sum()
            if current_sum >= min_count_per_proposal:
                append_porposal(proposals[-1], current_sum)
                current_sum = 0

            if (data < proposals[0]).sum() >= min_count_per_proposal:
                filter_proposals.append(proposals[0])
                filter_proposals_count.append((data < proposals[0]).sum())

            proposals = filter_proposals
            self.logger.info(f"PROPOSALS: Number of event-proposals (filtered): {len(proposals)}")
            self.logger.info(f"PROPOSALS: Event-proposals (filtered): {proposals}")
            print(f"PROPOSALS: Number of event-proposals (filtered): {len(proposals)}")

        # padding legths using -1
        for i, c in enumerate(caps_count):
            aux[i, c:] = -1

        # determine cluster of each interval
        result = torch.full_like(aux, -1, dtype=torch.int)
        result[(aux >= 0) * (aux < proposals[0])] = 0
        for i in range(1, len(proposals)):
            result[(aux >= proposals[i - 1]) * (aux < proposals[i])] = i
        result[aux >= proposals[-1]] = len(proposals)

        clusters_sizes = [(result == i).sum().item() for i in range(len(proposals) + 1)]
        self.logger.info(f"PROPOSALS: Count of intervals per cluster: {clusters_sizes}")
        self.logger.info(f"PROPOSALS: Total intervals grouped: {sum(clusters_sizes)}")
        print("PROPOSALS: Count of intervals per cluster: ", clusters_sizes)
        print("PROPOSALS: Total intervals grouped: ", sum(clusters_sizes))

        # compute mask for complete intervals
        # mask = torch.zeros(intervals.size(0), max_num_chunks, len(proposals) + 1)
        # for i in range(intervals.size(0)):
        #     for j in range(caps_count[i]):
        #         mask[i, int(intervals[i, j, 0]) : int(intervals[i, j, 1]), result[i, j]] = 1

        # compute masks for the positions where an interval starts or ends only
        s_mask = torch.zeros(intervals.size(0), max_num_chunks, len(proposals) + 1)
        e_mask = torch.zeros(intervals.size(0), max_num_chunks, len(proposals) + 1)
        for i in range(intervals.size(0)):
            for j in range(caps_count[i]):
                s = int(intervals[i, j, 0])
                e = int(min(intervals[i, j, 1], max_num_chunks - 1))

                # set start and end proposals for intervals that start before the current interval too
                for k in range(j):
                    if intervals[i, k, 1] > s:
                        # interval that starts before and ends after the current interval starts
                        s_mask[i, s, result[i, k]] = 1
                    if intervals[i, k, 1] >= e:
                        # interval that starts before and ends after the current interval ends
                        e_mask[i, e, result[i, k]] = 1

                # set start and end proposals for current interval
                s_mask[i, s, result[i, j]] = 1
                e_mask[i, e, result[i, j]] = 1

                # set end proposal for intervals tat start after the current interval too
                for k in range(j + 1, caps_count[i]):
                    if intervals[i, k, 0] < e and intervals[i, k, 1] >= e:
                        # interval that starts after and ends after the current interval ends
                        e_mask[i, e, result[i, k]] = 1

        # determine the number of positive examples per cluster
        s_pos_samples = s_mask.sum(dim=1).sum(dim=0)  # (len(proposals) + 1, )
        e_pos_samples = e_mask.sum(dim=1).sum(dim=0)  # (len(proposals) + 1, )
        self.logger.info(f"PROPOSALS: Count of positive examples per cluster (start positions): {s_pos_samples}")
        self.logger.info(f"PROPOSALS: Count of positive examples per cluster (end positions): {e_pos_samples}")
        print("PROPOSALS: Count of positive examples per cluster (start positions): ", s_pos_samples)
        print("PROPOSALS: Count of positive examples per cluster (end positions): ", e_pos_samples)

        # determine the number of negative examples per cluster, descarding the frames where we will not classify
        s_neg_mask = 1 - s_mask
        print(s_neg_mask.size())
        s_frame_mask = (s_neg_mask.sum(dim=-1, keepdim=True) != (len(proposals) + 1)).repeat(1, 1, len(proposals) + 1)
        print(s_frame_mask.size())
        s_neg_samples = (s_neg_mask * s_frame_mask).sum(dim=1).sum(dim=0)  # (len(proposals) + 1, )

        e_neg_mask = 1 - e_mask
        e_frame_mask = (e_neg_mask.sum(dim=-1, keepdim=True) != (len(proposals) + 1)).repeat(1, 1, len(proposals) + 1)
        e_neg_samples = (e_neg_mask * e_frame_mask).sum(dim=1).sum(dim=0)  # (len(proposals) + 1, )

        self.logger.info(f"PROPOSALS: Count of negative examples per cluster (start positions): {s_neg_samples}")
        self.logger.info(f"PROPOSALS: Count of negative examples per cluster (end positions): {e_neg_samples}")
        print("PROPOSALS: Count of negative examples per cluster (start positions): ", s_neg_samples)
        print("PROPOSALS: Count of negative examples per cluster (end positions): ", e_neg_samples)

        s_prop_pos_weights = s_neg_samples / s_pos_samples
        e_prop_pos_weights = e_neg_samples / e_pos_samples

        return s_mask, e_mask, proposals, s_prop_pos_weights, e_prop_pos_weights

    def __init_dense_loader(self):
        print("Initializing data loaders...")

        # get train split data
        print(" initializing train split data loader...")
        (vidxs, cidxs, intervals, fps, progs, prog_lens, caps, pos, upos, cap_lens,) = extract_split_data_from_corpus(
            self.corpus, split=0
        )
        (cidxs_t, intervals_t, caps_count_t, progs_t, caps_t, _, _, _,) = data2tensors(
            cidxs, intervals, progs, prog_lens, caps, pos, upos, cap_lens
        )
        self.max_prog = progs_t.size(1)
        self.max_caps = caps_t.size(1)
        self.max_words = caps_t.size(2)
        self.max_interval = torch.max(intervals_t.view(-1, 2)[:, 1] - intervals_t.view(-1, 2)[:, 0])
        self.last_interval_end = torch.max(intervals_t.view(-1, 2)[:, 1])

        # determine the K most frequent words for semantic encodings from the train split
        # freq_words = self.__get_most_freq_words(
        #     caps,
        #     upos,
        #     postags=self.modules_config["sem_tagger_config"].upos_tags,
        #     words_to_discard=self.modules_config["sem_tagger_config"].words_to_discard,
        # )

        # determine the ground truth for semantic enconding
        # caps_sem_enc_t = self.__get_sem_enc(freq_words, caps, upos)

        # determine the ground truth for event masking
        (
            event_s_mask_t,
            event_e_mask_t,
            event_proposals,
            s_prop_pos_weights,
            e_prop_pos_weights,
        ) = self.__get_interval_mask(
            intervals_t,
            caps_count_t,
            max_num_chunks=self.trainer_config.max_num_chunks,
            num_estimates=self.modules_config["proposals_tagger_config"].num_estimates,
            min_count_per_proposal=self.modules_config["proposals_tagger_config"].min_count_per_proposal,
        )
        self.s_prop_pos_weights = s_prop_pos_weights.to(self.device)
        self.e_prop_pos_weights = e_prop_pos_weights.to(self.device)
        self.num_proposals = len(event_proposals) + 1

        train_loader = get_dense_loader(
            h5_file_path=self.trainer_config.train_h5_file_path,
            h5_file_group_name=self.trainer_config.h5_file_group_name,
            vidxs=vidxs,
            vidxs_blcklist=self.trainer_config.train_blacklist,
            vfps=fps,
            cidxs=cidxs_t,
            intervals=intervals_t,
            caps_count=caps_count_t,
            captions=caps_t,
            # caps_sem_enc=caps_sem_enc_t,
            # pos=pos_t,
            # upos=upos_t,
            # cap_lens=cap_lens_t,
            progs=progs_t,
            prog_lens=prog_lens,
            event_proposals_s=event_s_mask_t,
            event_proposals_e=event_e_mask_t,
            batch_size=self.trainer_config.batch_size,
            train=True,
            num_workers=trainer_config.loader_num_workers,
            pin_memory=trainer_config.loader_pin_memory,
        )

        # get valid split data
        print(" initializing valid split data loader...")
        (vidxs, cidxs, intervals, fps, progs, prog_lens, caps, pos, upos, cap_lens,) = extract_split_data_from_corpus(
            self.corpus, split=1
        )
        (cidxs_t, intervals_t, caps_count_t, progs_t, caps_t, _, _, _,) = data2tensors(
            cidxs,
            intervals,
            progs,
            prog_lens,
            caps,
            pos,
            upos,
            cap_lens,
            self.max_prog,
            self.max_caps,
            self.max_words,
        )
        # self.max_prog = max(self.max_prog, progs_t.size(1))
        # self.max_caps = max(self.max_caps, caps_t.size(1))
        # self.max_words = max(self.max_words, caps_t.size(2))
        # self.max_interval = max(self.max_interval, torch.max(intervals_t.view(-1, 2)[:,1]-intervals_t.view(-1, 2)[:,0]))
        # self.last_interval_end = max(self.last_interval_end, torch.max(intervals_t.view(-1, 2)[:,1]))

        # determine the ground truth for semantic enconding
        # caps_sem_enc_t = self.__get_sem_enc(freq_words, caps, upos)

        # determine the ground truth for event masking
        event_s_mask_t, event_e_mask_t, _, _, _ = self.__get_interval_mask(
            intervals_t, caps_count_t, max_num_chunks=self.trainer_config.max_num_chunks, proposals=event_proposals,
        )

        val_loader = get_dense_loader(
            h5_file_path=self.trainer_config.valid_h5_file_path,
            h5_file_group_name=self.trainer_config.h5_file_group_name,
            vidxs=vidxs,
            vidxs_blcklist=self.trainer_config.valid_blacklist,
            vfps=fps,
            cidxs=cidxs_t,
            intervals=intervals_t,
            caps_count=caps_count_t,
            captions=caps_t,
            # caps_sem_enc=caps_sem_enc_t,
            # pos=pos_t,
            # upos=upos_t,
            # cap_lens=cap_lens_t,
            progs=progs_t,
            prog_lens=prog_lens,
            event_proposals_s=event_s_mask_t,
            event_proposals_e=event_e_mask_t,
            batch_size=self.trainer_config.batch_size * 10,
            train=False,
            num_workers=trainer_config.loader_num_workers,
            pin_memory=trainer_config.loader_pin_memory,
        )

        self.logger.info(f"Max program len: {self.max_prog}")
        self.logger.info(f"Max caption len: {self.max_words}")
        self.logger.info(f"Max intervals count: {self.max_caps}")
        self.logger.info(f"Max interval len: {int(self.max_interval)}")
        self.logger.info(f"Last interval end: {int(self.last_interval_end)}")

        print(f" Max program len: {self.max_prog}")
        print(f" Max caption len: {self.max_words}")
        print(f" Max intervals count: {self.max_caps}")
        print(f" Max interval len: {int(self.max_interval)}")
        print(f" Last interval end: {int(self.last_interval_end)}")
        print(f" Number of event-proposals: {self.num_proposals}")

        self.loaders = {"train": train_loader, "val_1": val_loader}

    def __process_batch(
        self,
        video_feats,
        feats_count,
        gt_intervals,
        gt_caps_count,
        # gt_captions,
        # gt_caps_sem_enc,
        # gt_pos,
        # gt_upos,
        # gt_cap_lens,
        gt_program,
        gt_prog_len,
        gt_prop_s,
        gt_prop_e,
        epoch,
        tf_ratio=0.5,
        phase="train",
    ):
        bsz = video_feats[0].size(0)

        # Move all tensors to device
        video_feats = [f.to(self.device) for f in video_feats]
        feats_count = feats_count.to(self.device)
        # gt_intervals = gt_intervals.to(self.device)
        # gt_captions = gt_captions.to(self.device)
        # gt_pos = gt_pos.to(self.device)
        # gt_upos = gt_upos.to(self.device)
        # gt_cap_lens = gt_cap_lens.to(self.device)
        gt_program = gt_program.to(self.device)
        # gt_prog_len = gt_prog_len.to(self.device)
        # gt_caps_sem_enc = gt_caps_sem_enc.to(self.device)

        # determine position for truncating the programs
        if phase == "train":
            # determine the number instructions that are necessary for matching the i-th interval
            # temp_prog_pos = torch.zeros(gt_intervals.size(0), gt_intervals.size(1)).to(gt_intervals.device)
            # temp_prog_pos[:, 0] = gt_intervals[:, 0, 1] + 1
            # for i in range(1, gt_intervals.size(1)):
            #     temp_prog_pos[:, i] = (
            #         temp_prog_pos[:, i - 1]
            #         + (gt_intervals[:, i, 0] - gt_intervals[:, i - 1, 0])
            #         + (gt_intervals[:, i, 1] - gt_intervals[:, i, 0])
            #         + 1
            #     )

            # at least minimum program length steps, considering at least min_caps_truncation captions for each video
            # truncate_prog_at = int(
            #     max(torch.min(gt_prog_len), torch.max(temp_prog_pos[:, self.trainer_config.min_caps_truncation - 1]),)
            # )
            # self.logger.info(f"the gt programs of len {gt_prog_len} will be truncated at {truncate_prog_at}")

            # determine the number of captions/intervals that must be generated for each video, truncating at truncate_prog_at
            self.logger.info(f"gt caps count: {gt_caps_count}")
            # gt_caps_count = torch.sum((gt_intervals[:, :, 1] > 0) * (temp_prog_pos <= truncate_prog_at), dim=1).to(
            #     self.device
            # )
            # gt_caps_count = torch.sum((gt_intervals[:,:,1] > 0) * (temp_prog_pos < truncate_prog_at), dim=1).to(self.device)
            # self.logger.info(f"tuncated gt caps count: {gt_caps_count}")

            # truncate gt batch tensors before move them to the device
            # max_caps = torch.max(gt_caps_count)
            # gt_program = gt_program[:, :truncate_prog_at].to(self.device)
            # gt_captions = gt_captions[:, :max_caps].to(self.device)
            # gt_caps_sem_enc = gt_caps_sem_enc[:, :max_caps].to(self.device)
            # gt_pos = gt_pos[:, :max_caps].to(self.device)
            # gt_intervals = gt_intervals[:, :max_caps].to(self.device)
            # gt_upos = gt_upos[:, :max_caps].to(self.device)
            # gt_proposals = gt_proposals[:, :int(torch.max(gt_intervals[:,gt_caps_count-1,0]))].to(self.device)

            # self.avg_truncation += truncate_prog_at
            self.avg_caps += int(torch.mean(gt_caps_count.float()))
            # self.avg_feats += int(torch.mean(gt_intervals[torch.arange(bsz), gt_caps_count - 1, 1]))
        elif "val" in phase:
            # truncate gt batch tensors according to average parameters, and move them to the device
            # truncate_prog_at = self.avg_truncation
            # gt_program = gt_program[:, :truncate_prog_at].to(self.device)
            # gt_captions = gt_captions[:, : self.avg_caps].to(self.device)
            # gt_caps_sem_enc = gt_caps_sem_enc[:, : self.avg_caps].to(self.device)
            # gt_pos = gt_pos[:, : self.avg_caps].to(self.device)
            # gt_intervals = gt_intervals[:, : self.avg_caps].to(self.device)
            # gt_upos = gt_upos[:, :self.avg_caps].to(self.device)
            # gt_proposals = gt_proposals[:, :int(torch.max(gt_intervals[:,self.avg_caps,0]))].to(self.device)

            # move the gt batch tensors to device for computing the generalization loss
            # gt_program = gt_program.to(self.device)
            # gt_captions = gt_captions.to(self.device)
            # gt_caps_sem_enc = gt_caps_sem_enc.to(self.device)
            # gt_pos = gt_pos.to(self.device)
            # gt_intervals = gt_intervals.to(self.device)
            # # gt_upos = gt_upos.to(self.device)
            # gt_proposals = gt_proposals.to(self.device)
            pass
        else:
            pass

        # filter proposals
        max_caps = torch.max(gt_caps_count)
        last_chunk = gt_prop_e.size(1) - 1
        gt_prop_s = torch.cat(
            [gt_prop_s[torch.arange(bsz), gt_intervals[:, i, 0].long()].unsqueeze(1) for i in range(max_caps)], dim=1
        ).to(self.device)
        gt_prop_e = torch.cat(
            [
                gt_prop_e[torch.arange(bsz), gt_intervals[:, i, 1].clamp(max=last_chunk).long()].unsqueeze(1)
                for i in range(max_caps)
            ],
            dim=1,
        ).to(self.device)

        # gt_cap_lens = gt_cap_lens.to(self.device)
        # gt_prog_len = gt_prog_len.to(self.device)
        # gt_proposals = gt_proposals.to(self.device)
        gt_caps_count = gt_caps_count.to(self.device)

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(phase == "train"):
            self.logger.info("model computation....")
            (prog_logits, _, _, _, _, _, _, caps_count, s_prop_logits, e_prop_logits, _,) = self.dense_captioner(
                v_feats=video_feats,
                feats_count=feats_count,
                prog_len=int(gt_prog_len.max()),  # truncate_prog_at,
                teacher_forcing_p=tf_ratio,
                gt_program=gt_program,
                gt_captions=None,  # gt_captions,
                gt_caps_count=gt_caps_count,
                gt_sem_enc=None,  # gt_caps_sem_enc,
                gt_pos=None,  # gt_pos,
                gt_intervals=gt_intervals,
                gt_prop_s=gt_prop_s,
                gt_prop_e=gt_prop_e,
                max_prog=self.avg_truncation,  # max_prog=self.max_prog,
                max_caps=self.avg_caps,  # max_caps=self.max_caps,
                max_cap=self.max_words,
                max_chunks=self.avg_truncation,
                max_back_steps=self.modules_config["proposals_tagger_config"].max_back_steps,
            )  # the maximum value of start pointers is lower than the max_prog to be generated
            self.logger.info(f"proposals count: {caps_count}")

            # if "val" in phase:
            #     gt_caps_count = gt_caps_count.to(self.device)
            #     gt_caps_count = torch.min(gt_caps_count, caps_count)

            # Evaluate the loss function
            self.logger.info("loss computation....")
            (loss, prog_loss, _, _, _, s_prop_loss, e_prop_loss, _,) = self.criterion(
                gt_captions=None,  # gt_captions,
                gt_cap_lens=None,  # gt_cap_lens,
                pred_captions=None,
                gt_caps_sem_enc=None,  # gt_caps_sem_enc,
                pred_caps_sem_enc=None,
                gt_pos_seq=None,  # gt_pos,
                pred_pos_seq=None,
                gt_program=None,  # gt_program,
                gt_prog_len=None,  # gt_prog_len,
                pred_program=prog_logits,
                gt_intervals=gt_intervals,
                pred_intervals=None,  # intervals,
                gt_prop_s=gt_prop_s,
                gt_prop_e=gt_prop_e,
                pred_prop_s=s_prop_logits,
                pred_prop_e=e_prop_logits,
                gt_caps_count=gt_caps_count,
                pred_caps_count=None,
                gt_prop_count=None,  # proposals_count,
                epoch=epoch,
                truncate_prog_at=None,  # truncate_prog_at,
            )

        if phase == "train":
            # compute backward pass for somputing the gradients
            self.logger.info("loss backward....")
            loss.backward()

            # clip gradients to prevent NaNs in the prog-loss
            # nn.utils.clip_grad_norm_(self.dense_captioner.rnn_cell.parameters(), 0.5)

            # update the parameters
            self.logger.info("optimizer step...")
            self.optimizer.step()

        return (
            loss,
            prog_loss,
            None,  # cap_loss,
            None,  # sem_enc_loss,
            None,  # pos_loss,
            s_prop_loss,
            e_prop_loss,
            None,  # iou_reward,
            None,  # program,
            None,  # captions,
            None,  # intervals,
            s_prop_logits,
            e_prop_logits,
            gt_prop_s,
            gt_prop_e,
            caps_count,
            None,  # truncate_prog_at,
        )

    def __save_checkpoint(self, epoch, save_checkpoints_dir, new_best=False):
        if new_best:
            chkpt_filename = f"best_chkpt_{epoch}.pt"
        else:
            chkpt_filename = f"chkpt_{epoch}.pt"

        if self.last_saved_epoch == epoch:
            # the checkpoint was saved after training at this epoch, we don't need to save it again, we copy it only
            copyfile(
                os.path.join(save_checkpoints_dir, f"chkpt_{epoch}.pt"),
                os.path.join(save_checkpoints_dir, chkpt_filename),
            )
            print(" the checkpoint was copied only")
        else:
            # save the new checkpoint (best or not)
            torch.save(
                obj={
                    "epoch": epoch,
                    "trainer_config": self.trainer_config,
                    "modules_config": self.modules_config,
                    "dense_captioner": self.dense_captioner.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_metrics": self.best_metrics,
                    "avg_tuncation": self.avg_truncation,
                    "avg_caps": self.avg_caps,
                },
                f=os.path.join(save_checkpoints_dir, chkpt_filename),
            )
            print(" saved")

        # remove previously saved
        if new_best and self.last_best_saved_epoch != -1:
            os.remove(os.path.join(save_checkpoints_dir, f"best_chkpt_{self.last_best_saved_epoch}.pt"))
        elif not new_best and self.last_saved_epoch != -1:
            os.remove(os.path.join(save_checkpoints_dir, f"chkpt_{self.last_saved_epoch}.pt"))

        # update the last saved epoch
        if new_best:
            self.last_best_saved_epoch = epoch
        else:
            self.last_saved_epoch = epoch

    def __process_results(self, metrics_results, prediction, phase, epoch, save_checkpoints_dir, component):
        self.logger.info(f"{phase} set metrics for {component}: {metrics_results}")
        min_metrics = []
        for name, result in metrics_results.items():
            self.writer.add_scalar(f"proposals/{phase}-{component}-{name}", result, epoch)
            if name in self.best_metrics[component][phase] and (
                (name in min_metrics and result < self.best_metrics[component][phase][name][1])
                or (name not in min_metrics and result > self.best_metrics[component][phase][name][1])
            ):
                self.best_metrics[component][phase][name] = (epoch, result)
                if name in ["Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "All_Metrics"]:
                    self.early_stop = 0
                    torch.save(
                        prediction, os.path.join(save_checkpoints_dir, f"chkpt_{epoch}_{component}_output.json",),
                    )
                    # with open(os.path.join(save_checkpoints_dir, f'chkpt_{epoch}_{component}_output.json'), 'w') as f:
                    #    json.dump(prediction, f)
                if component == "densecap" and name == "METEOR" and phase == "val_1":
                    print("saving best checkpoint...")
                    self.__save_checkpoint(epoch, save_checkpoints_dir, True)
                if component in ["s_prop", "e_prop"] and name in [
                    "Recall/weighted",
                    "F1/weighted",
                    "ROC-AUC/weighted",
                ]:
                    print(f"saving best checkpoint due to improvement on {component}-{name}...")
                    self.__save_checkpoint(epoch, save_checkpoints_dir, True)

    def train_model(self, resume=False, checkpoint_path=None, min_num_epochs=50, early_stop_limit=10):
        # parallel_pool = Pool()
        # self.logger.info('Training captioning model on [{}] dataset with [{}] encoder and [{}] decoder'
        #                  .format(self.config.dataset_name, self.encoder_name, self.decoder_name))

        save_checkpoints_dir = os.path.join(self.out_folder, "models", self.trainer_config.str, self.datetime_str)
        if not os.path.exists(save_checkpoints_dir):
            os.makedirs(save_checkpoints_dir)

        if resume and os.path.exists(checkpoint_path):
            self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            print(f"Resuming from checkpoint: {checkpoint_path}.... ")

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            begin_epoch = checkpoint["epoch"]
            self.best_metrics = checkpoint["best_metrics"]
            self.avg_caps = checkpoint["avg_caps"]

            # TEMPORAL
            self.best_metrics["s_prop"] = {"val_1": {"Recall": (0, 0), "Precision": (0, 0),}}
            self.best_metrics["e_prop"] = {"val_1": {"Recall": (0, 0), "Precision": (0, 0),}}

            log_msg = f" (epoch {begin_epoch})"
            for phase in ["val_1"]:
                log_msg += f"\n  Captioning metrics {phase}: \n   "
                log_msg += "\t".join(
                    [f"{k}:({e:03d}, {v:.3f})" for k, (e, v) in self.best_metrics["captioning"][phase].items()]
                )
                log_msg += f"\n  DenseCaptioning metrics {phase}: \n   "
                log_msg += "\t".join(
                    [f"{k}:({e:03d}, {v:.3f})" for k, (e, v) in self.best_metrics["densecap"][phase].items()]
                )
                log_msg += f"\n  S-Proposals metrics {phase}: \n   "
                log_msg += "\t".join(
                    [f"{k}:({e:03d}, {v:.3f})" for k, (e, v) in self.best_metrics["s_prop"][phase].items()]
                )
                log_msg += f"\n  E-Proposals metrics {phase}: \n   "
                log_msg += "\t".join(
                    [f"{k}:({e:03d}, {v:.3f})" for k, (e, v) in self.best_metrics["e_prop"][phase].items()]
                )
            print(log_msg, "\n")
            self.logger.info(log_msg)

            model_dict = self.dense_captioner.state_dict()

            # 1. filter out unnecessary parameters (not included in the current architecture)
            pretrained_dict = {k: v for k, v in checkpoint["dense_captioner"].items() if k in model_dict}

            # 2. include in the dictionary to be loaded the new parameters in the current architecture
            for k, v in model_dict.items():
                if k not in pretrained_dict:  # or k[:2]=='fc':
                    pretrained_dict[k] = v

            # 3. load the new state dict
            self.dense_captioner.load_state_dict(pretrained_dict)

            # 4. freeze the part of the model that was trained before
            if self.trainer_config.resume_config.unfreeze_at > 0:
                self.dense_captioner.freeze(resume_config=self.trainer_config.resume_config)
                if self.trainer_config.resume_config.begin_epoch != -1:
                    begin_epoch = self.trainer_config.resume_config.begin_epoch
        else:
            begin_epoch = 0
            self.best_metrics = {"programmer": {}, "captioning": {}, "densecap": {}, "s_prop": {}, "e_prop": {}}
            for p in ["val_1"]:
                # self.best_metrics['programmer'][p] = {'Bleu_1': (0, 0), 'Bleu_2': (0, 0), 'Bleu_3': (0, 0), 'Bleu_4': (0, 0),
                #                              'METEOR': (0, 0), 'ROUGE_L': (0, 0), 'CIDEr': (0, 0), 'SPICE': (0, 0), 'All_Metrics': (0, 0)}
                self.best_metrics["captioning"][p] = {
                    "Bleu_1": (0, 0),
                    "Bleu_2": (0, 0),
                    "Bleu_3": (0, 0),
                    "Bleu_4": (0, 0),
                    "METEOR": (0, 0),
                    "ROUGE_L": (0, 0),
                    "CIDEr": (0, 0),
                    "SPICE": (0, 0),
                    "All_Metrics": (0, 0),
                }
                self.best_metrics["densecap"][p] = {
                    "Bleu_1": (0, 0),
                    "Bleu_2": (0, 0),
                    "Bleu_3": (0, 0),
                    "Bleu_4": (0, 0),
                    "METEOR": (0, 0),
                    "ROUGE_L": (0, 0),
                    "CIDEr": (0, 0),
                    "SPICE": (0, 0),
                    "Recall": (0, float("inf")),
                    "Precision": (0, 0),
                    "All_Metrics": (0, 0),
                }
                self.best_metrics["s_prop"][p] = {
                    "Recall": (0, 0),
                    "Precision": (0, 0),
                }
                self.best_metrics["e_prop"][p] = {
                    "Recall": (0, 0),
                    "Precision": (0, 0),
                }

        self.dense_captioner.to(self.device)
        print("\nParameters of Dense Captioner model:\n")
        total_size = 0
        for n, p in self.dense_captioner.named_parameters():
            # print(n, p.size(), p.device)
            total_size += torch.numel(p)
        print(" total size: ", (total_size * 8) / (1024 ** 3), "\n")

        # Start training process
        self.early_stop, self.last_saved_epoch, self.last_best_saved_epoch = 0, -1, -1
        time_phases = {"train": 0, "val_1": 0}
        prog_metrics_results, cap_metrics_results, densecap_metrics_results = None, None, None
        s_prop_metrics_results = None
        e_prop_metrics_results = None
        for epoch in range(begin_epoch, 1000):
            # unfreeze the freezed part of the model if needed
            if epoch == self.trainer_config.resume_config.unfreeze_at:
                self.dense_captioner.unfreeze()

            tf_ratio = get_tf_ratio(self.trainer_config.tf_config, epoch)
            self.writer.add_scalar("proposals/teacher_forcing_ratio", tf_ratio, epoch)

            loss_phases = {"train": 0, "val_1": 0}
            s_prop_loss_phases = {"train": 0, "val_1": 0}
            e_prop_loss_phases = {"train": 0, "val_1": 0}
            for phase in ["train", "val_1"]:
                # prepare gradients of the model according to the phase to be performed
                if phase == "train":
                    self.dense_captioner.train()
                    self.avg_truncation = 0
                    self.avg_caps = 0
                    self.avg_feats = 0
                else:
                    self.dense_captioner.eval()
                    self.avg_truncation = 0
                    self.avg_caps = 0
                    self.avg_feats = 0
                    # self.avg_truncation //= len(self.loaders["train"])
                    # self.avg_caps //= len(self.loaders["train"])
                    # self.avg_feats //= len(self.loaders["train"])
                    self.writer.add_scalar(
                        "proposals/{}-epochs-avg_truncation".format(phase), self.avg_truncation, epoch,
                    )
                    self.writer.add_scalar("proposals/{}-epochs-avg_caps".format(phase), self.avg_caps, epoch)
                    self.writer.add_scalar(
                        "proposals/{}-epochs-avg_feats".format(phase), self.avg_feats, epoch,
                    )

                # predicted_sentences = {}
                time_start_epoch, total_time_iters = time.perf_counter(), 0
                loss_count, s_prop_loss_count, e_prop_loss_count = 0, 0, 0
                all_programs = []
                all_captions = []
                all_prog_ids = []
                all_caps_ids = []
                all_intervals = []
                all_tstamps = []
                all_props_s, all_gt_props_s = [], []
                all_props_e, all_gt_props_e = [], []
                all_cap_counts = []
                for (
                    i,
                    (
                        vidx,
                        cidxs,
                        cnn,
                        c3d,
                        feats_count,
                        tstamps,
                        fps,
                        gt_intervals,
                        gt_caps_count,
                        # gt_caps,
                        # gt_caps_sem_enc,
                        # gt_pos,
                        # gt_upos,
                        # gt_cap_lens,
                        gt_prog,
                        gt_prog_len,
                        gt_prop_s,
                        gt_prop_e,
                    ),
                ) in enumerate(self.loaders[phase], start=1):
                    time_start_iter = time.perf_counter()

                    video_feats = [cnn, c3d]
                    iteration = epoch * len(self.loaders[phase]) + i

                    (
                        loss,
                        _,
                        _,
                        _,
                        _,
                        s_prop_loss,
                        e_prop_loss,
                        _,
                        program,
                        _,
                        _,
                        s_prop_logits,
                        e_prop_logits,
                        gt_prop_s,
                        gt_prop_e,
                        caps_count,
                        truncated_pos,
                    ) = self.__process_batch(
                        video_feats,
                        feats_count,
                        gt_intervals,
                        gt_caps_count,
                        # gt_caps,
                        # gt_caps_sem_enc,
                        # gt_pos,
                        # gt_upos,
                        # gt_cap_lens,
                        gt_prog,
                        gt_prog_len,
                        gt_prop_s,
                        gt_prop_e,
                        epoch,
                        tf_ratio,
                        phase,
                    )
                    loss_count += loss.item()
                    s_prop_loss_count += s_prop_loss.item()
                    e_prop_loss_count += e_prop_loss.item()

                    rl_strategy = self.trainer_config.criterion_config.rl_strategy
                    self.writer.add_scalar(f"proposals/{phase}-iters-{rl_strategy}-loss", loss, iteration)
                    # self.writer.add_scalar(f"proposals/{phase}-iters-{rl_strategy}-prog_loss", prog_loss, iteration)
                    self.writer.add_scalar(f"proposals/{phase}-iters-s_proposals_loss", s_prop_loss, iteration)
                    self.writer.add_scalar(f"proposals/{phase}-iters-e_proposals_loss", e_prop_loss, iteration)
                    # self.writer.add_scalar(f"proposals/{phase}-iters-iou_reward", iou_reward, iteration)

                    # logging message
                    total_time_iters += time.perf_counter() - time_start_iter
                    lrs = self.lr_scheduler.get_last_lr()
                    gpu_temp = get_gpu_temps(self.device)
                    log_msg = (
                        "\rEpoch:{0:03d} Phase:{1:6s} Iter:{2:04d}/{3:04d} avg-Time:{4:.1f}s lr:{5:.6f} gpu-temp:{6:02d} Loss:{7:9.4f} "
                        "\t[s-proposals-loss:{8:9.4f} e-proposals-loss:{9:9.4f}]"
                    ).format(
                        epoch,
                        phase,
                        i,
                        len(self.loaders[phase]),
                        total_time_iters / i,
                        lrs[0],
                        gpu_temp,
                        # rl_strategy,
                        loss.item(),
                        # prog_loss.item(),
                        s_prop_loss.item(),
                        e_prop_loss.item(),
                        # iou_reward.item(),
                    )
                    self.logger.info(log_msg)
                    sys.stdout.write(log_msg)

                    # if iteration % self.trainer_config.step_to_print == 0:
                    #     if phase == "train":
                    #         pred = decode_from_tokens(
                    #             self.programs_vocab, program[0], until_eos=False, max_length=truncated_pos,
                    #         )
                    #         gt = decode_from_tokens(
                    #             self.programs_vocab, gt_prog[0], until_eos=False, max_length=truncated_pos,
                    #         )
                    #     else:
                    #         pred = decode_from_tokens(
                    #             self.programs_vocab, program[0], until_eos=False, max_length=self.max_prog,
                    #         )
                    #         gt = decode_from_tokens(
                    #             self.programs_vocab, gt_prog[0], until_eos=False, max_length=self.max_prog,
                    #         )
                    #     self.logger.info(f"PRED PROG: {pred}")
                    #     self.logger.info(f"PRED INTERV: {intervals[0, :gt_caps_count[0]]}")
                    #     self.logger.info(f"GT PROG: {gt}")
                    #     self.logger.info(f"GT INTERV: {gt_intervals[0, :gt_caps_count[0]]}")
                    #     # sample_diff = torch.sum(program[0] != gt_prog[0])
                    #     sample_diff = sum([s1 != s2 for s1, s2 in zip(pred.split(" "), gt.split(" "))])
                    #     self.logger.info(f"sample-pred-prog-diff: {sample_diff}")

                    if phase != "train":
                        # save proposals for computing evaluation metrics
                        all_props_s.append(s_prop_logits.to("cpu"))
                        all_gt_props_s.append(gt_prop_s.to("cpu"))

                        all_props_e.append(e_prop_logits.to("cpu"))
                        all_gt_props_e.append(gt_prop_e.to("cpu"))

                        all_cap_counts.append(caps_count.to("cpu"))

                    #     # save programs and the videos' idx for computing evaluation metrics
                    #     all_programs.append(program.to("cpu"))
                    #     all_prog_ids.append(vidx)

                    #     # save captions and the captions' idx for computing evaluation metrics (only the first caps_count captions are evaluated)
                    #     all_captions.append((gt_caps.to("cpu"), caps_count.to("cpu"), gt_caps_count.to("cpu"),))
                    #     all_caps_ids.append(cidxs)

                    #     # save intervals for computing evaluation metrics
                    #     all_intervals.append(intervals.to("cpu"))
                    #     all_tstamps.append(tstamps / (fps.unsqueeze(1) ** 2))

                    #     # for predicted_tokens, vid in zip(outputs, video_ids):
                    #     #     predicted_sentences[vid] = [self.__decode_from_tokens(predicted_tokens)]

                    #     # logging sample sentences of prediction and target
                    #     # self.logger.info('[vid:{}]'.format(video_ids[0]))
                    #     # self.logger.info('\nWE: {}\nGT: {}'.format(predicted_sentences[video_ids[0]],
                    #     #                                            self.__decode_from_tokens(captions[0].squeeze())))

                # (Sanity) replace the last checkpoint when epoch is power of 2
                if phase == "train" and (self.last_saved_epoch == -1 or not (epoch & (epoch - 1))):
                    print("saving checkpoint...")
                    self.__save_checkpoint(epoch, save_checkpoints_dir, False)

                avg_loss = loss_count / len(self.loaders[phase])
                loss_phases[phase] = avg_loss
                self.writer.add_scalar(f"proposals/{phase}-epochs-avg-loss", avg_loss, epoch)

                s_prop_avg_loss = s_prop_loss_count / len(self.loaders[phase])
                s_prop_loss_phases[phase] = s_prop_avg_loss
                self.writer.add_scalar(f"proposals/{phase}-epochs-s_proposals-avg-loss", s_prop_avg_loss, epoch)

                e_prop_avg_loss = e_prop_loss_count / len(self.loaders[phase])
                e_prop_loss_phases[phase] = e_prop_avg_loss
                self.writer.add_scalar(f"proposals/{phase}-epochs-e_proposals-avg-loss", e_prop_avg_loss, epoch)

                if phase != "train":
                    self.early_stop += 1
                    print("evaluating proposals...")
                    s_prop_metrics_results = multilabel_evaluate_from_logits(
                        all_gt_props_s, all_props_s, all_cap_counts
                    )
                    e_prop_metrics_results = multilabel_evaluate_from_logits(
                        all_gt_props_e, all_props_e, all_cap_counts
                    )
                    self.__process_results(
                        s_prop_metrics_results, None, phase, epoch, save_checkpoints_dir, "s_prop",
                    )
                    self.__process_results(
                        e_prop_metrics_results, None, phase, epoch, save_checkpoints_dir, "e_prop",
                    )
                    #     # predicted_sentences = pool.apply_async(self.__get_sentences, [all_outputs, all_video_ids])

                    #     # if cap_metrics_results is not None:
                    #     # get async results
                    #     # cap_metrics_results, pred_caps = cap_metrics_results.get()
                    #     # prog_metrics_results, pred_progs = prog_metrics_results.get()
                    #     # densecap_metrics_results, pred_intervals = densecap_metrics_results.get()

                    #     # print('evaluating progs...')
                    #     # prog_metrics_results, pred_progs = evaluate_from_tokens(self.programs_vocab, all_programs, all_prog_ids, self.ref_programs[phase], False)
                    #     print("evaluating captions (basic)...")
                    #     cap_metrics_results, pred_caps = evaluate_from_tokens(
                    #         self.caps_vocab, all_captions, all_caps_ids, self.ref_captions[phase],
                    #     )
                    #     print("evaluating captions (dense)...")
                    #     (densecap_metrics_results, pred_intervals,) = densecap_evaluate_from_tokens(
                    #         self.caps_vocab,
                    #         all_prog_ids,
                    #         all_tstamps,
                    #         all_intervals,
                    #         all_captions,
                    #         self.ref_densecaps[phase],
                    #     )

                    #     # process results, saving the checkpoint if any improvement occurs
                    #     # self.__process_results(prog_metrics_results, pred_progs, phase, epoch-1, save_checkpoints_dir, 'programmer')
                    #     self.__process_results(
                    #         cap_metrics_results, pred_caps, phase, epoch, save_checkpoints_dir, "captioning",
                    #     )
                    #     self.__process_results(
                    #         densecap_metrics_results, pred_intervals, phase, epoch, save_checkpoints_dir, "densecap",
                    #     )

                    # report results if any improvement occurs
                    if self.early_stop == 0:
                        log_msg = f"\n IMPROVEMENT ON {phase} at epoch {epoch} !"

                        #         # log_msg += '\n Programmer metrics: \n   '
                        #         # log_msg += '\t'.join([f'{k}:({e:03d}, {v:.3f})' for k, (e, v) in self.best_metrics['programmer'][phase].items()])

                        #         log_msg += "\n  Captioning metrics: \n   "
                        #         log_msg += "\t".join(
                        #             [f"{k}:({e:03d}, {v:.3f})" for k, (e, v) in self.best_metrics["captioning"][phase].items()]
                        #         )

                        #         log_msg += "\n  DenseCaptioning metrics: \n   "
                        #         log_msg += "\t".join(
                        #             [f"{k}:({e:03d}, {v:.3f})" for k, (e, v) in self.best_metrics["densecap"][phase].items()]
                        #         )

                        log_msg += f"\n  S-Proposals metrics {phase}: \n   "
                        log_msg += "\t".join(
                            [f"{k}:({e:03d}, {v:.3f})" for k, (e, v) in self.best_metrics["s_prop"][phase].items()]
                        )
                        log_msg += f"\n  E-Proposals metrics {phase}: \n   "
                        log_msg += "\t".join(
                            [f"{k}:({e:03d}, {v:.3f})" for k, (e, v) in self.best_metrics["e_prop"][phase].items()]
                        )

                        print(log_msg, "\n")
                        self.logger.info(log_msg)

                #     # prog_metrics_results = parallel_pool.apply_async(evaluate_from_tokens, [self.programs_vocab, all_programs, all_prog_ids, self.ref_programs[phase], False])
                #     # cap_metrics_results = parallel_pool.apply_async(evaluate_from_tokens, [self.caps_vocab, all_captions, all_caps_ids, self.ref_captions[phase]])
                #     # densecap_metrics_results = parallel_pool.apply_async(densecap_evaluate_from_tokens, [self.caps_vocab, all_intervals, all_captions, all_caps_ids, self.ref_densecaps[phase]])

                time_phases[phase] += time.perf_counter() - time_start_epoch

            # logging message
            log_msg = "\n"
            for k, v in loss_phases.items():
                log_msg += " {0}-avg-loss:{1:9.4f}".format(k, v)
            for k, v in time_phases.items():
                log_msg += " {0}-avg-time:{1:9.3f}h".format(k, (v / 3600) / (epoch + 1))
            log_msg += " tf_ratio:{0:.3f} lr:{1:.6f}".format(tf_ratio, lrs[0])
            self.logger.info(log_msg)
            sys.stdout.write(log_msg + "\n")

            # check if the training must be early sopped
            if epoch >= min_num_epochs and self.early_stop >= early_stop_limit * 2:
                # get async results
                # cap_metrics_results, pred_caps = cap_metrics_results.get()
                # prog_metrics_results, pred_progs = prog_metrics_results.get()
                # densecap_metrics_results, pred_intervals = densecap_metrics_results.get()

                # self.__process_results(prog_metrics_results, pred_caps, phase, epoch-1, save_checkpoints_dir, 'programmer')
                # self.__process_results(
                #     cap_metrics_results, pred_caps, phase, epoch - 1, save_checkpoints_dir, "captioning",
                # )
                # self.__process_results(
                #     densecap_metrics_results, pred_intervals, phase, epoch - 1, save_checkpoints_dir, "densecap",
                # )

                msg = "----early stopped at epoch {} after {} without any improvement-----".format(
                    epoch, early_stop_limit
                )
                self.logger.debug(msg)
                print(msg)
                break

            self.writer.add_scalar("proposals/learning-rate", self.optimizer.param_groups[0]["lr"], epoch)
            self.lr_scheduler.step()

        # close h5 files
        # self.h5_train.close()
        # self.h5_val.close()
        for loader in self.loaders.values():
            loader.dataset.close_h5_file()

        # log best results
        self.logger.info("Best results: {}".format(str(self.best_metrics)))

        return self.best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tha model for dense video captioning")
    parser.add_argument(
        "-comp",
        "--component",
        type=str,
        default="",
        required=True,
        help="Set the component of the model that you want to train model.",
    )
    parser.add_argument(
        "-chkpt",
        "--checkpoint_path",
        type=str,
        default="",
        help="Set the path to pre-trained model (by default is empty).",
    )
    parser.add_argument(
        "-data",
        "--dataset_folder",
        type=str,
        default="data/MSVD",
        help="Set the path to dataset folder (by default is data/MSVD).",
    )
    parser.add_argument(
        "-out",
        "--output_folder",
        type=str,
        default="results/MSVD",
        help="Set the path to output folder (by default is results/MSVD).",
    )

    args = parser.parse_args()

    # load hiper-parameters
    print("Loading configuration file...")
    config_path = os.path.join(args.dataset_folder, "prop_train_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    trainer_config = ConfigDict(config["trainer_config"])
    trainer_config.str = get_trainer_str(trainer_config)
    print(trainer_config.str)

    dense_captioner_config = ConfigDict(config["dense_captioner_config"])
    dense_captioner_config.str = get_dense_captioner_str(dense_captioner_config)
    print(dense_captioner_config.str)

    sem_tagger_config = ConfigDict(config["sem_tagger_config"])
    sem_tagger_config.str = get_sem_tagger_str(sem_tagger_config)
    print(sem_tagger_config.str)

    syn_embedd_config = ConfigDict(config["syn_embedd_config"])
    syn_embedd_config.str = get_syn_embedd_str(syn_embedd_config)
    print(syn_embedd_config.str)

    syn_tagger_config = ConfigDict(config["syn_tagger_config"])
    syn_tagger_config.str = get_syn_tagger_str(syn_tagger_config)
    print(syn_tagger_config.str)

    avscn_dec_config = ConfigDict(config["avscn_decoder_config"])
    avscn_dec_config.str = get_avscn_decoder_str(avscn_dec_config)
    print(avscn_dec_config.str)

    semsynan_dec_config = ConfigDict(config["semsynan_decoder_config"])
    semsynan_dec_config.str = get_semsynan_decoder_str(semsynan_dec_config)
    print(semsynan_dec_config.str)

    mm_config = ConfigDict(config["multimodal_config"])
    mm_config.str = get_mm_str(mm_config)
    print(mm_config.str)

    proposals_tagger_config = ConfigDict(config["proposals_tagger_config"])
    proposals_tagger_config.str = get_proposals_tagger_str(proposals_tagger_config)
    print(proposals_tagger_config)

    vncl_cell_config = ConfigDict(config["vncl_cell_config"])
    vncl_cell_config.str = get_vncl_cell_str(vncl_cell_config)
    print(vncl_cell_config.str, "\n")

    print("Initializing the experiment.......")
    modules_config = {
        "dense_captioner_config": dense_captioner_config,
        "sem_tagger_config": sem_tagger_config,
        "syn_embedd_config": syn_embedd_config,
        "syn_tagger_config": syn_tagger_config,
        "avscn_dec_config": avscn_dec_config,
        "semsynan_dec_config": semsynan_dec_config,
        "mm_config": mm_config,
        "vncl_cell_config": vncl_cell_config,
        "proposals_tagger_config": proposals_tagger_config,
    }
    # modules_config = [sem_tagger_config, syn_embedd_config, avscn_dec_config, semsynan_dec_config, vncl_cell_config]
    trainer = DenseVideo2TextTrainer(trainer_config, modules_config, args.dataset_folder, args.output_folder)

    print("Training.........")
    # try:
    best_results = trainer.train_model(
        resume=args.checkpoint_path != "",
        checkpoint_path=args.checkpoint_path,
        early_stop_limit=trainer_config.early_stop_limit,
    )
    print("Best results in the test set: {}".format(str(best_results)))
    # except Exception as e:
    #     print(f'An error occurred during training/validation process: {e}')
    #     trainer.h5_train.close()
    #     trainer.h5_val.close()

    print("--- END ---")
