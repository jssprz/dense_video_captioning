import os
import sys
import argparse
import pickle
import json
import datetime
import logging
import time
import random
import itertools
import heapq
from shutil import copyfile
from multiprocessing import Pool

import numpy as np
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import nltk

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

from utils import (
    get_freer_gpu,
    get_gpu_temps,
    decode_from_tokens,
    load_texts,
    evaluate_from_tokens,
    densecap_evaluate_from_tokens,
    multilabel_evaluate_from_logits,
    multiclass_evaluate_from_logits,
    get_tf_ratio,
    get_trainer_str,
    get_dense_captioner_str,
    get_visual_enc_str,
    get_sem_tagger_str,
    get_syn_tagger_str,
    get_ensemble_decoder_str,
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
            log_dir=os.path.join(
                self.out_folder,
                "log/runs/",
                f"{self.datetime_str} {trainer_config.str}",
            )
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
        super(DenseVideo2TextTrainer, self).__init__(
            trainer_config, modules_config, out_folder
        )

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
            self.modules_config["visual_enc_config"],
            self.modules_config["sem_tagger_config"],
            self.modules_config["syn_tagger_config"],
            self.modules_config["ensemble_dec_config"],
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
            params = []
            lr_lambda = []
            if self.dense_captioner.training_proposals:
                params.append(
                    {
                        "params": self.dense_captioner.proposals_enc.parameters(),
                        "lr": opt_conf.proposals_lr,
                    }
                )
                lr_lambda.append(
                    lambda epoch: opt_conf.lr_decay_factor
                    ** (epoch // opt_conf.proposals_lr_decay_epochs)
                )
            if self.dense_captioner.training_programmer:
                params.append(
                    {
                        "params": self.dense_captioner.fc.parameters(),
                        "lr": opt_conf.programmer_lr,
                    }
                )
                lr_lambda.append(
                    lambda epoch: opt_conf.lr_decay_factor
                    ** (epoch // opt_conf.programmer_lr_decay_epochs)
                )
            if self.dense_captioner.training_captioning:
                params.append(
                    {
                        "params": self.dense_captioner.clip_captioner.decoder.parameters(),
                        "lr": opt_conf.captioning_lr,
                    }
                )
                lr_lambda.append(
                    lambda epoch: opt_conf.lr_decay_factor
                    ** (epoch // opt_conf.captioning_lr_decay_epochs)
                )

                params.append(
                    {
                        "params": self.dense_captioner.clip_captioner.encoder.visual_model.parameters(),
                        "lr": opt_conf.captioning_lr,
                    }
                )
                lr_lambda.append(
                    lambda epoch: opt_conf.lr_decay_factor
                    ** (epoch // opt_conf.captioning_lr_decay_epochs)
                )
            if self.dense_captioner.training_sem_enc:
                params.append(
                    {
                        "params": self.dense_captioner.clip_captioner.encoder.sem_model.parameters(),
                        "lr": opt_conf.sem_enc_lr,
                    }
                )
                lr_lambda.append(
                    lambda epoch: opt_conf.lr_decay_factor
                    ** (epoch // opt_conf.sem_enc_lr_decay_epochs)
                )
            if self.dense_captioner.training_syn_enc:
                params.append(
                    {
                        "params": self.dense_captioner.clip_captioner.encoder.syn_model.parameters(),
                        "lr": opt_conf.syn_enc_lr,
                    }
                )
                lr_lambda.append(
                    lambda epoch: opt_conf.lr_decay_factor
                    ** (epoch // opt_conf.syn_enc_lr_decay_epochs)
                )

            self.optimizer = optim.Adam(
                params=params,
                lr=opt_conf.learning_rate,
            )  # , weight_decay=.0001)

        # learning-rate decay scheduler
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lr_lambda,
        )

        # Loss function
        self.criterion = DenseCaptioningLoss(
            config=trainer_config.criterion_config,
            c_max_len=self.max_words,
            p_max_len=self.max_prog,
            sem_enc_pos_weights=self.sem_enc_pos_weights,
            s_prop_pos_weights=self.s_prop_pos_weights,
            e_prop_pos_weights=self.e_prop_pos_weights,
            training_proposals=self.dense_captioner.training_proposals,
            training_programmer=self.dense_captioner.training_programmer,
            training_pos_tagging=self.dense_captioner.training_syn_enc,
            training_sem_tagging=self.dense_captioner.training_sem_enc,
            training_captioning=self.dense_captioner.training_captioning,
            device=self.device,
        )
        self.use_dynamic_backward = trainer_config.criterion_config.use_dynamic_backward

        print("\n****We are ready to start the training process****\n")

    def __load_ground_truth(self):
        self.ref_programs, self.ref_captions, self.ref_pos, self.ref_densecaps = (
            {},
            {},
            {},
            {},
        )

        ref_progams_txt_path = {
            "val_1": os.path.join(self.dataset_folder, "val_1_ref_programs.txt")
        }
        ref_captions_txt_path = {
            "val_1": os.path.join(self.dataset_folder, "val_1_ref_captions.txt")
        }
        ref_densecap_json_path = {
            "val_1": os.path.join(self.dataset_folder, "val_1_ref_densecap.json")
        }

        ref_vidxs_blacklists = {"val_1": self.trainer_config.valid_blacklist}
        ref_cidxs_blacklists = {
            "val_1": [
                cidx
                for vidx in ref_vidxs_blacklists["val_1"]
                for cidx in self.corpus[1][1][self.corpus[1][0].index(vidx)]
            ]
        }

        for phase in ["val_1"]:
            self.ref_programs[phase] = load_texts(
                ref_progams_txt_path[phase], blacklist=ref_vidxs_blacklists[phase]
            )
            self.ref_captions[phase] = load_texts(
                ref_captions_txt_path[phase], blacklist=ref_cidxs_blacklists[phase]
            )
            self.ref_pos[phase] = {
                k: [
                    " ".join(
                        [t[1] for t in nltk.pos_tag(nltk.word_tokenize(cap.lower()))]
                    )
                    for cap in caps
                ]
                for k, caps in self.ref_captions[phase].items()
            }

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

    def __get_most_freq_words(
        self,
        caps,
        caps_upos,
        postags=["NOUN", "ADJ", "VERB"],
        words_to_discard=["<unk>"],
    ):
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

        freq_words = heapq.nlargest(
            self.modules_config["sem_tagger_config"].out_size,
            widx2count,
            key=widx2count.get,
        )
        self.sem_enc_keywords = [self.caps_vocab.idx_to_word(idx) for idx in freq_words]

        self.logger.info(f"TAGs-IDXs: {freq_words}")
        self.logger.info(f"TAGs-words: " + " ".join(self.sem_enc_keywords))
        self.logger.info(f"TAGs-freq: {[widx2count[idx] for idx in freq_words]}")
        self.logger.info(
            f"TAGs-total freq of tags: {sum([widx2count[idx] for idx in freq_words])}"
        )
        self.logger.info(
            f"TAGs-mean freq of tags: {np.mean([widx2count[idx] for idx in freq_words])}"
        )

        return freq_words

    def __get_sem_enc(
        self, freq_words, caps, caps_upos, postags=["NOUN", "ADJ", "VERB"]
    ):
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

        # For maximizing Recall
        pos_weights = neg_samples / pos_samples

        # For maximizing Precision (diden't work)
        # pos_weights = pos_samples / neg_samples

        return X, pos_weights, total_num_caps

    def __get_interval_mask(
        self,
        intervals,
        caps_count,
        max_num_chunks,
        proposals=None,
        num_estimates=128,
        min_count_per_proposal=20,
    ):
        # compute the length of all intervals, including padding region
        aux = intervals[:, :, 1] - intervals[:, :, 0]

        # sanity: replace negative interval durations by zero
        aux[aux < 0] = 0

        # filter the length of real intervals only, discarding the padding region that can affect clustering
        data = torch.cat([aux[i, :c] for i, c in enumerate(caps_count)])
        # data = (aux[aux>0]).view(-1)

        # for determining the masks of validation split we use the proposals determined from training split
        if proposals is None:
            # determine clusters according to intervals length
            print(
                "PROPOSALS: Computing event-proposals by the KernelDensity algorithm "
            )
            kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(
                data.unsqueeze(1).numpy()
            )
            s = np.linspace(0, self.max_interval, num=num_estimates)
            e = kde.score_samples(s.reshape(-1, 1))
            proposals = s[argrelextrema(e, np.less)[0]]
            self.logger.info(f"PROPOSALS: Number of event-proposals: {len(proposals)}")
            self.logger.info(f"PROPOSALS: Event-proposals: {proposals}")
            print(f"PROPOSALS: Number of event-proposals: {len(proposals)}")

            # filter proposals with less than min_count_per_proposal of events
            filter_proposals, filter_proposals_count = [], []

            def append_porposal(p, count):
                filter_proposals.append(p)
                filter_proposals_count.append(count.item())

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
            if current_sum < min_count_per_proposal:
                filter_proposals.pop()
                filter_proposals_count[-1] += current_sum
            else:
                filter_proposals_count.append(current_sum.item())

            proposals = filter_proposals
            self.logger.info(
                f"PROPOSALS: Number of event-proposals (filtered): {len(proposals)}"
            )
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

        if "filter_proposals_count" in dir():
            # check results were correctly created
            assert clusters_sizes == filter_proposals_count

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
                        if (
                            result[i, k] != 0
                        ):  # to help the low represented clusters, set the next frames too
                            s_mask[i, s + 1, result[i, k]] = 1
                    if intervals[i, k, 1] >= e:
                        # interval that starts before and ends after the current interval ends
                        e_mask[i, e, result[i, k]] = 1
                        if (
                            result[i, k] != 0
                        ):  # to help the low represented clusters, set the prev frames too
                            e_mask[i, e - 1, result[i, k]] = 1

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
        self.logger.info(
            f"PROPOSALS: Count of positive examples per cluster (start positions): {s_pos_samples}"
        )
        self.logger.info(
            f"PROPOSALS: Count of positive examples per cluster (end positions): {e_pos_samples}"
        )
        print(
            "PROPOSALS: Count of positive examples per cluster (start positions): ",
            s_pos_samples,
        )
        print(
            "PROPOSALS: Count of positive examples per cluster (end positions): ",
            e_pos_samples,
        )

        # determine the number of negative examples per cluster, descarding the frames where we will not classify
        s_neg_mask = 1 - s_mask
        s_frame_mask = (
            s_neg_mask.sum(dim=-1, keepdim=True) != (len(proposals) + 1)
        ).repeat(1, 1, len(proposals) + 1)
        s_neg_samples = (
            (s_neg_mask * s_frame_mask).sum(dim=1).sum(dim=0)
        )  # (len(proposals) + 1, )

        e_neg_mask = 1 - e_mask
        e_frame_mask = (
            e_neg_mask.sum(dim=-1, keepdim=True) != (len(proposals) + 1)
        ).repeat(1, 1, len(proposals) + 1)
        e_neg_samples = (
            (e_neg_mask * e_frame_mask).sum(dim=1).sum(dim=0)
        )  # (len(proposals) + 1, )

        self.logger.info(
            f"PROPOSALS: Count of negative examples per cluster (start positions): {s_neg_samples}"
        )
        self.logger.info(
            f"PROPOSALS: Count of negative examples per cluster (end positions): {e_neg_samples}"
        )
        print(
            "PROPOSALS: Count of negative examples per cluster (start positions): ",
            s_neg_samples,
        )
        print(
            "PROPOSALS: Count of negative examples per cluster (end positions): ",
            e_neg_samples,
        )

        s_prop_pos_weights = s_neg_samples / s_pos_samples
        e_prop_pos_weights = e_neg_samples / e_pos_samples

        # save correlation matrices of start and end proposals
        # print("PROPOSALS: generating correlation images...")
        # s_mask_corr = torch.zeros((len(proposals)+1, len(proposals)+1))
        # e_mask_corr = torch.zeros((len(proposals)+1, len(proposals)+1))
        # for p1 in range(len(proposals)+1):
        #     for p2 in range(p1):
        #         s_mask_corr[p1,p2] = len(s_mask[(s_mask[:,:,p1]==1.) & (s_mask[:,:,p2]==1.)])
        #         e_mask_corr[p1,p2] = len(e_mask[(e_mask[:,:,p1]==1.) & (e_mask[:,:,p2]==1.)])
        # print("PROPOSALS: correlation data extracted")

        # fig = plt.figure(figsize=((len(proposals)+1)//2, (len(proposals)+1)//2))
        # threshold = s_mask_corr.max() / 2.0
        # for i, j in itertools.product(range(s_mask_corr.shape[0]), range(s_mask_corr.shape[1])):
        #     color = "black" if s_mask_corr[i, j] > threshold else "white"
        #     plt.text(j, i, s_mask_corr[i, j].item(), horizontalalignment="center", color=color)
        # plt.imshow(s_mask_corr)
        # self.writer.add_figure(f"proposals/s-prop-corr", fig)
        # print("PROPOSALS: correlation image saved (start positions)")

        # fig = plt.figure(figsize=((len(proposals)+1)//2, (len(proposals)+1)//2))
        # threshold = e_mask_corr.max() / 2.0
        # for i, j in itertools.product(range(e_mask_corr.shape[0]), range(e_mask_corr.shape[1])):
        #     color = "black" if e_mask_corr[i, j] > threshold else "white"
        #     plt.text(j, i, e_mask_corr[i, j].item(), horizontalalignment="center", color=color)
        # plt.imshow(e_mask_corr)
        # self.writer.add_figure(f"proposals/e-prop-corr", fig)
        # print("PROPOSALS: correlation image saved (end positions)")

        return s_mask, e_mask, proposals, s_prop_pos_weights, e_prop_pos_weights

    def __init_dense_loader(self):
        print("Initializing data loaders...")

        # get train split data
        print(" initializing train split data loader...")
        (
            vidxs,
            cidxs,
            intervals,
            fps,
            progs,
            prog_lens,
            caps,
            pos,
            upos,
            cap_lens,
        ) = extract_split_data_from_corpus(self.corpus, split=0)
        (
            cidxs_t,
            intervals_t,
            caps_count_t,
            progs_t,
            caps_t,
            pos_t,
            upos_t,
            cap_lens_t,
        ) = data2tensors(cidxs, intervals, progs, prog_lens, caps, pos, upos, cap_lens)
        self.max_prog = progs_t.size(1)
        self.max_caps = caps_t.size(1)
        self.max_words = caps_t.size(2)
        self.max_interval = torch.max(
            intervals_t.view(-1, 2)[:, 1] - intervals_t.view(-1, 2)[:, 0]
        )
        self.last_interval_end = torch.max(intervals_t.view(-1, 2)[:, 1])

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
            min_count_per_proposal=self.modules_config[
                "proposals_tagger_config"
            ].min_count_per_proposal,
        )
        self.s_prop_pos_weights = s_prop_pos_weights.to(self.device)
        self.e_prop_pos_weights = e_prop_pos_weights.to(self.device)
        self.num_proposals = len(event_proposals) + 1

        # determine the K most frequent words for semantic encodings from the train split
        freq_words = self.__get_most_freq_words(
            caps,
            upos,
            postags=self.modules_config["sem_tagger_config"].upos_tags,
            words_to_discard=self.modules_config["sem_tagger_config"].words_to_discard,
        )

        caps_sem_enc_t, sem_enc_pos_weights, total_num_caps = self.__get_sem_enc(
            freq_words, caps, upos
        )
        self.sem_enc_pos_weights = sem_enc_pos_weights.to(self.device)

        self.logger.info(
            f"Train split TAGs-percent: {caps_sem_enc_t.sum(dim=0).sum(dim=0) / total_num_caps}"
        )

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
            caps_sem_enc=caps_sem_enc_t,
            pos=pos_t,
            upos=upos_t,
            cap_lens=cap_lens_t,
            progs=progs_t,
            prog_lens=prog_lens,
            event_proposals_s=event_s_mask_t,
            event_proposals_e=event_e_mask_t,
            batch_size=self.trainer_config.train_batch_size,
            train=True,
            num_workers=self.trainer_config.loader_num_workers,
            pin_memory=self.trainer_config.loader_pin_memory,
        )

        # get valid split data
        print(" initializing valid split data loader...")
        (
            vidxs,
            cidxs,
            intervals,
            fps,
            progs,
            prog_lens,
            caps,
            pos,
            upos,
            cap_lens,
        ) = extract_split_data_from_corpus(self.corpus, split=1)
        (
            cidxs_t,
            intervals_t,
            caps_count_t,
            progs_t,
            caps_t,
            pos_t,
            upos_t,
            cap_lens_t,
        ) = data2tensors(
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

        # determine the ground truth for event masking
        event_s_mask_t, event_e_mask_t, _, _, _ = self.__get_interval_mask(
            intervals_t,
            caps_count_t,
            max_num_chunks=self.trainer_config.max_num_chunks,
            proposals=event_proposals,
        )
        caps_sem_enc_t, _, total_num_caps = self.__get_sem_enc(freq_words, caps, upos)

        self.logger.info(
            f"Validation split TAGs-percent: {caps_sem_enc_t.sum(dim=0).sum(dim=0) / total_num_caps}"
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
            caps_sem_enc=caps_sem_enc_t,
            pos=pos_t,
            upos=upos_t,
            cap_lens=cap_lens_t,
            progs=progs_t,
            prog_lens=prog_lens,
            event_proposals_s=event_s_mask_t,
            event_proposals_e=event_e_mask_t,
            batch_size=self.trainer_config.valid_batch_size,
            train=False,
            num_workers=self.trainer_config.loader_num_workers,
            pin_memory=self.trainer_config.loader_pin_memory,
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
        # print(f" Number of event-proposals: {self.num_proposals}")

        self.loaders = {"train": train_loader, "val_1": val_loader}

    @staticmethod
    def trained_epochs_per_module(epoch, change_after, module, dynamic=True):
        if dynamic:
            prev_periods = epoch // (change_after * 3)
            current_period_epochs = epoch % (change_after * 3)
            if module == "sem_enc":
                # epoch
                return prev_periods * 3 * change_after + (epoch % (change_after * 3))
            if module == "syn_enc":
                return prev_periods * 2 * change_after + (
                    (current_period_epochs - change_after)
                    if current_period_epochs > change_after
                    else 0
                )
            if module == "cap_dec":
                return prev_periods * 1 * change_after + (
                    (current_period_epochs - change_after * 2)
                    if current_period_epochs > change_after * 2
                    else 0
                )
        else:
            return epoch

    def tf_ratio_per_module(self, tf_config, module):
        return get_tf_ratio(tf_config, self.trained_epochs[module])

    def get_unfreezed_modules(self):
        return [
            m for m in ["sem_enc", "syn_enc", "cap_dec"] if not self.freezed_modules[m]
        ]

    def freeze_modules(
        self,
        epoch,
        phase="val_1",
        early_stop_limits={"sem_enc": 2, "syn_enc": 3, "cap_dec": 10},
    ):
        if (
            self.sem_loss_phase[phase] > self.best_sem_loss_phase[phase]
            and self.sem_enc_metrics_results["AP/weighted"]
            < self.best_metrics["sem_enc"][phase]["AP/weighted"][1]
        ):
            self.sem_early_stop[phase] += 1
            if self.sem_early_stop[phase] == early_stop_limits["sem_enc"]:
                self.freezing_last_change = epoch
                self.freezed_modules["sem_enc"] = True
                self.dense_captioner.freeze_dict(self.freezed_modules)

                # TODO: reload weights for the best saved checkpoint, for evaluation only
                # this can affect the others models because the semantic representations can change
        else:
            self.sem_early_stop[phase] = 0

        unfreezed = self.get_unfreezed_modules()

        if (len(unfreezed) == 3 and self.stage in [1, 2]) or len(unfreezed) in [1, 2]:
            if (
                self.pos_loss_phase[phase] > self.best_pos_loss_phase[phase]
                and self.syn_enc_metrics_results["Bleu_4"]
                < self.best_metrics["syn_enc"][phase]["Bleu_4"][1]
            ):
                self.pos_early_stop[phase] += 1
                if self.pos_early_stop[phase] == early_stop_limits["syn_enc"]:
                    self.freezing_last_change = epoch
                    self.freezed_modules["syn_enc"] = True
                    self.dense_captioner.freeze_dict(self.freezed_modules)
            else:
                self.pos_early_stop[phase] = 0

        if (
            (len(unfreezed) == 3 and self.stage == 2)
            or (len(unfreezed) == 2 and self.stage == 1)
            or len(unfreezed) == 1
        ):
            if (
                self.cap_loss_phase[phase] > self.best_cap_loss_phase[phase]
                and self.cap_metrics_results["METEOR"]
                < self.best_metrics["captioning"][phase]["METEOR"][1]
            ):
                self.cap_early_stop[phase] += 1
                if self.cap_early_stop[phase] == early_stop_limits["cap_dec"]:
                    self.freezing_last_change = epoch
                    self.freezed_modules["cap_dec"] = True
                    self.dense_captioner.freeze_dict(self.freezed_modules)
            else:
                self.cap_early_stop[phase] = 0

        return (
            self.freezed_modules["sem_enc"]
            and self.freezed_modules["syn_enc"]
            and self.freezed_modules["cap_dec"]
        )

    def determine_stage(self, epoch, change_after):
        unfreezed = self.get_unfreezed_modules()
        if len(unfreezed) > 0:
            self.stage = ((epoch - self.freezing_last_change) // change_after) % len(
                unfreezed
            )

    def dynamic_backward(self, loss1, loss2, loss3):
        if self.use_dynamic_backward:
            unfreezed = self.get_unfreezed_modules()
            if len(unfreezed) == 3:
                loss1.backward()  # sem_enc is trained all the time
                if self.stage in [1, 2]:
                    loss2.backward()  # syn_enc is trained in stages 1 and 2
                if self.stage == 2:
                    loss3.backward()  # cap_dec is only trained in stage 2
            elif len(unfreezed) == 2:  # will be only two training stages too
                # only one between sem_enc and syn_enc is unfreezed and will be trained all the time
                if "sem_enc" in unfreezed:
                    loss1.backward()  # sem module is always trained
                elif "syn_enc" in unfreezed:
                    loss2.backward()  # syn module is always trained

                if self.stage == 1:
                    # I am assuming the decoder is always unfreezed, but will be trained in stage 1 only
                    loss3.backward()
            elif len(unfreezed) == 1:
                # I am assuming the decoder is always unfreezed and is the only mosule to be trained
                loss3.backward()
        else:
            if not self.freezed_modules["sem_enc"]:
                loss1.backward()
            if not self.freezed_modules["syn_enc"]:
                loss2.backward()
            if not self.freezed_modules["cap_dec"]:
                loss3.backward()

    def update_trained_epochs(self):
        if self.use_dynamic_backward:
            unfreezed = self.get_unfreezed_modules()

            if len(unfreezed) == 3:
                self.trained_epochs["sem_enc"] += 1
                if self.stage in [1, 2]:
                    self.trained_epochs["syn_enc"] += 1
                if self.stage == 2:
                    self.trained_epochs["cap_dec"] += 1
            elif len(unfreezed) == 2:
                if "sem_enc" in unfreezed:
                    # sem module is always trained
                    self.trained_epochs["sem_enc"] += 1
                elif "syn_enc" in unfreezed:
                    # syn module is always trained
                    self.trained_epochs["syn_enc"] += 1
                if self.stage == 1:
                    # I am assuming the decoder is always unfreezed
                    self.trained_epochs["cap_dec"] += 1
            elif len(unfreezed) == 1:
                self.trained_epochs["cap_dec"] += 1
        else:
            for k, v in self.freezed_modules:
                if not v:
                    self.trained_epochs[k] += 1

    def __process_batch(
        self,
        epoch,
        video_feats,
        feats_count,
        gt_intervals,
        gt_caps_count,
        gt_captions,
        gt_caps_sem_enc,
        gt_pos,
        gt_upos,
        gt_cap_lens,
        gt_program,
        gt_prog_len,
        gt_prop_s,
        gt_prop_e,
        tf_ratio=0.5,
        phase="train",
    ):
        bsz = video_feats[0].size(0)

        # Move all tensors to device
        video_feats = [f.to(self.device) for f in video_feats]
        feats_count = feats_count.to(self.device)
        # gt_intervals = gt_intervals.to(self.device)
        gt_captions = gt_captions.to(self.device)
        gt_pos = gt_pos.to(self.device)
        # gt_upos = gt_upos.to(self.device)
        gt_cap_lens = gt_cap_lens.to(self.device)
        gt_program = gt_program.to(self.device)
        gt_prog_len = gt_prog_len.to(self.device)
        gt_caps_sem_enc = gt_caps_sem_enc.to(self.device)

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
            [
                gt_prop_s[torch.arange(bsz), gt_intervals[:, i, 0].long()].unsqueeze(1)
                for i in range(max_caps)
            ],
            dim=1,
        ).to(self.device)
        gt_prop_e = torch.cat(
            [
                gt_prop_e[
                    torch.arange(bsz),
                    gt_intervals[:, i, 1].clamp(max=last_chunk).long(),
                ].unsqueeze(1)
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

            (
                prog_logits,
                caps_logits,
                caps_sem_enc,
                caps_pos_tags,
                pos_tags_logits,
                caps,
                intervals,
                caps_count,
                s_prop_logits,
                e_prop_logits,
                _,
            ) = self.dense_captioner(
                v_feats=video_feats,
                feats_count=feats_count,
                prog_len=int(gt_prog_len.max()),  # truncate_prog_at,
                tf_ratios=tf_ratio,
                gt_prog=gt_program,
                gt_caps=gt_captions,
                gt_caps_count=gt_caps_count,
                gt_sem_enc=gt_caps_sem_enc,
                gt_pos=gt_pos,
                gt_intervals=gt_intervals,
                gt_prop_s=gt_prop_s,
                gt_prop_e=gt_prop_e,
                max_prog=self.avg_truncation,  # max_prog=self.max_prog,
                max_caps=self.avg_caps,  # max_caps=self.max_caps,
                max_cap=self.max_words,
                max_chunks=self.avg_truncation,
                max_back_steps=self.modules_config[
                    "proposals_tagger_config"
                ].max_back_steps,
                captioning_batch=self.modules_config[
                    "dense_captioner_config"
                ].captioning_batch,
            )  # the maximum value of start pointers is lower than the max_prog to be generated
            self.logger.info(f"proposals count: {caps_count}")

            # if "val" in phase:
            #     gt_caps_count = gt_caps_count.to(self.device)
            #     gt_caps_count = torch.min(gt_caps_count, caps_count)

            # compute the loss functions
            self.logger.info("loss computation....")

            loss = self.criterion(
                gt_caps=gt_captions,
                gt_cap_lens=gt_cap_lens,
                pred_caps=caps_logits,
                gt_caps_sem_enc=gt_caps_sem_enc,
                pred_caps_sem_enc=caps_sem_enc,
                gt_pos_seq=gt_pos,
                pred_pos_seq=pos_tags_logits,
                gt_program=gt_program,
                gt_prog_len=gt_prog_len,
                pred_program=None,  # prog_logits,
                gt_intervals=gt_intervals,
                pred_intervals=None,  # intervals,
                gt_prop_s=gt_prop_s,
                gt_prop_e=gt_prop_e,
                pred_prop_s=s_prop_logits,
                pred_prop_e=e_prop_logits,
                gt_caps_count=gt_caps_count,
                pred_caps_count=caps_count,
                gt_prop_count=None,  # proposals_count,
                epoch=epoch,
                truncate_prog_at=None,  # truncate_prog_at,
            )

        if phase == "train":
            # compute backward pass for somputing the gradients
            self.logger.info("loss backward....")

            # self.dynamic_backward(loss1=sem_enc_loss, loss2=pos_loss, loss3=cap_loss)

            loss["total_loss"].backward()

            # clip gradients to prevent NaNs in the prog-loss
            # nn.utils.clip_grad_norm_(self.dense_captioner.rnn_cell.parameters(), 0.5)

            # update the parameters
            self.logger.info("optimizer step...")
            self.optimizer.step()

        return (
            loss,
            prog_logits,
            caps,
            intervals,
            caps_sem_enc,
            caps_pos_tags,
            s_prop_logits,
            e_prop_logits,
            gt_prop_s,
            gt_prop_e,
            caps_count,
            None,  # truncate_prog_at,
        )

    def __save_checkpoint(
        self,
        epoch,
        save_checkpoints_dir,
        phase=None,
        new_best=False,
        component=None,
        metric_name=None,
    ):
        if new_best:
            parsed_metric_name = metric_name.replace("/", "-")
            chkpt_filename = (
                f"best_chkpt_{epoch}_{phase}_{component}_{parsed_metric_name}.pt"
            )
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
                    "trained_epochs": self.trained_epochs,
                    "avg_tuncation": self.avg_truncation,
                    "avg_caps": self.avg_caps,
                },
                f=os.path.join(save_checkpoints_dir, chkpt_filename),
            )
            print(" saved")

        # remove previously saved
        if new_best and self.last_best_saved_epoch[component][phase][metric_name] != -1:
            last_best_epoch = self.last_best_saved_epoch[component][phase][metric_name]
            os.remove(
                os.path.join(
                    save_checkpoints_dir,
                    f"best_chkpt_{last_best_epoch}_{phase}_{component}_{parsed_metric_name}.pt",
                )
            )
        elif not new_best and self.last_saved_epoch != -1:
            os.remove(
                os.path.join(save_checkpoints_dir, f"chkpt_{self.last_saved_epoch}.pt")
            )

        # update the last saved epoch
        if new_best:
            self.last_best_saved_epoch[component][phase][metric_name] = epoch
        else:
            self.last_saved_epoch = epoch

    def __process_results(
        self, metrics_results, prediction, phase, epoch, save_checkpoints_dir, component
    ):
        self.logger.info(f"{phase} set metrics for {component}: {metrics_results}")
        min_metrics = []
        output_saved = False
        for name, result in metrics_results.items():
            if not "ml-conf-mat" in name:
                self.writer.add_scalar(f"{component}/{phase}-{name}", result, epoch)

            if (
                phase != "train"
                and name in self.best_metrics[component][phase]
                and (
                    (
                        name in min_metrics
                        and result < self.best_metrics[component][phase][name][1]
                    )
                    or (
                        name not in min_metrics
                        and result > self.best_metrics[component][phase][name][1]
                    )
                )
            ):
                self.best_metrics[component][phase][name] = (epoch, result)
                if name in ["Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "All_Metrics"]:
                    self.early_stop_count = 0
                    if not output_saved:
                        with open(
                            os.path.join(
                                save_checkpoints_dir,
                                f"chkpt_{epoch}_{component}_output.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(prediction, f)
                        output_saved = True

                if component in [
                    "s_prop",
                    "e_prop",
                    "sem_enc",
                    "syn_enc",
                    "captioning",
                ]:
                    print(
                        f"saving best checkpoint due to improvement on {component}-{name}..."
                    )
                    self.early_stop_count = 0
                    if not output_saved:
                        with open(
                            os.path.join(
                                save_checkpoints_dir,
                                f"chkpt_{epoch}_{component}_output.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(prediction, f)
                        output_saved = True
                    self.__save_checkpoint(
                        epoch, save_checkpoints_dir, phase, True, component, name
                    )

        if "ml-conf-mat" in metrics_results:
            if "prop" in component:
                cols = int(np.sqrt(self.num_proposals))
                rows = self.num_proposals // cols
            elif "sem" in component:
                cols = 8
                rows = len(self.sem_enc_keywords) // cols

            fig = plt.figure(figsize=(int(rows * 2), int(cols * 2)))
            for i, norm_conf_mat in enumerate(metrics_results["norm-ml-conf-mat"]):
                plt.subplot(rows + 1, cols, i + 1, title=f"proposal-{i}")
                labels = np.around(norm_conf_mat, decimals=2)

                # Use white text if squares are dark; otherwise black.
                threshold = norm_conf_mat.max() / 2.0
                # threshold = norm_conf_mat.min() + (norm_conf_mat.max() - norm_conf_mat.min() / 2)
                for i, j in itertools.product(
                    range(norm_conf_mat.shape[0]), range(norm_conf_mat.shape[1])
                ):
                    color = "white" if norm_conf_mat[i, j] > threshold else "black"
                    plt.text(
                        j, i, labels[i, j], horizontalalignment="center", color=color
                    )

                # plt.ylabel('True label')
                # plt.xlabel('Predicted label')
                plt.imshow(norm_conf_mat, cmap="OrRd")
            fig.tight_layout(pad=1.0)
            self.writer.add_figure(
                f"{component}/{phase}-norm-confusion-matrix", fig, epoch
            )
            fig = plt.figure(figsize=(int(rows * 2), int(cols * 2)))
            for i, conf_mat in enumerate(metrics_results["ml-conf-mat"]):
                if "prop" in component:
                    plt.subplot(rows + 1, cols, i + 1, title=f"proposal-{i}")
                elif "sem" in component:
                    plt.subplot(rows + 1, cols, i + 1, title=self.sem_enc_keywords[i])

                # Use white text if squares are dark; otherwise black.
                threshold = conf_mat.max() / 2.0
                for i, j in itertools.product(
                    range(conf_mat.shape[0]), range(conf_mat.shape[1])
                ):
                    color = "white" if conf_mat[i, j] > threshold else "black"
                    plt.text(
                        j, i, conf_mat[i, j], horizontalalignment="center", color=color
                    )

                # plt.ylabel('True label')
                # plt.xlabel('Predicted label')
                plt.imshow(conf_mat, cmap="OrRd")
            fig.tight_layout(pad=1.0)
            self.writer.add_figure(f"{component}/{phase}-confusion-matrix", fig, epoch)

    def parse_proposals_prediction(self, vidxs, gt_multihots, pred_logits, cap_counts):
        predictions = {}
        for batch_vidxs, batch_gt, batch_pred, batch_count in zip(
            vidxs, gt_multihots, pred_logits, cap_counts
        ):
            for v_idx, v_gt, v_pred, v_count in zip(
                batch_vidxs, batch_gt, batch_pred, batch_count
            ):
                predictions[v_idx] = (
                    (v_gt[:v_count] == 1).nonzero().tolist(),
                    (torch.sigmoid(v_pred[:v_count]) > 0.5).nonzero().tolist(),
                )
        return predictions

    def train_model(
        self, resume=False, checkpoint_path=None, min_num_epochs=50, early_stop_limit=10
    ):
        # parallel_pool = Pool()
        # self.logger.info('Training captioning model on [{}] dataset with [{}] encoder and [{}] decoder'
        #                  .format(self.config.dataset_name, self.encoder_name, self.decoder_name))

        save_checkpoints_dir = os.path.join(
            self.out_folder, "models", self.trainer_config.str, self.datetime_str
        )
        if not os.path.exists(save_checkpoints_dir):
            os.makedirs(save_checkpoints_dir)

        val_phases = ["val_1"]
        prop_s_metrics = [
            "Recall/weighted",
            "F1/weighted",
            "ROC-AUC/weighted",
            "AP/weighted",
        ]
        prop_e_metrics = prop_s_metrics
        prog_metrics = ["Precision/weighted", "Recall/weighted", "F1/weighted"]
        sem_enc_metrics = prop_s_metrics
        syn_enc_metrics = [
            "Bleu_4",
            "METEOR",
            "ROUGE_L",
            "All_Metrics",
        ]
        cap_metrics = [
            "Bleu_4",
            "METEOR",
            "ROUGE_L",
            "CIDEr",
            "SPICE",
            "All_Metrics",
        ]
        densecap_metrics = [
            "Bleu_1",
            "Bleu_2",
            "Bleu_3",
            "Bleu_4",
            "METEOR",
            "ROUGE_L",
            "CIDEr",
            "SPICE",
            "Recall",
            "Precision",
            "All_Metrics",
        ]

        if resume and os.path.exists(checkpoint_path):
            self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            print(f"Resuming from checkpoint: {checkpoint_path}.... ")

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            begin_epoch = checkpoint["epoch"] + 1
            self.best_metrics = checkpoint["best_metrics"]
            # self.best_metrics = {"programmer": {}, "sem_enc": {}, "syn_enc": {}, "captioning": {}, "densecap": {}}
            for p in val_phases:
                # self.best_metrics["sem_enc"][p] = {m: (0, 0) for m in sem_enc_metrics}
                # self.best_metrics["syn_enc"][p] = {m: (0, 0) for m in syn_enc_metrics}
                self.best_metrics["captioning"][p] = {m: (0, 0) for m in cap_metrics}
                self.best_metrics["densecap"][p] = {m: (0, 0) for m in densecap_metrics}

            self.avg_caps = checkpoint["avg_caps"]

            log_msg = f" (epoch {begin_epoch})"
            for phase in val_phases:
                log_msg += f"\n  SemanticEncoding metrics {phase}: \n   "
                log_msg += "\t".join(
                    [
                        f"{k}:({e:03d}, {v:.3f})"
                        for k, (e, v) in self.best_metrics["sem_enc"][phase].items()
                    ]
                )
                log_msg += f"\n  SyntacticEncoding metrics {phase}: \n   "
                log_msg += "\t".join(
                    [
                        f"{k}:({e:03d}, {v:.3f})"
                        for k, (e, v) in self.best_metrics["syn_enc"][phase].items()
                    ]
                )
                log_msg += f"\n  S-Proposals metrics {phase}: \n   "
                log_msg += "\t".join(
                    [
                        f"{k}:({e:03d}, {v:.3f})"
                        for k, (e, v) in self.best_metrics["s_prop"][phase].items()
                    ]
                )
                log_msg += f"\n  E-Proposals metrics {phase}: \n   "
                log_msg += "\t".join(
                    [
                        f"{k}:({e:03d}, {v:.3f})"
                        for k, (e, v) in self.best_metrics["e_prop"][phase].items()
                    ]
                )
            print(log_msg, "\n")
            self.logger.info(log_msg)

            model_dict = self.dense_captioner.state_dict()

            # 1. filter out unnecessary parameters (not included in the current architecture)
            pretrained_dict = {
                "proposals_enc." + k: v
                for k, v in checkpoint["dense_captioner"].items()
                if "proposals_enc." + k in model_dict
            }

            # 2. include in the dictionary to be loaded the new parameters in the current architecture
            # for k, v in model_dict.items():
            #    if k not in pretrained_dict:  # or k[:2]=='fc':
            #        pretrained_dict[k] = v

            # 3. load the new state dict
            self.dense_captioner.load_state_dict(
                pretrained_dict, self.trainer_config.resume_config
            )

            # 5. set the begin_epoch variable for logging
            if self.trainer_config.resume_config.begin_epoch != -1:
                begin_epoch = self.trainer_config.resume_config.begin_epoch

            if "trained_epochs" in checkpoint:
                self.trained_epochs = checkpoint["trained_epochs"]
            else:
                self.trained_epochs = {
                    m: DenseVideo2TextTrainer.trained_epochs_per_module(
                        begin_epoch,
                        self.trainer_config.change_stage_after,
                        m,
                        self.use_dynamic_backward,
                    )
                    for m in ["cap_dec", "sem_enc", "syn_enc"]
                }
        else:
            begin_epoch = 0
            self.best_metrics = {
                "prop_s": {p: {m: (0, 0) for m in prop_s_metrics} for p in val_phases},
                "prop_e": {p: {m: (0, 0) for m in prop_e_metrics} for p in val_phases},
                "programmer": {
                    p: {m: (0, 0) for m in prog_metrics} for p in val_phases
                },
                "sem_enc": {
                    p: {m: (0, 0) for m in sem_enc_metrics} for p in val_phases
                },
                "syn_enc": {
                    p: {m: (0, 0) for m in syn_enc_metrics} for p in val_phases
                },
                "captioning": {p: {m: (0, 0) for m in cap_metrics} for p in val_phases},
                "densecap": {
                    p: {m: (0, 0) for m in densecap_metrics} for p in val_phases
                },
            }
            self.trained_epochs = {
                m: 0
                for m in [
                    "prop_s",
                    "prop_e",
                    "programmer",
                    "cap_dec",
                    "sem_enc",
                    "syn_enc",
                ]
            }

        self.last_best_saved_epoch = {
            comp: {p: {m: -1 for m in metrics} for p in val_phases}
            for comp, metrics in [
                ("prop_s", prop_s_metrics),
                ("prop_e", prop_e_metrics),
                ("programmer", prog_metrics),
                ("cap_dec", cap_metrics),
                ("sem_enc", sem_enc_metrics),
                ("syn_enc", syn_enc_metrics),
            ]
        }

        self.dense_captioner.to(self.device)
        print("\nParameters of Dense Captioner model:\n")
        total_size = 0
        for n, p in self.dense_captioner.named_parameters():
            # print(n, p.size(), p.device)
            total_size += torch.numel(p)
        print(" total size: ", (total_size * 8) / (1024**3), "\n")

        # Start training process
        self.early_stop_count, self.last_saved_epoch = 0, -1
        time_phase = {p: 0 for p in ["train"] + val_phases}
        (
            self.prog_metrics_results,
            self.cap_metrics_results,
            self.densecap_metrics_results,
        ) = (None, None, None)
        s_prop_metrics_results = None
        e_prop_metrics_results = None
        self.best_loss_phase = {p: float("inf") for p in ["train"] + val_phases}
        self.best_s_prop_loss_phase, self.s_prop_early_stop = (
            {p: float("inf") for p in ["train"] + val_phases},
            {p: 0 for p in ["train"] + val_phases},
        )
        self.best_e_prop_loss_phase, self.e_prop_early_stop = (
            {p: float("inf") for p in ["train"] + val_phases},
            {p: 0 for p in ["train"] + val_phases},
        )
        self.best_prog_loss_phase, self.prog_early_stop = (
            {p: float("inf") for p in ["train"] + val_phases},
            {p: 0 for p in ["train"] + val_phases},
        )
        self.best_cap_loss_phase, self.cap_early_stop = (
            {p: float("inf") for p in ["train"] + val_phases},
            {p: 0 for p in ["train"] + val_phases},
        )
        self.best_sem_loss_phase, self.sem_early_stop = (
            {p: float("inf") for p in ["train"] + val_phases},
            {p: 0 for p in ["train"] + val_phases},
        )
        self.best_pos_loss_phase, self.pos_early_stop = (
            {p: float("inf") for p in ["train"] + val_phases},
            {p: 0 for p in ["train"] + val_phases},
        )

        self.dense_captioner.freeze_config(
            config_obj=self.trainer_config.freezing_config
        )
        self.freezed_modules = {
            "sem_enc": self.trainer_config.freezing_config.freeze_cap_sem_enc,
            "syn_enc": self.trainer_config.freezing_config.freeze_cap_syn_enc,
            "cap_dec": self.trainer_config.freezing_config.freeze_cap_decoder,
        }
        self.freezing_last_change = 0

        for epoch in range(begin_epoch, 1000):
            self.determine_stage(
                epoch, change_after=self.trainer_config.change_stage_after
            )
            self.update_trained_epochs()

            # unfreeze the freezed part of the model if needed
            if (
                not self.use_dynamic_backward
                and epoch == self.trainer_config.resume_config.unfreeze_at
            ):
                self.dense_captioner.unfreeze()

            prog_tf_ratio = self.tf_ratio_per_module(
                self.trainer_config.tf_config, "programmer"
            )
            prop_s_tf_ratio = self.tf_ratio_per_module(
                self.trainer_config.tf_config, "prop_s"
            )
            prop_e_tf_ratio = self.tf_ratio_per_module(
                self.trainer_config.tf_config, "prop_e"
            )
            syn_enc_tf_ratio = self.tf_ratio_per_module(
                self.trainer_config.tf_config, "syn_enc"
            )
            cap_dec_tf_ratio = self.tf_ratio_per_module(
                self.trainer_config.tf_config, "cap_dec"
            )

            self.writer.add_scalar("programmer/tf_ratio", prog_tf_ratio, epoch)
            self.writer.add_scalar("prop_s/tf_ratio", prop_s_tf_ratio, epoch)
            self.writer.add_scalar("prop_e/tf_ratio", prop_e_tf_ratio, epoch)
            self.writer.add_scalar("syn_enc/tf_ratio", syn_enc_tf_ratio, epoch)
            self.writer.add_scalar("captioning/tf_ratio", cap_dec_tf_ratio, epoch)

            tf_ratios = {
                "programmer": prog_tf_ratio,
                "prop_s": prop_s_tf_ratio,
                "prop_e": prop_e_tf_ratio,
                "syn_enc": syn_enc_tf_ratio,
                "cap_dec": cap_dec_tf_ratio,
            }

            lrs = self.lr_scheduler.get_last_lr()
            lr_idx = 0
            if self.dense_captioner.training_proposals:
                self.writer.add_scalar("prop_s/lr", lrs[lr_idx], epoch)
                self.writer.add_scalar("prop_e/lr", lrs[lr_idx], epoch)
                lr_idx += 1
            if self.dense_captioner.training_programmer:
                self.writer.add_scalar("programmer/lr", lrs[lr_idx], epoch)
                lr_idx += 1
            if self.dense_captioner.training_captioning:
                self.writer.add_scalar("captioning/lr_cap_dec", lrs[lr_idx], epoch)
                lr_idx += 1
                self.writer.add_scalar("captioning/lr_visual_enc", lrs[lr_idx], epoch)
                lr_idx += 1
            if self.dense_captioner.training_sem_enc:
                self.writer.add_scalar("sem_enc/lr", lrs[lr_idx], epoch)
                lr_idx += 1
            if self.dense_captioner.training_syn_enc:
                self.writer.add_scalar("sem_enc/lr", lrs[lr_idx], epoch)
                lr_idx += 1

            loss_phase = {p: 0 for p in ["train"] + val_phases}
            s_prop_loss_phase = {p: 0 for p in ["train"] + val_phases}
            e_prop_loss_phase = {p: 0 for p in ["train"] + val_phases}
            prog_loss_phase = {p: 0 for p in ["train"] + val_phases}
            cap_loss_phase = {p: 0 for p in ["train"] + val_phases}
            sem_loss_phase = {p: 0 for p in ["train"] + val_phases}
            pos_loss_phase = {p: 0 for p in ["train"] + val_phases}
            for phase in ["train"] + val_phases:
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
                        f"data/{phase}-epochs-avg_truncation",
                        self.avg_truncation,
                        epoch,
                    )
                    self.writer.add_scalar(
                        f"data/{phase}-epochs-avg_caps", self.avg_caps, epoch
                    )
                    self.writer.add_scalar(
                        f"data/{phase}-epochs-avg_feats", self.avg_feats, epoch
                    )

                # predicted_sentences = {}
                time_start_epoch, total_time_iters = time.perf_counter(), 0
                loss_count = 0
                prog_loss_count = 0
                s_prop_loss_count = 0
                e_prop_loss_count = 0
                sem_loss_count = 0
                pos_loss_count = 0
                cap_loss_count = 0
                all_programs, all_gt_programs = [], []
                all_captions = []
                all_syn_enc = []
                all_sem_enc, all_gt_sem_enc = [], []
                all_prog_ids = []
                all_caps_ids = []
                all_tstamps = []
                all_vidxs = []
                all_props_s = []
                all_gt_props_s = []
                all_props_e = []
                all_gt_props_e = []
                all_cap_counts = []
                all_prog_lens = []
                all_f_counts = []
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
                        gt_caps,
                        gt_caps_sem_enc,
                        gt_pos,
                        gt_upos,
                        gt_cap_lens,
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
                        prog_logits,
                        captions,
                        intervals,
                        caps_sem_enc_logits,
                        caps_pos_tags,
                        s_prop_logits,
                        e_prop_logits,
                        gt_prop_s,
                        gt_prop_e,
                        caps_count,
                        _,
                    ) = self.__process_batch(
                        epoch,
                        video_feats,
                        feats_count,
                        gt_intervals,
                        gt_caps_count,
                        gt_caps,
                        gt_caps_sem_enc,
                        gt_pos,
                        gt_upos,
                        gt_cap_lens,
                        gt_prog,
                        gt_prog_len,
                        gt_prop_s,
                        gt_prop_e,
                        tf_ratios,
                        phase,
                    )
                    loss_count += loss["total_loss"].item()

                    rl_strategy = self.trainer_config.criterion_config.rl_strategy
                    self.writer.add_scalar(
                        f"end2end/{phase}-iters-{rl_strategy}-loss",
                        loss["total_loss"],
                        iteration,
                    )

                    # logging message
                    total_time_iters += time.perf_counter() - time_start_iter
                    lrs = self.lr_scheduler.get_last_lr()
                    gpu_temp = get_gpu_temps(self.device)
                    log_msg = (
                        "\rEpoch:{0:03d} Phase:{1:6s} Iter:{2:04d}/{3:04d} avg-Time:{4:.1f}s lr:{5:.6f} gpu-temp:{6:02d} Loss:{7:9.4f} "
                    ).format(
                        epoch,
                        phase,
                        i,
                        len(self.loaders[phase]),
                        total_time_iters / i,
                        lrs[0],
                        gpu_temp,
                        loss["total_loss"].item(),
                    )

                    log_msg += "\t["
                    if self.dense_captioner.training_proposals:
                        s_prop_loss_count += loss["prop_loss"][1].item()
                        e_prop_loss_count += loss["prop_loss"][2].item()

                        self.writer.add_scalar(
                            f"proposals/{phase}-iters-s_prop_loss",
                            loss["prop_loss"][1],
                            iteration,
                        )
                        self.writer.add_scalar(
                            f"proposals/{phase}-iters-e_prop_loss",
                            loss["prop_loss"][2],
                            iteration,
                        )

                        log_msg += " s-prop-loss:{0:9.4f} e-prop-loss:{1:9.4f} ".format(
                            loss["prop_loss"][1].item(), loss["prop_loss"][2].item()
                        )
                    if self.dense_captioner.training_programmer:
                        prog_loss_count += loss["prog_loss"].item()
                        self.writer.add_scalar(
                            f"programmer/{phase}-iters-prog_loss",
                            loss["prog_loss"],
                            iteration,
                        )
                        log_msg += " prog-loss:{0:9.4f} ".format(
                            loss["prog_loss"].item()
                        )
                    if self.dense_captioner.training_captioning:
                        cap_loss_count += loss["cap_loss"].item()
                        self.writer.add_scalar(
                            f"captioning/{phase}-iters-loss",
                            loss["cap_loss"],
                            iteration,
                        )
                        log_msg += " cap-loss:{0:9.4f} ".format(
                            loss["prog_loss"].item()
                        )
                    if self.dense_captioner.training_sem_enc:
                        sem_loss_count += loss["sem_enc_loss"].item()
                        self.writer.add_scalar(
                            f"sem_enc/{phase}-iters-loss",
                            loss["sem_enc_loss"],
                            iteration,
                        )
                        log_msg += " sem-loss:{0:9.4f} ".format(
                            loss["sem_enc_loss"].item()
                        )
                    if self.dense_captioner.training_syn_enc:
                        pos_loss_count += loss["pos_loss"].item()
                        self.writer.add_scalar(
                            f"syn_enc/{phase}-iters-loss", loss["pos_loss"], iteration
                        )
                        log_msg += " pos-loss:{0:9.4f} ".format(loss["pos_loss"].item())
                    log_msg += "]"

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

                    all_cap_counts.append(caps_count.to("cpu"))

                    if self.dense_captioner.training_proposals:
                        all_vidxs.append(vidx.tolist())
                        # save s_proposals for computing evaluation metrics
                        all_props_s.append(s_prop_logits.to("cpu").detach())
                        all_gt_props_s.append(gt_prop_s.to("cpu"))
                        # save e_proposals for computing evaluation metrics
                        all_props_e.append(e_prop_logits.to("cpu").detach())
                        all_gt_props_e.append(gt_prop_e.to("cpu"))
                    if self.dense_captioner.training_programmer:
                        # save programs for computing evaluation metrics
                        all_programs.append(prog_logits.to("cpu").detach())
                        all_gt_programs.append(gt_prog.to("cpu"))
                        all_prog_lens.append(gt_prog_len.to("cpu"))
                    if self.dense_captioner.training_captioning:
                        # save captions and the captions' idx for computing evaluation metrics (only the first caps_count captions are evaluated)
                        all_captions.append(
                            (
                                captions.to("cpu"),
                                caps_count.to("cpu"),
                                gt_caps_count.to("cpu"),
                            )
                        )
                        all_caps_ids.append(cidxs)
                    if self.dense_captioner.training_sem_enc:
                        # save semantic encodings for computing evaluation metrics
                        all_sem_enc.append(caps_sem_enc_logits.to("cpu"))
                        all_gt_sem_enc.append(gt_caps_sem_enc.to("cpu"))
                    if self.dense_captioner.training_syn_enc:
                        # save syntatic taggings for computing evaluation metrics
                        all_syn_enc.append(
                            (
                                caps_pos_tags.to("cpu"),
                                caps_count.to("cpu"),
                                gt_caps_count.to("cpu"),
                            )
                        )

                    #     # save programs and the videos' idx for computing evaluation metrics
                    #     all_programs.append(program.to("cpu"))
                    #     all_prog_ids.append(vidx)

                    #     # save intervals for computing evaluation metrics
                    #     all_intervals.append(intervals.to("cpu"))
                    #     all_tstamps.append(tstamps / (fps.unsqueeze(1) ** 2))

                    #     # for predicted_tokens, vid in zip(outputs, video_ids):
                    #     #     predicted_sentences[vid] = [self.__decode_from_tokens(predicted_tokens)]

                    #     # logging sample sentences of prediction and target
                    #     # self.logger.info('[vid:{}]'.format(video_ids[0]))
                    #     # self.logger.info('\nWE: {}\nGT: {}'.format(predicted_sentences[video_ids[0]],
                    #     #                                            self.__decode_from_tokens(captions[0].squeeze())))

                    # if phase != "train":
                    #     # save programs and the videos' idx for computing evaluation metrics
                    #     # all_programs.append(program.to("cpu"))
                    #     all_prog_ids.append(vidx)
                    #     all_f_counts.append(feats_count)

                    #     # save intervals for computing evaluation metrics
                    #     all_tstamps.append((intervals * 16 / fps.unsqueeze(1).unsqueeze(1)).to("cpu"))

                    #     all_cap_counts.append(caps_count.to("cpu"))

                # for sanity, replace the last checkpoint when epoch is power of 2
                if phase == "train" and (
                    self.last_saved_epoch == -1 or not (epoch & (epoch - 1))
                ):
                    print("saving checkpoint...")
                    self.__save_checkpoint(epoch, save_checkpoints_dir)

                avg_loss = loss_count / len(self.loaders[phase])
                loss_phase[phase] = avg_loss
                self.writer.add_scalar(
                    f"end2end/{phase}-epochs-avg-loss", avg_loss, epoch
                )

                if avg_loss < self.best_loss_phase[phase]:
                    self.best_loss_phase[phase] = avg_loss

                if phase != "train":
                    self.early_stop_count += 1

                if self.dense_captioner.training_proposals:
                    s_prop_avg_loss = s_prop_loss_count / len(self.loaders[phase])
                    s_prop_loss_phase[phase] = s_prop_avg_loss
                    if s_prop_avg_loss < self.best_s_prop_loss_phase[phase]:
                        self.best_s_prop_loss_phase[phase] = s_prop_avg_loss
                    self.writer.add_scalar(
                        f"proposals/{phase}-epochs-s_prop-avg-loss",
                        s_prop_avg_loss,
                        epoch,
                    )

                    e_prop_avg_loss = e_prop_loss_count / len(self.loaders[phase])
                    e_prop_loss_phase[phase] = e_prop_avg_loss
                    if e_prop_avg_loss < self.best_e_prop_loss_phase[phase]:
                        self.best_e_prop_loss_phase[phase] = e_prop_avg_loss
                    self.writer.add_scalar(
                        f"proposals/{phase}-epochs-e_prop-avg-loss",
                        e_prop_avg_loss,
                        epoch,
                    )

                    print("evaluating s_proposals...")
                    s_prop_metrics_results = multilabel_evaluate_from_logits(
                        all_gt_props_s, all_props_s, all_cap_counts
                    )
                    s_prediction = self.parse_proposals_prediction(
                        all_vidxs, all_gt_props_s, all_props_s, all_cap_counts
                    )
                    self.__process_results(
                        s_prop_metrics_results,
                        s_prediction,
                        phase,
                        epoch,
                        save_checkpoints_dir,
                        "s_prop",
                    )

                    print("evaluating e_proposals...")
                    e_prop_metrics_results = multilabel_evaluate_from_logits(
                        all_gt_props_e, all_props_e, all_cap_counts
                    )
                    e_prediction = self.parse_proposals_prediction(
                        all_vidxs, all_gt_props_e, all_props_e, all_cap_counts
                    )
                    self.__process_results(
                        e_prop_metrics_results,
                        e_prediction,
                        phase,
                        epoch,
                        save_checkpoints_dir,
                        "e_prop",
                    )
                if self.dense_captioner.training_programmer:
                    prog_avg_loss = prog_loss_count / len(self.loaders[phase])
                    prog_loss_phase[phase] = prog_avg_loss
                    if prog_avg_loss < self.best_prog_loss_phase[phase]:
                        self.best_prog_loss_phase[phase] = prog_avg_loss
                    self.writer.add_scalar(
                        f"programmer/{phase}-epochs-avg-loss", prog_avg_loss, epoch
                    )

                    print("evaluating programs...")
                    prog_metrics_results = multiclass_evaluate_from_logits(
                        all_gt_programs, all_programs, all_prog_lens
                    )
                    self.__process_results(
                        prog_metrics_results,
                        None,
                        phase,
                        epoch,
                        save_checkpoints_dir,
                        "programmer",
                    )
                if self.dense_captioner.training_captioning:
                    avg_cap_loss = cap_loss_count / len(self.loaders[phase])
                    cap_loss_phase[phase] = avg_cap_loss
                    if avg_cap_loss < self.best_cap_loss_phase[phase]:
                        self.best_cap_loss_phase[phase] = avg_cap_loss
                    self.writer.add_scalar(
                        "captioning/{}-epochs-avg-cap_loss".format(phase),
                        avg_cap_loss,
                        epoch,
                    )

                    print("evaluating captions (basic)...")
                    self.cap_metrics_results, pred_caps = evaluate_from_tokens(
                        self.caps_vocab,
                        all_captions,
                        all_caps_ids,
                        self.ref_captions[phase],
                    )
                    self.__process_results(
                        self.cap_metrics_results,
                        pred_caps,
                        phase,
                        epoch,
                        save_checkpoints_dir,
                        "captioning",
                    )

                    print("evaluating captions (dense)...")
                    (
                        self.densecap_metrics_results,
                        pred_intervals,
                    ) = densecap_evaluate_from_tokens(
                        self.caps_vocab,
                        all_prog_ids,
                        all_tstamps,
                        all_captions,
                        self.ref_densecaps[phase],
                    )
                    self.__process_results(
                        self.densecap_metrics_results,
                        pred_intervals,
                        phase,
                        epoch,
                        save_checkpoints_dir,
                        "densecap",
                    )
                if self.dense_captioner.training_sem_enc:
                    avg_sem_loss = sem_loss_count / len(self.loaders[phase])
                    sem_loss_phase[phase] = avg_sem_loss
                    if avg_sem_loss < self.best_sem_loss_phase[phase]:
                        self.best_sem_loss_phase[phase] = avg_sem_loss
                    self.writer.add_scalar(
                        "captioning/{}-epochs-avg-sem_loss".format(phase),
                        avg_sem_loss,
                        epoch,
                    )

                    print("evaluating semanctic tagging...")
                    self.sem_enc_metrics_results = multilabel_evaluate_from_logits(
                        all_gt_sem_enc, all_sem_enc, all_cap_counts
                    )
                    self.__process_results(
                        self.sem_enc_metrics_results,
                        None,
                        phase,
                        epoch,
                        save_checkpoints_dir,
                        "sem_enc",
                    )
                if self.dense_captioner.training_syn_enc:
                    avg_pos_loss = pos_loss_count / len(self.loaders[phase])
                    pos_loss_phase[phase] = avg_pos_loss
                    if avg_pos_loss < self.best_pos_loss_phase[phase]:
                        self.best_pos_loss_phase[phase] = avg_pos_loss
                    self.writer.add_scalar(
                        "captioning/{}-epochs-avg-pos_loss".format(phase),
                        avg_pos_loss,
                        epoch,
                    )

                    print("evaluating syntactic tagging...")
                    self.syn_enc_metrics_results, pred_pos_tags = evaluate_from_tokens(
                        self.pos_vocab,
                        all_syn_enc,
                        all_caps_ids,
                        self.ref_pos[phase],
                        True,
                        ".",
                    )
                    self.__process_results(
                        self.syn_enc_metrics_results,
                        pred_pos_tags,
                        phase,
                        epoch,
                        save_checkpoints_dir,
                        "syn_enc",
                    )

                # freeze modules without improvement in validation loss or metrics
                loss_early_stop = self.freeze_modules(epoch=epoch, phase=phase)

                # report results if any improvement occurs
                if self.early_stop_count == 0:
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
                        [
                            f"{k}:({e:03d}, {v:.3f})"
                            for k, (e, v) in self.best_metrics["s_prop"][phase].items()
                        ]
                    )
                    log_msg += f"\n  E-Proposals metrics {phase}: \n   "
                    log_msg += "\t".join(
                        [
                            f"{k}:({e:03d}, {v:.3f})"
                            for k, (e, v) in self.best_metrics["e_prop"][phase].items()
                        ]
                    )

                    print(log_msg, "\n")
                    self.logger.info(log_msg)

                #     # prog_metrics_results = parallel_pool.apply_async(evaluate_from_tokens, [self.programs_vocab, all_programs, all_prog_ids, self.ref_programs[phase], False])
                #     # cap_metrics_results = parallel_pool.apply_async(evaluate_from_tokens, [self.caps_vocab, all_captions, all_caps_ids, self.ref_captions[phase]])
                #     # densecap_metrics_results = parallel_pool.apply_async(densecap_evaluate_from_tokens, [self.caps_vocab, all_intervals, all_captions, all_caps_ids, self.ref_densecaps[phase]])

                time_phase[phase] += time.perf_counter() - time_start_epoch

            # logging message
            log_msg = "\n"
            for k, v in loss_phase.items():
                log_msg += " {0}-avg-loss:{1:3.4f}".format(k, v)
            for k, v in time_phase.items():
                log_msg += " {0}-avg-time:{1:3.3f}h".format(k, (v / 3600) / (epoch + 1))
            log_msg += f" epochs:{self.trained_epochs} tf_ps:{tf_ratios} lrs:{lrs} freezed:{self.freezed_modules}"

            self.logger.info(log_msg)
            sys.stdout.write(log_msg + "\n")

            # check if the training must be early sopped
            if loss_early_stop or (
                epoch >= min_num_epochs
                and self.early_stop_count >= early_stop_limit * 2
            ):
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

                if not loss_early_stop:
                    msg = f"----early stopped at epoch {epoch} after {early_stop_limit} without any improvement on metrics-----"
                else:
                    msg = f"----early stopped at epoch {epoch} after all compnents were freezed-----"
                self.logger.info(msg)
                print(msg)
                break

            self.lr_scheduler.step()

        # close h5 files
        # self.h5_train.close()
        # self.h5_val.close()
        for phase in ["train"] + val_phases:
            self.loaders[phase].dataset.close_h5_file()

        # log best results
        self.logger.info("Best results: {}".format(str(self.best_metrics)))

        return self.best_metrics


def main(dataset_folder, output_folder, checkpoint_path=""):
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    # Set a specific seed value
    seed_value = 42
    set_seed(seed_value)

    # load hiper-parameters
    print("Loading configuration file...")
    config_path = os.path.join(dataset_folder, "train_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    trainer_config = ConfigDict(config["trainer_config"])
    trainer_config.str = get_trainer_str(trainer_config)
    print(trainer_config.str)

    dense_captioner_config = ConfigDict(config["dense_captioner_config"])
    dense_captioner_config.str = get_dense_captioner_str(dense_captioner_config)
    print(dense_captioner_config.str)

    visual_enc_config = ConfigDict(config["visual_enc_config"])
    visual_enc_config.str = get_visual_enc_str(visual_enc_config)
    print(visual_enc_config.str)

    sem_tagger_config = ConfigDict(config["sem_tagger_config"])
    sem_tagger_config.str = get_sem_tagger_str(sem_tagger_config)
    print(sem_tagger_config.str)

    syn_tagger_config = ConfigDict(config["syn_tagger_config"])
    syn_tagger_config.str = get_syn_tagger_str(syn_tagger_config)
    print(syn_tagger_config.str)

    ensemble_dec_config = ConfigDict(config["ensemble_dec_config"])
    ensemble_dec_config.str = get_ensemble_decoder_str(ensemble_dec_config)
    print(ensemble_dec_config.str)

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
        "visual_enc_config": visual_enc_config,
        "sem_tagger_config": sem_tagger_config,
        "syn_tagger_config": syn_tagger_config,
        "ensemble_dec_config": ensemble_dec_config,
        "avscn_dec_config": avscn_dec_config,
        "semsynan_dec_config": semsynan_dec_config,
        "mm_config": mm_config,
        "vncl_cell_config": vncl_cell_config,
        "proposals_tagger_config": proposals_tagger_config,
    }
    # modules_config = [sem_tagger_config, syn_embedd_config, avscn_dec_config, semsynan_dec_config, vncl_cell_config]
    trainer = DenseVideo2TextTrainer(
        trainer_config, modules_config, dataset_folder, output_folder
    )

    print("Training.........")
    # try:
    best_results = trainer.train_model(
        resume=checkpoint_path != "",
        checkpoint_path=checkpoint_path,
        early_stop_limit=trainer_config.early_stop_limit,
    )
    print("Best results in the test set: {}".format(str(best_results)))
    # except Exception as e:
    #     print(f'An error occurred during training/validation process: {e}')
    #     trainer.h5_train.close()
    #     trainer.h5_val.close()

    print("--- END ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model for dense video captioning"
    )
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

    main(args.dataset_folder, args.output_folder, args.checkpoint_path)
