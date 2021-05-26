import os
import argparse
import pickle
import json

import h5py
import torch
import numpy as np

from utils import decode_from_tokens
from vocabulary import Vocabulary
from configuration_dict import ConfigDict
# from loader import extract_split_data_from_corpus, data2tensors, get_dense_loader
from model.dense_captioner import DenseCaptioner


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate captions por test samples')
  parser.add_argument('-chkpt', '--checkpoint_path', type=str, default='pretrain/chckpt.pt',
                      help='Set the path to pre-trained model (default is pretrain/chckpt.pt).')
  parser.add_argument('-data', '--dataset_folder', type=str, default='data/MSVD',
                      help='Set the path to dataset folder (default is data/MSVD).')
  parser.add_argument('-out', '--output_folder', type=str, default='results/MSVD',
                      help='Set the path to output folder (default is results/MSVD).')

  args = parser.parse_args()

  # load vocabularies
  with open(os.path.join(args.dataset_folder, 'dense_corpus2.pkl'), "rb") as f:
      corpus = pickle.load(f)
      idx2op_dict = corpus[4]
      idx2word_dict = corpus[6]
      idx2pos_dict = corpus[8]
      idx2upos_dict = corpus[9]
      test_vidxs = corpus[1][0]
      test_fps = corpus[1][3]
      test_gt_progs = corpus[1][4]

  programs_vocab = Vocabulary.from_idx2word_dict(idx2op_dict, False)
  print('Size of programs_vocab: {}'.format(len(programs_vocab)))

  caps_vocab = Vocabulary.from_idx2word_dict(idx2word_dict, False)
  print('Size of caps_vocab: {}'.format(len(caps_vocab)))

  pos_vocab = Vocabulary.from_idx2word_dict(idx2pos_dict, False)
  print('Size of pos_vocab: {}'.format(len(pos_vocab)))

  upos_vocab = Vocabulary.from_idx2word_dict(idx2upos_dict, False)
  print('Size of upos_vocab: {}'.format(len(upos_vocab)), '\n')

  # Pretrained Embedding
  # pretrained_we = torch.Tensor(corpus[5])
  pretrained_ope = None
  pretrained_we = None
  pretrained_pe = None

  #max_frames = 20 #30
  # cnn_feature_size = 2048
  # c3d_feature_size = 4096
  # i3d_feature_size = 400
  # res_eco_features_size = 3584
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

  # config = ConfigurationFile(os.path.join(args.dataset_folder, 'config.json'), 'attn-vscn-max')

  # load hiper-parameters
  print('Loading configuration file...')
  config_path = os.path.join(args.dataset_folder, 'test_config.json')
  with open(config_path, 'r') as f:
      config = json.load(f)

  tester_config = ConfigDict(config['tester_config'])

  # Checkpoint
  checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
  modules_config = checkpoint['modules_config']
  truncate_at = checkpoint['avg_tuncation']
  max_caps = checkpoint['avg_caps']

  if tester_config.truncate_at < truncate_at:
    print(f' The model is trained for generating {truncate_at} long programs, but {tester_config.truncate_at} long is going to be used')
    truncate_at = tester_config.truncate_at
  elif tester_config.truncate_at >= truncate_at:
    print(f' The model is trained for generating {truncate_at} long programs only. You need to train the model for longer programs')

  if tester_config.max_caps < max_caps:
    print(f' The model is trained for generating until {max_caps} captions, but {tester_config.max_caps} are goint to be generated')
    max_caps = tester_config.max_caps
  elif tester_config.max_caps >= max_caps:
    print(f' The model is trained for generating {max_caps} long programs only. You need to train the model for longer programs')

  # config_path = os.path.join(args.dataset_folder, 'train_config.json')
  # with open(config_path, 'r') as f:
  #     modules_config = json.load(f)

  # Model
  print('\nInitializing the Model...')
  dense_captioner = DenseCaptioner(config=modules_config['dense_captioner_config'],
                                   sem_tagger_config=modules_config['sem_tagger_config'],
                                   syn_embedd_config=modules_config['syn_embedd_config'],
                                   syn_tagger_config=modules_config['syn_tagger_config'],
                                   avscn_dec_config=modules_config['avscn_dec_config'],
                                   semsynan_dec_config=modules_config['semsynan_dec_config'],
                                   mm_config=modules_config['mm_config'],
                                   vncl_cell_config=modules_config['vncl_cell_config'],
                                   proposals_tagger_config=modules_config['proposals_tagger_config'],
                                   num_proposals=tester_config.num_proposals,
                                   progs_vocab=programs_vocab,
                                   pretrained_ope=pretrained_ope,
                                   caps_vocab=caps_vocab,
                                   pretrained_we=pretrained_we,
                                   pos_vocab=pos_vocab,
                                   pretrained_pe=pretrained_pe,
                                   device='cpu')

  # 1. filter out unnecessary keys for encoder
  # chckpt_dict = {k: v for k, v in checkpoint['encoder'].items() if k not in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']}
  # encoder_dict = encoder.state_dict()
  # encoder_dict.update(chckpt_dict)

  # encoder.load_state_dict(encoder_dict)
  # decoder.load_state_dict(checkpoint['decoder'])

  dense_captioner.load_state_dict(checkpoint['dense_captioner'])
  dense_captioner.eval()

  print('Initializing test split data loader...')
  # vidxs, cidxs, intervals, fps, progs, prog_lens, caps, pos, upos, cap_lens = extract_split_data_from_corpus(corpus, split=1)
  # cidxs_t, intervals_t, caps_count_t, progs_t, caps_t, pos_t, upos_t, cap_lens_t = data2tensors(cidxs, intervals, progs, prog_lens, caps, pos, upos, cap_lens,
  #                                                                                               tester_config.max_prog, tester_config.max_caps, tester_config.max_words)

  h5_test = h5py.File(tester_config.h5_file_path, 'r')
  h5_dataset = h5_test[tester_config.h5_file_group_name]
  cnn_feats = h5_dataset['cnn_features']
  c3d_feats = h5_dataset['c3d_features']
  feat_count = h5_dataset['count_features']
  frame_tstamps = h5_dataset['frames_tstamp']
  # test_loader = get_dense_loader(h5_dataset=test_dataset, vidxs=vidxs, vidxs_blcklist=tester_config.blacklist, cidxs=cidxs_t, 
  #                               intervals=intervals_t, caps_count=caps_count_t, captions=caps_t, caps_sem_enc=caps_sem_enc_t, pos=pos_t, 
  #                               upos=upos_t, cap_lens=cap_lens_t, progs=progs_t, prog_lens=prog_lens, batch_size=tester_config.batch_size, train=False)

  with torch.no_grad():
    programs, caps, intervs = [], [], []
    for i, (vidx, fps) in enumerate(zip(test_vidxs, test_fps)):
      video_feats = [torch.from_numpy(cnn_feats[vidx]).unsqueeze(0), torch.from_numpy(c3d_feats[vidx]).unsqueeze(0)]
      feats_count = torch.tensor(feat_count[vidx])
      tstamps = torch.from_numpy(frame_tstamps[vidx]) / (fps**2)
      gt_prog = test_gt_progs[i]

      prog_logits, program, _, _, _, captions, intervals, caps_count, _, _ = dense_captioner(v_feats=video_feats, feats_count=feats_count, prog_len=None, teacher_forcing_p=0,
                                                                                             gt_program=None, gt_captions=None, gt_caps_count=None, gt_sem_enc=None, gt_pos=None, 
                                                                                             gt_intervals=None, max_prog=max_prog, max_caps=max_caps, 
                                                                                             max_cap=tester_config.max_words, max_chunks=max_prog)

      print(f'video {vidx}:')
      p = decode_from_tokens(programs_vocab, program[0], until_eos=False)
      # print(f'  program: {p}')
      print(f' program-probs-avg: {torch.mean(torch.softmax(prog_logits, dim=-1), dim=-2)}')
      programs.append(p)

      print(f' {caps_count.item()} captions predicted:')
      v_caps, v_intervs = [], []
      for i in range(caps_count):
        caption = decode_from_tokens(caps_vocab, captions[0, i], until_eos=True)
        interval = [float(tstamps[int(intervals[0,i,0])]), float(tstamps[int(intervals[0,i,1])])]
        print(f'  {interval}:  {caption}')
        v_caps.append(caption)
        v_intervs.append(interval)
      caps.append(v_caps)
      intervs.append(v_intervs)

  if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

  with open(os.path.join(args.output_folder, 'predictions.txt'), 'w') as fo:
    for vidx, p, cs, ints in zip(test_vidxs, programs, captions, intervals):
      fo.write(f'{vidx}\n')
      fo.write(f'{p}')
      for c, i in zip(cs, ints):
        fo.write(f'{i}\t{s}\n')