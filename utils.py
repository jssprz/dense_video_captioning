import sys

import torch
import torch.nn as nn

sys.path.append('video_description_eval/coco-caption')
from video_description_eval.evaluate import score


def decode_from_tokens(tokens, vocab):
  words = []
  for token in tokens:
    if token.item() == vocab('<eos>'):
        break
    words.append(vocab.idx_to_word(token.item()))
  return ' '.join(words)


def get_sentences(vocab, outputs, idxs):
    pred_sentences = {}
    for batch_outs, batch_idxs in zip(outputs, idxs):
        for pred_tokens, idx in zip(batch_outs, batch_idxs):
            pred_sentences[idx] = [decode_from_tokens(vocab, pred_tokens)]
    return pred_sentences


def get_scores(pred_sentences, ground_truth):
    scores = score(ground_truth, pred_sentences)
    weights = {'Bleu_1':0., 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 1.4, 'CIDEr': 1.17, 'METEOR': 2., 'ROUGE_L':1.}
    scores['All_Metrics'] = sum([scores[k] * weights[k] for k in scores.keys()])
    return scores            


def evaluate_from_tokens(vocab, outputs, idxs, ground_truth):
    pred_sentences = get_sentences(vocab, outputs, idxs)
    metrics_results = get_scores(pred_sentences, ground_truth)
    return metrics_results, predicted_sentences


def evaluate_from_sentences(pred_sentences, ground_truth):
    metrics_results = get_scores(pred_sentences, ground_truth)
    return metrics_results


def load_ground_truth_captions(reference_txt_path):
    gt = {}
    for line in list(open(reference_txt_path, 'r')):
        row = line.split('\t')
        idx = int(row[0])
        sentence = row[1].strip()
        if idx in gt:
            gt[idx].append(sentence)
        else:
            gt[idx] = [sentence]
    return gt


def load_texts(path):
  result_dict = {}
  for line in list(open(path, 'r')):
    row = line.split('\t')
    idx = int(row[0])
    sentence = row[1].strip()
    if idx in result_dict:
      result_dict[idx].append(sentence)
    else:
      result_dict[idx] = [sentence]
  return result_dict


def get_init_weights(shape):
    W = torch.rand(shape)
    nn.init.xavier_normal_(W.data)
    return nn.Parameter(W)


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def make_bow_vector(cap, vocab_len, norm=False, eos=0):
    vec = torch.zeros(vocab_len)
    for widx in [idx for idx in cap if idx != eos]:
        vec[widx] += 1
    
    if not norm:
      return vec
    else:
      return vec / sum(vec)


def get_trainer_str(config):
  return f'{config.dataset_name} batch-{config.batch_size}.lr-{config.learning_rate}.{config.optimizer_name}.closs-{config.criterion_config.closs}-{config.criterion_config.closs_reduction}.ploss-{config.criterion_config.ploss}-{config.criterion_config.ploss_reduction}.iloss-{config.criterion_config.iloss}-{config.criterion_config.iloss_reduction}'


def get_sem_tagger_str(config):
  drops = str([config.in_drop_p] + config.drop_ps)
  hs = str(config.h_sizes)
  return f'sem in-{config.in_size}.hs-{hs}.out-{config.out_size}.drops-{drops}.lastbn-{config.have_last_bn}'


def get_syn_embedd_str(config):
  hs = str(config.v_enc_config.h_sizes)
  in_size = config.v_enc_config.cnn_feats_size+config.v_enc_config.c3d_feats_size
  return f'syn in-{in_size}.hs-{hs}.out-{config.v_enc_config.out_size}.drop-{config.v_enc_config.drop_p}.lastbn-{config.v_enc_config.have_last_bn}.norm-{config.v_enc_config.norm}'


def get_avscn_decoder_str(config):
  hs = str([config.h_size] + [config.rnn_h_size])
  train_sample = 'max' if config.train_sample_max else 'dist'
  test_sample = 'max' if config.test_sample_max else 'dist'
  return f'avscn-dec in-{config.in_seq_length}.rnnin-{config.rnn_in_size}.hs-{hs}.drop-{config.drop_p}.layers-{config.num_layers}.train-{train_sample}.test-{test_sample}'


def get_semsynan_decoder_str(config):
  hs = str([config.h_size] + [config.rnn_h_size])
  train_sample = 'max' if config.train_sample_max else 'dist'
  test_sample = 'max' if config.test_sample_max else 'dist'
  return f'semsynan-dec in-{config.in_seq_length}.posemb-{config.posemb_size}.rnnin-{config.rnn_in_size}.hs-{hs}. drop-{config.drop_p}.layers-{config.num_layers}.train-{train_sample}.test-{test_sample}'


def get_mm_str(config):
  return f'mm out-{config.out_size}.v-norm-{config.v_enc_config.norm}.t-norm-{config.t_enc_config.norm}'


def get_vncl_cell_str(config):
  return f'vncl-cell mm-{config.mm_size}.vh-{config.vh_size}.h1-{config.h1_size}'
