import sys
import os
import re

import torch
import torch.nn as nn
import numpy as np

sys.path.append("video_description_eval/coco-caption")
from video_description_eval.evaluate import score
from video_description_eval.densecap_eval import densecap_score


def get_freer_gpu():
    if os.name == "posix":
        os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
        memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    else:
        import subprocess
        subprocess.run(["powershell", "-Command", "nvidia-smi -q -d Memory | Select-String -Pattern GPU -Context 0,4 > tmp2"], capture_output=True)
        subprocess.run(["powershell", "-Command", "cat tmp2 | Select-String -Pattern Free -Context 0,0  > tmp"], capture_output=True)
        memory_available = [int(x.split()[2]) for x in open("tmp", "r", encoding="utf-16").readlines() if x != "\n"]
    return np.argmax(memory_available)


def get_gpu_temps(device=None):
    if os.name == "posix":
        os.system("nvidia-smi -q -d Temperature |grep -A4 GPU|grep 'GPU Current Temp' >tmp")
        temps = [int(x.split()[-2]) for x in open("tmp", "r").readlines()]
    else:
        import subprocess
        subprocess.run(["powershell", "-Command", "nvidia-smi -q -d Temperature | Select-String -Pattern GPU -Context 0,4 > tmp2"], capture_output=True)
        subprocess.run(["powershell", "-Command", "cat tmp2 | Select-String -Pattern 'GPU Current Temp' -Context 0,0  > tmp"], capture_output=True)
        temps = [int(x.split()[-2]) for x in open("tmp", "r", encoding="utf-16").readlines() if x != "\n"]
    return temps if device is None else (temps[device.index] if "cuda" == device.type else temps[0])


def decode_from_tokens(vocab, tokens, until_eos=True, max_length=10000):
    words = []
    for token in tokens[:max_length]:
        if until_eos and token.item() == vocab("<eos>"):
            break
        words.append(vocab.idx_to_word(token.item()))
    result = " ".join(words)
    
    # replace unnecessary spaces before punctation
    result = re.sub(r"\s([.,;:?!'\"](?:|$))", r"\1", result)

    return result


def get_sentences(vocab, outputs, gt_idxs, until_eos=True):
    pred_sentences = {}
    for batch_outs, batch_gt_idxs in zip(outputs, gt_idxs):
        if torch.is_tensor(batch_outs):
            for pred_tokens, gt_idx in zip(batch_outs, batch_gt_idxs):
                # print('tensor case', pred_tokens.size())
                pred_sentences[gt_idx.item()] = [decode_from_tokens(vocab, pred_tokens, until_eos)]
        elif type(batch_outs) is tuple:
            for v_output, v_caps_count, v_gt_caps_count, v_cidxs in zip(
                batch_outs[0], batch_outs[1], batch_outs[2], batch_gt_idxs
            ):
                count = min(v_caps_count, v_gt_caps_count)
                for pred_tokens, cidx in zip(v_output[:count], v_cidxs[:count]):
                    # print('tuple case', pred_tokens.size())
                    pred_sentences[cidx.item()] = [decode_from_tokens(vocab, pred_tokens, until_eos)]
        else:
            raise TypeError(f"wrong type {type(batch_outs)} for batch outputs")
    return pred_sentences


def get_scores(pred_sentences, ground_truth):
    scores = score(ground_truth, pred_sentences)
    weights = {
        "Bleu_1": 0.0,
        "Bleu_2": 0.0,
        "Bleu_3": 0.0,
        "Bleu_4": 1.4,
        "CIDEr": 1.17,
        "METEOR": 2.0,
        "ROUGE_L": 1.0,
    }
    scores["All_Metrics"] = sum([scores[k] * weights[k] for k in scores.keys()])
    return scores


def evaluate_from_tokens(vocab, outputs, gt_idxs, ground_truth, until_eos=True):
    pred_sentences = get_sentences(vocab, outputs, gt_idxs, until_eos)

    # sanity
    for idx in ground_truth.keys():
        if idx not in pred_sentences:
            pred_sentences[idx] = [""]

    metrics_results = get_scores(pred_sentences, ground_truth)
    return metrics_results, pred_sentences


def densecap_evaluate_from_tokens(vocab, vidxs, pred_tstamps, pred_caps, ground_truth_dict):
    prediction = {}
    for batch_pred_caps, batch_vidxs, batch_tstamps in zip(
        pred_caps, vidxs, pred_tstamps
    ):
        for v_caps, v_caps_count, vidx, v_tstamps in zip(
            batch_pred_caps[0], batch_pred_caps[1], batch_vidxs, batch_tstamps
        ):
            if v_caps_count > 0:
                prediction[str(vidx.item())] = [
                    {
                        "sentence": decode_from_tokens(vocab, pred_tokens),
                        "timestamp": [ts[0].item(), ts[1].item()],
                    }
                    for ts, pred_tokens in zip(v_tstamps[:v_caps_count], v_caps[:v_caps_count])
                ]

    scores = densecap_score(
        args={"tiou": [0.3, 0.5, 0.7, 0.9], "max_proposals_per_video": 1000, "verbose": True,},
        ref=ground_truth_dict,
        hypo=prediction,
    )
    weights = {
        "Bleu_1": 0.0,
        "Bleu_2": 0.0,
        "Bleu_3": 0.0,
        "Bleu_4": 1.4,
        "CIDEr": 1.17,
        "METEOR": 2.0,
        "ROUGE_L": 1.0,
        "Recall": 1.0,
        "Precision": 1.0,
    }
    scores["All_Metrics"] = sum([scores[k] * weights[k] for k in scores.keys() if k in weights])
    return scores, prediction


def evaluate_from_sentences(pred_sentences, ground_truth):
    metrics_results = get_scores(pred_sentences, ground_truth)
    return metrics_results


def load_ground_truth_captions(reference_txt_path):
    gt = {}
    for line in list(open(reference_txt_path, "r")):
        row = line.split("\t")
        idx = int(row[0])
        sentence = row[1].strip()
        if idx in gt:
            gt[idx].append(sentence)
        else:
            gt[idx] = [sentence]
    return gt


def load_texts(path, blacklist=[]):
    result_dict = {}
    for line in list(open(path, "r", encoding="utf8")):
        row = line.split("\t")
        idx, sentence = int(row[0]), row[1].strip()

        # skip idxs in the blacklist
        if idx in blacklist:
            continue

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


def bow_vectors(caps, vocab_len, norm=False, eos=0):
    vecs = torch.zeros(caps.size(0), vocab_len).to(caps.device)

    for i, cap in enumerate(caps):
        for widx in cap:
            if widx != eos:
                vecs[i, widx] += 1

    if not norm:
        return vecs
    else:
        return vecs / torch.sum(vecs, dim=1)


def get_tf_ratio(tf_config, epoch):
    # determine teacher_forcing_ratio according to the convergence_speed_factor and current epoch
    if tf_config.tf_strategy == "classic":
        k = tf_config.classic_config.speed_factor
        min_ratio = tf_config.classic_config.min_ratio

        # inverse sigmoid decay
        return max(min_ratio, k / (k + np.exp(epoch / k)))
    return 0


def get_trainer_str(config):
    crit_config = config.criterion_config
    return (
        f"{config.dataset_name} batch-{config.batch_size}.lr-{config.optimizer_config.learning_rate}.{config.optimizer_config.optimizer_name}"
        f".closs-{crit_config.captioning_loss}-{crit_config.captioning_loss_reduction}"
        f".ploss-{crit_config.programer_loss}-{crit_config.programer_loss_reduction}"
        f".tagloss-{crit_config.tagging_loss}-{crit_config.tagging_loss_reduction}"
        f".iloss-{crit_config.intervals_loss}-{crit_config.intervals_loss_reduction}"
    )



def get_tf_strategy_str(config):
    result = f".tf-{config.tf_strategy}"
    if config.tf_strategy == "classic":
        result += f"-{config.classic_config.speed_factor}-{config.classic_config.min_ratio}"
    else:
        pass
    return result


def get_dense_captioner_str(config):
    hs = str([config.h_size])
    train_sample = "max" if config.train_sample_max else "dist"
    test_sample = "max" if config.test_sample_max else "dist"
    return f"programmer max_clip_len-{config.max_clip_len}.future_steps-{config.future_steps}.hs-{hs}.drop-{config.drop_p}.train-{train_sample}.test-{test_sample}"


def get_visual_enc_str(config):
    drops = str([config.mapping_in_drop_p] + config.mapping_h_drop_ps)
    hs = str(config.mapping_h_sizes)
    return f"v-enc hs-{hs}.out-{config.out_size}.drops-{drops}.lastbn-{config.have_last_bn}"


def get_sem_tagger_str(config):
    drops = str([config.mapping_in_drop_p] + config.mapping_h_drop_ps)
    hs = str(config.mapping_h_sizes)
    return f"sem hs-{hs}.out-{config.out_size}.drops-{drops}.lastbn-{config.have_last_bn}"


def get_syn_tagger_str(config):
    c = config.enc_config
    hs = str(c.h_sizes)
    in_size = c.cnn_feats_size + c.c3d_feats_size
    enc = f"syn-enc in-{in_size}.hs-{hs}.out-{c.out_size}.drop-{c.drop_p}.lastbn-{c.have_last_bn}.norm-{c.norm}"

    c = config.dec_config
    hs = str([c.h_size] + [c.rnn_h_size])
    train_sample = "max" if c.train_sample_max else "dist"
    test_sample = "max" if c.test_sample_max else "dist"
    dec = f"syn-dec in-{c.in_seq_length}.posemb-{c.embedding_size}.rnnin-{c.rnn_in_size}.hs-{hs}. drop-{c.drop_p}.layers-{c.num_layers}.train-{train_sample}.test-{test_sample}"

    return f"{enc} {dec}"


def get_avscn_decoder_str(config):
    hs = str([config.h_size] + [config.rnn_h_size])
    train_sample = "max" if config.train_sample_max else "dist"
    test_sample = "max" if config.test_sample_max else "dist"
    return f"avscn-dec in-{config.in_seq_length}.rnnin-{config.rnn_in_size}.hs-{hs}.drop-{config.drop_p}.layers-{config.num_layers}.train-{train_sample}.test-{test_sample}"


def get_semsynan_decoder_str(config):
    hs = str([config.h_size] + [config.rnn_h_size])
    train_sample = "max" if config.train_sample_max else "dist"
    test_sample = "max" if config.test_sample_max else "dist"
    return f"semsynan-dec in-{config.in_seq_length}.posemb-{config.syn_enc_size}.rnnin-{config.rnn_in_size}.hs-{hs}. drop-{config.drop_p}.layers-{config.num_layers}.train-{train_sample}.test-{test_sample}"


def get_mm_str(config):
    return f"mm out-{config.out_size}.v-norm-{config.v_enc_config.norm}.t-norm-{config.t_enc_config.norm}"


def get_vncl_cell_str(config):
    return f"vncl-cell mm-{config.mm_size}.vh-{config.vh_size}.h1-{config.h1_size}"


def get_proposals_tagger_str(config):
    drops = str([config.in_drop_p] + config.drop_ps)
    hs = str(config.h_sizes)
    return f"sem hs-{hs}.drops-{drops}.lastbn-{config.have_last_bn}"

