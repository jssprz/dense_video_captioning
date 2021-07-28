import torch
import torch.nn as nn


class SentenceLengthLoss(nn.Module):
    def __init__(self, max_words, class_weights, beta, reduction):
        super(SentenceLengthLoss, self).__init__()
        self.max_words = max_words
        self.beta = beta
        self.reduction = reduction
        if class_weights is not None:
            self.crit = nn.NLLLoss(weight=class_weights, reduction="sum")
        else:
            self.crit = nn.NLLLoss(reduction="sum")

    def forward(self, logits, targets, lens=None, rewards=None):
        if (lens is None) and (rewards is None):
            log_probs = torch.log_softmax(logits, dim=1)
            loss = self.crit(log_probs, targets)
            if self.reduction == "mean":
                return loss / logits.size(0)
        elif rewards is None:
            mask = torch.cat([l.repeat(l) for l in lens], dim=0).unsqueeze(1).to(logits.device)
            wighted_log_probs = torch.reciprocal(mask ** self.beta) * torch.log_softmax(logits, dim=1)
            loss = self.crit(wighted_log_probs, targets)
            if self.reduction == "mean":
                return loss / lens.size(0)
        else:
            mask1 = torch.cat([l.repeat(l) for l in lens], dim=0).unsqueeze(1).to(logits.device)
            mask2 = torch.cat([r.repeat(l) for l, r in zip(lens, rewards)], dim=0).unsqueeze(1).to(logits.device)
            # mask = torch.max(torch.ones_like(mask1), mask1**self.beta - mask2**self.beta)
            wighted_log_probs = torch.reciprocal(mask1 ** self.beta) * mask2 * torch.log_softmax(logits, dim=1)
            loss = self.crit(wighted_log_probs, targets)
            if self.reduction == "mean":
                return loss / lens.size(0)

        return loss  # self.reduction=='sum'


def temp_iou(pred_intervals, gt_intervals, gt_count):
    # compute intersection
    max_min = torch.max(torch.stack([pred_intervals[:, :, 0], gt_intervals[:, :, 0]]), dim=0)[0]
    min_max = torch.min(torch.stack([pred_intervals[:, :, 1], gt_intervals[:, :, 1]]), dim=0)[0]
    intersection = (min_max - max_min).clamp(min=0)
    # print('intersec:', intersection)

    # compute union
    min_min = torch.min(torch.stack([pred_intervals[:, :, 0], gt_intervals[:, :, 0]]), dim=0)[0]
    max_max = torch.max(torch.stack([pred_intervals[:, :, 1], gt_intervals[:, :, 1]]), dim=0)[0]
    union = max_max - min_min
    # print('union:', union)

    # compute total IoU
    return torch.FloatTensor(
        [
            (torch.reciprocal(union[n, : gt_count[n]]) * intersection[n, : gt_count[n]]).mean() if gt_count[n] else 0
            for n in range(gt_count.size(0))
        ]
    )


class DenseCaptioningLoss(nn.Module):
    def __init__(self, config, c_max_len, p_max_len, sem_enc_pos_weights, device):
        super(DenseCaptioningLoss, self).__init__()

        self.config = config

        # captioning_loss function
        if config.captioning_loss == "MSE":
            self.captioning_loss = nn.MSELoss(reduction=config.captioning_loss_reduction)
            self.pos_tag_loss = nn.MSELoss(reduction=config.captioning_loss_reduction)
        elif config.captioning_loss == "NLL":
            self.captioning_loss = nn.NLLLoss(reduction=config.captioning_loss_reduction)
            self.pos_tag_loss = nn.NLLLoss(reduction=config.captioning_loss_reduction)
        elif config.captioning_loss == "LenW":
            self.captioning_loss = SentenceLengthLoss(
                c_max_len,
                class_weights=None,
                beta=config.captioning_loss_b,
                reduction=config.captioning_loss_reduction,
            )
            self.pos_tag_loss = SentenceLengthLoss(
                c_max_len,
                class_weights=None,
                beta=config.captioning_loss_b,
                reduction=config.captioning_loss_reduction,
            )
        elif config.captioning_loss == "XEnt":
            self.captioning_loss = nn.CrossEntropyLoss(reduction=config.captioning_loss_reduction)
            self.pos_tag_loss = nn.CrossEntropyLoss(reduction=config.captioning_loss_reduction)
        else:
            raise ValueError(
                f"wrong value '{config.captioning_loss}' for the captioning_loss option in Loss configuration"
            )

        # programer_loss function
        if config.programer_loss == "MSE":
            self.programer_loss = nn.MSELoss(reduction=config.programer_loss_reduction)
        elif config.programer_loss == "NLL":
            self.programer_loss = nn.NLLLoss(reduction=config.programer_loss_reduction)
        elif config.programer_loss == "LenW":
            class_weights = torch.tensor(config.programer_loss_weights).to(device)
            self.programer_loss = SentenceLengthLoss(
                p_max_len,
                class_weights=class_weights,
                beta=config.programer_loss_b,
                reduction=config.programer_loss_reduction,
            )
        elif config.programer_loss == "XEnt":
            class_weights = torch.tensor(config.programer_loss_weights).to(device)
            self.programer_loss = nn.CrossEntropyLoss(weight=class_weights, reduction=config.programer_loss_reduction)
        else:
            raise ValueError(
                f"wrong value '{config.programer_loss}' for the programer_loss option in Loss configuration"
            )

        # semantic_tagging_loss function
        if config.tagging_loss == "BXE":
            self.tagging_loss = nn.BCEWithLogitsLoss(pos_weight=sem_enc_pos_weights, reduction=config.tagging_loss_reduction)
        else:
            raise ValueError(f"wrong value '{config.tagging_loss}' for the tagging_loss option in Loss configuration")

        # proposals_loss function
        if config.proposals_loss == "BXE":
            # self.proposals_loss = nn.BCELoss(reduction=config.proposals_loss_reduction)
            self.proposals_loss = nn.BCEWithLogitsLoss(reduction=config.proposals_loss_reduction)
        else:
            raise ValueError(
                f"wrong value '{config.proposals_loss}' for the proposals_loss option in Loss configuration"
            )

        # multimodal_loss function
        # if mmloss == 'MSE':
        #     self.multimodal_loss = nn.MSELoss(reduction=config.mmloss_reduction)

        if config.learn_comb_weights:
            self.comb_weights = nn.Parameter(torch.Tensor(config.comb_weights))
        else:
            self.comb_weights = torch.tensor(config.comb_weights)

    def forward(
        self,
        gt_captions,
        gt_cap_lens,
        pred_captions,
        gt_caps_sem_enc,
        pred_caps_sem_enc,
        gt_pos_seq,
        pred_pos_seq,
        gt_program,
        gt_prog_len,
        pred_program,
        gt_intervals,
        pred_intervals,
        gt_proposals,
        pred_proposals,
        gt_caps_count,
        pred_caps_count,
        gt_proposals_count,
        truncate_prog_at=None,
        mm_v_encs=None,
        mm_t_encs=None,
    ):
        bs, _, _, caps_vocab_size = pred_captions.size()
        pos_vocab_size = pred_pos_seq.size(3)

        # TODO: compute gt flatten in the Dataset for removing it from here

        l1, l2, l3, l4, l5, l6, = [], [], [], [], [], []
        for n in range(bs):
            for i in range(gt_caps_count[n]):
                l1.append(pred_captions[n, i, : gt_cap_lens[n, i]].reshape(-1, caps_vocab_size))
                l2.append(gt_captions[n, i, : gt_cap_lens[n, i]].flatten())

                l3.append(pred_pos_seq[n, i, : gt_cap_lens[n, i]].reshape(-1, pos_vocab_size))
                l4.append(gt_pos_seq[n, i, : gt_cap_lens[n, i]].flatten())

            l5.append(pred_caps_sem_enc[n, : gt_caps_count[n], :])
            l6.append(gt_caps_sem_enc[n, : gt_caps_count[n], :])

        # captioning loss
        pred_captions = torch.cat(l1)
        gt_captions = torch.cat(l2)
        # cap_loss = self.captioning_loss(pred_captions, gt_captions, gt_cap_lens)  # length-weighted
        cap_loss = self.captioning_loss(pred_captions, gt_captions)  # CELoss

        # pos-tagging loss
        pred_pos_seq = torch.cat(l3)
        gt_pos_seq = torch.cat(l4)
        # cap_loss = self.pos_tag_loss(pred_pos_seq, gt_pos_seq, gt_cap_lens)  # length-weighted
        pos_loss = self.pos_tag_loss(pred_pos_seq, gt_pos_seq)  # CELoss

        # semantic tagging loss
        pred_caps_sem_enc = torch.cat(l5)
        gt_caps_sem_enc = torch.cat(l6)
        sem_enc_loss = self.tagging_loss(pred_caps_sem_enc, gt_caps_sem_enc)

        loss = cap_loss + sem_enc_loss + pos_loss

        return (
            loss,
            None, # prog_loss,
            cap_loss,
            sem_enc_loss,
            pos_loss,
            None, # proposals_loss,
            None, # iou_reward.mean(),
        )
