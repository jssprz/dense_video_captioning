import torch
import torch.nn as nn


class SentenceLengthLoss(nn.Module):
    def __init__(self, max_words, class_weights, beta, reduction):
        super(SentenceLengthLoss, self).__init__()
        self.max_words = max_words
        self.beta = beta
        self.reduction= reduction
        if class_weights is not None:
            self.crit = nn.NLLLoss(weight=class_weights, reduction='sum')
        else:
            self.crit = nn.NLLLoss(reduction='sum')

    def forward(self, logits, targets, lens=None, rewards=None):
        if (lens is None) and (rewards is None):
            log_probs = torch.log_softmax(logits, dim=1)
            loss = self.crit(log_probs, targets)
            if self.reduction=='mean':
                return loss / logits.size(0)
        elif rewards is None:
            mask = torch.cat([l.repeat(l) for l in lens], dim=0).unsqueeze(1).to(logits.device)
            wighted_log_probs = torch.reciprocal(mask ** self.beta) * torch.log_softmax(logits, dim=1)
            loss = self.crit(wighted_log_probs, targets)
            if self.reduction=='mean':
                return loss / lens.size(0)
        else:
            mask1 = torch.cat([l.repeat(self.max_words) for l in lens], dim=0).unsqueeze(1)
            mask2 = torch.cat([r.repeat(self.max_words) for r in rewards], dim=0).unsqueeze(1)
            wighted_log_probs = torch.reciprocal((mask1 * mask2) ** self.beta) * torch.log_softmax(logits, dim=1)
            loss = self.crit(wighted_log_probs, targets)
            if self.reduction=='mean':
                return loss / lens.size(0)

        return loss  # self.reduction=='sum'


class IoULoss(nn.Module):
    def __init__(self, reduction):
        super(IoULoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred_intervals, gt_intervals):
        # compute intersection
        intersection = torch.zeros(pred_intervals.size(0))
        max_min = torch.max(torch.stack([pred_intervals[:,0], gt_intervals[:,0]]), dim=0)[0]
        min_max = torch.min(torch.stack([pred_intervals[:,1], gt_intervals[:,1]]), dim=0)[0]
        intersection = (min_max - max_min).clamp(min=0)
        # print('intersec:', intersection)

        # compute union
        min_min = torch.min(torch.stack([pred_intervals[:,0], gt_intervals[:,0]]), dim=0)[0]
        max_max = torch.max(torch.stack([pred_intervals[:,1], gt_intervals[:,1]]), dim=0)[0]
        union = max_max - min_min
        # print('union:', union)

        # compute IoU
        loss = torch.sum(torch.reciprocal(union) * intersection)

        if self.reduction=='mean':
            return 1 -  loss / pred_intervals.size(0)
        elif self.reduction=='sum':
            return pred_intervals.size(0) - loss


class DenseCaptioningLoss(nn.Module):
    def __init__(self, config, c_max_len, p_max_len, device):
        super(DenseCaptioningLoss, self).__init__()

        # captioning_loss function
        if config.captioning_loss == 'MSE':
            self.captioning_loss = nn.MSELoss(reduction=config.captioning_loss_reduction)
        elif config.captioning_loss == 'NLL':
            self.captioning_loss = nn.NLLLoss(reduction=config.captioning_loss_reduction)
        elif config.captioning_loss == 'SentLen':
            self.captioning_loss = SentenceLengthLoss(c_max_len, class_weights=None, beta=config.captioning_loss_b, reduction=config.captioning_loss_reduction)
        elif config.captioning_loss == 'XEnt':
            self.captioning_loss = nn.CrossEntropyLoss(reduction=config.captioning_loss_reduction)
        else:
            raise ValueError(f'wrong value \'{config.captioning_loss}\' for the captioning_loss option in Loss configuration')

        # programer_loss function
        if config.programer_loss == 'MSE':
            self.programer_loss = nn.MSELoss(reduction=config.programer_loss_reduction)
        elif config.programer_loss == 'NLL':
            self.programer_loss = nn.NLLLoss(reduction=config.programer_loss_reduction)
        elif config.programer_loss == 'SentLen':
            class_weights = torch.tensor(config.programer_loss_weights).to(device)
            self.programer_loss = SentenceLengthLoss(p_max_len, class_weights=class_weights, beta=config.programer_loss_b, reduction=config.programer_loss_reduction)
        elif config.programer_loss == 'XEnt':
            class_weights = torch.tensor(config.programer_loss_weights).to(device)
            self.programer_loss = nn.CrossEntropyLoss(weight=class_weights, reduction=config.programer_loss_reduction)
        else:
            raise ValueError(f'wrong value \'{config.programer_loss}\' for the programer_loss option in Loss configuration')

        if config.tagging_loss == 'BXEnt':
            self.tagging_loss = nn.BCELoss(reduction=config.tagging_loss_reduction)
        else:
            raise ValueError(f'wrong value \'{config.tagging_loss}\' for the tagging_loss option in Loss configuration')

        # intervals_loss function
        if config.intervals_loss == 'tIoU':
            self.intervals_loss = IoULoss(reduction=config.intervals_loss_reduction)
        else:
            raise ValueError(f'wrong value \'{config.intervals_loss}\' for the intervals_loss option in Loss configuration')

        # multimodal_loss function
        # if mmloss == 'MSE':
        #     self.multimodal_loss = nn.MSELoss(reduction=config.mmloss_reduction)

        if config.learn_comb_weights:
            self.comb_weights = nn.Parameter(torch.Tensor(config.comb_weights))
        else:
            self.comb_weights = torch.tensor(config.comb_weights)

    def forward(self, gt_captions, gt_cap_lens, pred_captions, gt_caps_sem_enc, pred_caps_sem_enc, gt_program, gt_prog_len, pred_program,
                gt_intervals, pred_intervals, gt_caps_count, pred_caps_count, truncate_prog_at=None, mm_v_encs=None, mm_t_encs=None):

        bs, _, _, caps_vocab_size = pred_captions.size()
        progs_vocab_size = pred_captions.size(2)

        # print(pred_captions.requires_grad, pred_intervals.requires_grad)

        #TODO: compute gt strighten in the Dataset for removing it from here

        # straighten the output captions (removing the part of the pad)
        # (total_len_of_captions x caps_vocab_size)
        pred_captions = torch.cat([pred_captions[j, i, :gt_cap_lens[j,i]].reshape(-1, caps_vocab_size) for j in range(bs) for i in range(gt_caps_count[j])], dim=0)

        # straighten the target captions (remove the part of the pad) and then flatten it
        # (total_len_of_captions)
        gt_captions = torch.cat([gt_captions[j, i, :gt_cap_lens[j,i]].flatten() for j in range(bs) for i in range(gt_caps_count[j])], dim=0)

        # straighten the output semantic encodings (removing the part of the pad)
        # (total_num_captions x tags_count)
        pred_caps_sem_enc = torch.cat([pred_caps_sem_enc[j, :gt_caps_count[j], :] for j in range(bs)], dim=0)

        # straighten the target semantic encodings (removing the part of the pad)
        # (total_num_captions x tags_count)
        gt_caps_sem_enc = torch.cat([gt_caps_sem_enc[j, :gt_caps_count[j], :] for j in range(bs)], dim=0)

        # straighten the output intervals (removing the part of the pad)
        # (total_num_captions x 2)
        pred_intervals = torch.cat([pred_intervals[j, :gt_caps_count[j], :] for j in range(bs)], dim=0)

        # straighten the target intervals (removing the part of the pad)
        # (total_num_captions x 2)
        gt_intervals = torch.cat([gt_intervals[j, :gt_caps_count[j], :] for j in range(bs)], dim=0)

        # straighten the lens of target captions (remove the part of the pad) and then flatten it
        # (total_num_captions)
        gt_cap_lens = torch.tensor([gt_cap_lens[j, i] for j in range(bs) for i in range(gt_caps_count[j])], dtype=torch.int32)

        if truncate_prog_at is not None:
            # (bs*truncate_prog_at x progs_vocab_size)
            pred_program = pred_program[:, :truncate_prog_at].reshape(-1, pred_program.size(2))
            # (bs*truncate_prog_at)
            gt_program = gt_program[:, :truncate_prog_at].flatten()
        else:
            # straighten the output program (removing the part of the pad) and then flatten it
            # (total_len_of_programs x progs_vocab_size)
            pred_program = torch.cat([pred_program[j, :gt_prog_len[j]] for j in range(bs)], dim=0)

            # straighten the target captions (remove the part of the pad) and then flatten it
            # (total_len_of_programs)
            gt_program = torch.cat([gt_program[j, :gt_prog_len[j]] for j in range(bs)], dim=0)


        # Compute All Loss Functions

        # programmer loss
        # prog_loss = self.programer_loss(pred_program, gt_program, gt_prog_len)  # length-weighted
        prog_loss = self.programer_loss(pred_program, gt_program)  # CELoss

        # captioning loss
        # cap_loss = self.captioning_loss(pred_captions, gt_captions, gt_cap_lens)  # length-weighted
        cap_loss = self.captioning_loss(pred_captions, gt_captions)  # CELoss

        # tagging loss
        sem_enc_loss = self.tagging_loss(pred_caps_sem_enc, gt_caps_sem_enc)  # CELoss

        # tIoU loss of intervals
        iou_loss = self.intervals_loss(pred_intervals, gt_intervals)

        # mm_loss = self.multimodal_loss(mm_v_encs, mm_t_encs)

        # combine and return losses
        # print(cap_loss.requires_grad, prog_loss.requires_grad, iou_loss.requires_grad)
        # losses = torch.tensor([cap_loss, prog_loss])
        # loss = torch.sum(self.comb_weights * losses)
        loss = cap_loss + prog_loss + sem_enc_loss

        return loss, prog_loss, cap_loss, sem_enc_loss, iou_loss
