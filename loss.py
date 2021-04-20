import torch
import torch.nn as nn


class SentenceLengthLoss(nn.Module):
    def __init__(self, max_words, beta, reduction):
        super(SentenceLengthLoss, self).__init__()
        self.max_words = max_words
        self.beta = beta
        self.reduction= reduction
        self.crit = nn.NLLLoss(reduction='sum')

    def forward(self, logits, targets, lens, rewards=None):
        if rewards is None:
            mask = torch.cat([l.repeat(l) for l in lens], dim=0).unsqueeze(1).to(logits.device)
            wighted_log_probs = torch.reciprocal(mask ** self.beta) * torch.log_softmax(logits, dim=1)
            loss = self.crit(wighted_log_probs, targets)
        else:
            mask1 = torch.cat([l.repeat(self.max_words) for l in lens], dim=0).unsqueeze(1)
            mask2 = torch.cat([r.repeat(self.max_words) for r in rewards], dim=0).unsqueeze(1)
            wighted_log_probs = torch.reciprocal((mask1 * mask2) ** self.beta) * torch.log_softmax(logits, dim=1)
            loss = self.crit(wighted_log_probs, targets)

        if self.reduction=='mean':
            return loss / lens.size(0)
        elif self.reduction=='sum':
            return loss


class IoULoss(nn.Module):
    def __init__(self, reduction):
        super(IoULoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred_intervals, gt_intervals):
        # compute intersection
        intersection = torch.zeros(pred_intervals.size(0))
        M = torch.max(torch.cat((pred_intervals[:,0], gt_intervals[:,0]), dim=1), dim=1)[0]
        m = torch.min(torch.cat((pred_intervals[:,1], gt_intervals[:,1]), dim=1), dim=1)[0]
        intersection = (m-M).clamp(min=0)
        print('intersec:', intersection)

        # compute union
        m = torch.min(torch.cat((pred_intervals[:,0], gt_intervals[:,0]), dim=1), dim=1)[0]
        M = torch.max(torch.cat((pred_intervals[:,1], gt_intervals[:,1]), dim=1), dim=1)[0]
        union = M-m
        print('union:', union)

        # compute IoU
        loss = torch.sum(torch.reciprocal(union) * intersection)

        if self.reduction=='mean':
            return 1 -  loss / pred_intervals.size(0)
        elif self.reduction=='sum': 
            return pred_intervals.size(0) - loss


class DenseCaptioningLoss(nn.Module):
    def __init__(self, config, c_max_len, p_max_len):
        super(DenseCaptioningLoss, self).__init__()

        # captioning_loss function
        if config.closs == 'MSE':
            self.captioning_loss = nn.MSELoss(reduction=config.closs_reduction)
        elif config.closs == 'NLL':
            self.captioning_loss = nn.NLLLoss(reduction=config.closs_reduction)
        elif config.closs == 'SentLen':
            self.captioning_loss = SentenceLengthLoss(c_max_len, beta=config.closs_b, reduction=config.closs_reduction)
        else:
            self.captioning_loss = nn.CrossEntropyLoss(reduction=config.closs_reduction)

        # programer_loss function
        if config.ploss == 'MSE':
            self.programer_loss = nn.MSELoss(reduction=config.ploss_reduction)
        elif config.ploss == 'NLL':
            self.programer_loss = nn.NLLLoss(reduction=config.ploss_reduction)
        elif config.ploss == 'SentLen':
            self.programer_loss = SentenceLengthLoss(p_max_len, beta=config.ploss_b, reduction=config.ploss_reduction)
        else:
            self.programer_loss = nn.CrossEntropyLoss(reduction=config.ploss_reduction)

        # intervals_loss function
        if config.iloss == 'IoU':
            self.intervals_loss = IoULoss(reduction=config.iloss_reduction)
        
        # multimodal_loss function
        # if mmloss == 'MSE':
        #     self.multimodal_loss = nn.MSELoss(reduction=config.mmloss_reduction)

        if config.learn_scores:
            self.scores = nn.Parameter(torch.Tensor(config.combination_scores))
        else:
            self.scores = torch.tensor(config.combination_scores)

    def forward(self, gt_captions, gt_cap_lens, pred_captions, gt_program, gt_prog_len, pred_program, 
                gt_intervals, pred_intervals, gt_caps_count, pred_caps_count, mm_v_encs=None, mm_t_encs=None):

        bs, _, _, caps_vocab_size = pred_captions.size()
        progs_vocab_size = pred_captions.size(2)

        print(pred_captions.requires_grad, pred_intervals.requires_grad)

        #TODO: compute gt strighten in the Dataset for removing it from here

        # straighten the output captions (removing the part of the pad)
        # (total_len_of_captions x caps_vocab_size)
        pred_captions = torch.cat([pred_captions[j, i, :gt_cap_lens[j,i]].reshape(-1, caps_vocab_size) for j in range(bs) for i in range(gt_caps_count[j])], dim=0)

        # straighten the target captions (remove the part of the pad) and then flatten it
        # (total_len_of_captions)
        gt_captions = torch.cat([gt_captions[j, i, :gt_cap_lens[j,i]].flatten() for j in range(bs) for i in range(gt_caps_count[j])], dim=0)

        # straighten the lens of target captions (remove the part of the pad) and then flatten it
        # (total_num_captions)
        gt_cap_lens = torch.tensor([gt_cap_lens[j, i] for j in range(bs) for i in range(gt_caps_count[j])], dtype=torch.int32)

        # straighten the output program (removing the part of the pad) and then flatten it
        # (total_len_of_programs x progs_vocab_size)
        pred_program = torch.cat([pred_program[j, :gt_prog_len[j]] for j in range(bs)], dim=0)

        # straighten the target captions (remove the part of the pad) and then flatten it
        # (total_len_of_programs)
        gt_program = torch.cat([gt_program[j, :gt_prog_len[j]] for j in range(bs)], dim=0)

        # Compute All Loss Functions
        cap_loss = self.captioning_loss(pred_captions, gt_captions, gt_cap_lens)
        prog_loss = self.programer_loss(pred_program, gt_program, gt_prog_len)
        iou_loss = self.intervals_loss(pred_intervals, gt_intervals)
        # mm_loss = self.multimodal_loss(mm_v_encs, mm_t_encs)

        # combine and return losses
        losses = torch.tensor([cap_loss, prog_loss, iou_loss], requires_grad=True)
        loss = torch.sum(self.scores * losses)

        return loss, cap_loss, prog_loss, iou_loss
