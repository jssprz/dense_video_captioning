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

    def forward(self, logits, targets, lens, reinforce_from, rewards=None, epsilon=1e-8):
        log_probs = torch.log_softmax(logits, dim=1)
        len_mask = torch.cat([l.repeat(l) for l in lens], dim=0).unsqueeze(1).to(logits.device)

        if rewards is None:
            log_probs = torch.reciprocal(len_mask ** self.beta) * log_probs
        else:
            rewards = rewards.clamp(min=epsilon)
            r_mask = (
                torch.cat(
                    [torch.cat((torch.ones(s), r.repeat(l - s))) for l, r, s in zip(lens, rewards, reinforce_from)],
                    dim=0,
                )
                .unsqueeze(1)
                .to(logits.device)
            )
            log_probs = torch.reciprocal(len_mask ** self.beta) * (log_probs + torch.log(r_mask))

        loss = self.crit(log_probs, targets)
        return (loss / lens.size(0)) if self.reduction == "mean" else loss


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


def get_reinforce_strategy(criterion_config, epoch, gt_prog_len):
    rl_strategy = criterion_config.rl_strategy
    if rl_strategy == "reinforce":
        step_0_epochs = criterion_config.reinforce_config.step_0_epochs
        return (epoch + 1) > step_0_epochs, torch.zeros_like(gt_prog_len)
    elif rl_strategy == "mixer":
        step_0_epochs = criterion_config.mixer_config.step_0_epochs
        step_k_epochs = criterion_config.mixer_config.step_k_epochs
        samples_delta = criterion_config.mixer_config.samples_delta
        delta = (epoch + 1 - step_0_epochs) // step_k_epochs * samples_delta
        return (epoch + 1) > step_0_epochs, (gt_prog_len - delta).clamp(0)


class DenseCaptioningLoss(nn.Module):
    def __init__(self, config, c_max_len, p_max_len, s_proposal_pos_weights, e_proposal_pos_weights, device):
        super(DenseCaptioningLoss, self).__init__()

        self.config = config

        # captioning_loss function
        if config.captioning_loss == "MSE":
            self.captioning_loss = nn.MSELoss(reduction=config.captioning_loss_reduction)
        elif config.captioning_loss == "NLL":
            self.captioning_loss = nn.NLLLoss(reduction=config.captioning_loss_reduction)
        elif config.captioning_loss == "LenW":
            self.captioning_loss = SentenceLengthLoss(
                c_max_len,
                class_weights=None,
                beta=config.captioning_loss_b,
                reduction=config.captioning_loss_reduction,
            )
        elif config.captioning_loss == "XEnt":
            self.captioning_loss = nn.CrossEntropyLoss(reduction=config.captioning_loss_reduction)
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
            self.tagging_loss = nn.BCELoss(reduction=config.tagging_loss_reduction)
        else:
            raise ValueError(f"wrong value '{config.tagging_loss}' for the tagging_loss option in Loss configuration")

        # proposals_loss function
        if config.proposals_loss == "BXE":
            # self.proposals_loss = nn.BCELoss(reduction=config.proposals_loss_reduction)
            self.s_proposals_loss = nn.BCEWithLogitsLoss(
                pos_weight=s_proposal_pos_weights, reduction=config.proposals_loss_reduction
            )
            self.e_proposals_loss = nn.BCEWithLogitsLoss(
                pos_weight=e_proposal_pos_weights, reduction=config.proposals_loss_reduction
            )
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
        gt_proposals_s,
        gt_proposals_e,
        pred_proposals,
        gt_caps_count,
        pred_caps_count,
        gt_proposals_count,
        epoch,
        truncate_prog_at=None,
        mm_v_encs=None,
        mm_t_encs=None,
    ):
        # bs, _, _, caps_vocab_size = pred_captions.size()
        # pos_vocab_size = pred_pos_seq.size(3)
        bs, _, _ = pred_proposals.size()

        # TODO: compute gt flatten in the Dataset for removing it from here

        l7, l8 = [], []
        for n in range(bs):
            # for i in range(gt_caps_count[n]):
            #     l1.append(pred_captions[n, i, : gt_cap_lens[n, i]].reshape(-1, caps_vocab_size))
            #     l2.append(gt_captions[n, i, : gt_cap_lens[n, i]].flatten())

            l7.append(pred_proposals[n, : gt_caps_count[n], :])
            l8.append(gt_proposals_s[n, : gt_caps_count[n], :])

        # if len(l1):
        #     # captioning loss
        #     pred_captions = torch.cat(l1)
        #     gt_captions = torch.cat(l2)
        #     # cap_loss = self.captioning_loss(pred_captions, gt_captions, gt_cap_lens)  # length-weighted
        #     cap_loss = self.captioning_loss(pred_captions, gt_captions)  # CELoss
        # else:
        #     cap_loss = torch.tensor(float("inf"))

        # programmer loss
        # if truncate_prog_at is not None:
        #     # (bs*truncate_prog_at x progs_vocab_size)
        #     pred_program = pred_program[:, :truncate_prog_at].reshape(-1, progs_vocab_size)

        #     # (bs*truncate_prog_at)
        #     gt_program = gt_program[:, :truncate_prog_at].flatten()

        #     # (bs)
        #     gt_prog_len = torch.tensor([truncate_prog_at] * bs)
        # else:
        #     # straighten the output program (removing the part of the pad) and then flatten it
        #     # (total_len_of_programs x progs_vocab_size)
        #     pred_program = torch.cat([pred_program[n, : gt_prog_len[n]] for n in range(bs)], dim=0)

        #     # straighten the target captions (remove the part of the pad) and then flatten it
        #     # (total_len_of_programs)
        #     gt_program = torch.cat([gt_program[n, : gt_prog_len[n]] for n in range(bs)], dim=0)

        # iou_reward = temp_iou(pred_intervals, gt_intervals, gt_caps_count)

        # reinforce, reinforce_from = get_reinforce_strategy(self.config, epoch, gt_prog_len)
        # if reinforce and self.config.programer_use_iou_reward:
        #     # length-weighted + reward (mixer or not)
        #     prog_loss = self.programer_loss(pred_program, gt_program, gt_prog_len, reinforce_from, iou_reward)
        # else:
        #     # length-weighted
        #     prog_loss = self.programer_loss(pred_program, gt_program, gt_prog_len, None)
        #     # prog_loss = self.programer_loss(pred_program, gt_program)  # CELoss

        # event proposals loss
        pred_proposals = torch.cat(l7)
        gt_proposals = torch.cat(l8)
        proposals_loss = self.proposals_loss(pred_proposals, gt_proposals)

        # tIoU loss of intervals
        # iou_loss = self.intervals_loss(pred_intervals, gt_intervals)

        # mm_loss = self.multimodal_loss(mm_v_encs, mm_t_encs)

        # combine and return losses
        # print(cap_loss.requires_grad, prog_loss.requires_grad, iou_loss.requires_grad)
        # losses = torch.tensor([cap_loss, prog_loss])
        # loss = torch.sum(self.comb_weights * losses)
        loss = proposals_loss

        return (
            loss,
            None,  # prog_loss,
            None,  # cap_loss,
            None,  # sem_enc_loss,
            None,  # pos_loss,
            proposals_loss,
            None,  # iou_reward.mean(),
        )
