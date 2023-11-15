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

    def forward(
        self, logits, targets, lens, reinforce_from, rewards=None, epsilon=1e-8
    ):
        log_probs = torch.log_softmax(logits, dim=1)
        len_mask = (
            torch.cat([l.repeat(l) for l in lens], dim=0).unsqueeze(1).to(logits.device)
        )

        if rewards is None:
            log_probs = torch.reciprocal(len_mask**self.beta) * log_probs
        else:
            rewards = rewards.clamp(min=epsilon)
            r_mask = (
                torch.cat(
                    [
                        torch.cat((torch.ones(s), r.repeat(l - s)))
                        for l, r, s in zip(lens, rewards, reinforce_from)
                    ],
                    dim=0,
                )
                .unsqueeze(1)
                .to(logits.device)
            )
            log_probs = torch.reciprocal(len_mask**self.beta) * (
                log_probs + torch.log(r_mask)
            )

        loss = self.crit(log_probs, targets)
        return (loss / lens.size(0)) if self.reduction == "mean" else loss


def temp_iou(pred_intervals, gt_intervals, gt_count):
    # compute intersection
    max_min = torch.max(
        torch.stack([pred_intervals[:, :, 0], gt_intervals[:, :, 0]]), dim=0
    )[0]
    min_max = torch.min(
        torch.stack([pred_intervals[:, :, 1], gt_intervals[:, :, 1]]), dim=0
    )[0]
    intersection = (min_max - max_min).clamp(min=0)
    # print('intersec:', intersection)

    # compute union
    min_min = torch.min(
        torch.stack([pred_intervals[:, :, 0], gt_intervals[:, :, 0]]), dim=0
    )[0]
    max_max = torch.max(
        torch.stack([pred_intervals[:, :, 1], gt_intervals[:, :, 1]]), dim=0
    )[0]
    union = max_max - min_min
    # print('union:', union)

    # compute total IoU
    return torch.FloatTensor(
        [
            (
                torch.reciprocal(union[n, : gt_count[n]])
                * intersection[n, : gt_count[n]]
            ).mean()
            if gt_count[n]
            else 0
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
    def __init__(
        self,
        config,
        c_max_len,
        p_max_len,
        sem_enc_pos_weights,
        s_prop_pos_weights,
        e_prop_pos_weights,
        training_proposals,
        training_programmer,
        training_pos_tagging,
        training_sem_tagging,
        training_captioning,
        device,
    ):
        super(DenseCaptioningLoss, self).__init__()

        self.config = config

        self.training_proposals = training_proposals
        self.training_programmer = training_programmer
        self.training_captioning = training_captioning
        self.training_pos_tagging = training_pos_tagging
        self.training_sem_tagging = training_sem_tagging

        if training_captioning:
            # captioning_loss function
            if config.captioning_loss == "MSE":
                self.captioning_loss = nn.MSELoss(
                    reduction=config.captioning_loss_reduction
                )
            elif config.captioning_loss == "NLL":
                self.captioning_loss = nn.NLLLoss(
                    reduction=config.captioning_loss_reduction
                )
            elif config.captioning_loss == "SentLen":
                self.captioning_loss = SentenceLengthLoss(
                    c_max_len,
                    class_weights=None,
                    beta=config.captioning_loss_b,
                    reduction=config.captioning_loss_reduction,
                )
            elif config.captioning_loss == "XEnt":
                self.captioning_loss = nn.CrossEntropyLoss(
                    reduction=config.captioning_loss_reduction
                )
            else:
                raise ValueError(
                    f"wrong value '{config.captioning_loss}' for the captioning_loss option in Loss configuration"
                )

        if training_pos_tagging:
            # pos_tagging_loss function
            if config.pos_tag_loss == "MSE":
                self.pos_tag_loss = nn.MSELoss(reduction=config.pos_tag_loss_reduction)
            elif config.pos_tag_loss == "NLL":
                self.pos_tag_loss = nn.NLLLoss(reduction=config.pos_tag_loss_reduction)
            elif config.pos_tag_loss == "LenW":
                self.pos_tag_loss = SentenceLengthLoss(
                    c_max_len,
                    class_weights=None,
                    beta=config.pos_tag_loss_b,
                    reduction=config.pos_tag_loss_reduction,
                )
            elif config.pos_tag_loss == "XEnt":
                self.pos_tag_loss = nn.CrossEntropyLoss(
                    reduction=config.pos_tag_loss_reduction
                )
            else:
                raise ValueError(
                    f"wrong value '{config.pos_tag_loss}' for the pos_tag_loss option in Loss configuration"
                )

        if training_proposals:
            # proposals_loss function
            if config.proposals_loss == "BXE":
                # self.proposals_loss = nn.BCELoss(reduction=config.proposals_loss_reduction)
                self.s_prop_loss = nn.BCEWithLogitsLoss(
                    pos_weight=s_prop_pos_weights,
                    reduction=config.proposals_loss_reduction,
                )
                self.e_prop_loss = nn.BCEWithLogitsLoss(
                    pos_weight=e_prop_pos_weights,
                    reduction=config.proposals_loss_reduction,
                )
            else:
                raise ValueError(
                    f"wrong value '{config.proposals_loss}' for the proposals_loss option in Loss configuration"
                )

        if training_programmer:
            # programer_loss function
            if config.programer_loss == "MSE":
                self.programer_loss = nn.MSELoss(
                    reduction=config.programer_loss_reduction
                )
            elif config.programer_loss == "NLL":
                self.programer_loss = nn.NLLLoss(
                    reduction=config.programer_loss_reduction
                )
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
                self.programer_loss = nn.CrossEntropyLoss(
                    weight=class_weights, reduction=config.programer_loss_reduction
                )
            else:
                raise ValueError(
                    f"wrong value '{config.programer_loss}' for the programer_loss option in Loss configuration"
                )
        if training_sem_tagging:
            # semantic_tagging_loss function
            if config.tagging_loss == "BXEnt":
                self.tagging_loss = nn.BCELoss(reduction=config.tagging_loss_reduction)
            else:
                raise ValueError(
                    f"wrong value '{config.tagging_loss}' for the tagging_loss option in Loss configuration"
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
        gt_caps,
        gt_cap_lens,
        pred_caps,
        gt_caps_sem_enc,
        pred_caps_sem_enc,
        gt_pos_seq,
        pred_pos_seq,
        gt_program,
        gt_prog_len,
        pred_program,
        gt_intervals,
        pred_intervals,
        gt_prop_s,
        gt_prop_e,
        pred_prop_s,
        pred_prop_e,
        gt_caps_count,
        pred_caps_count,
        gt_prop_count,
        epoch,
        truncate_prog_at=None,
        mm_v_encs=None,
        mm_t_encs=None,
    ):
        # TODO: compute gt flatten in the Dataset for removing it from here

        pred_ls = {
            "prop_s": [],
            "prop_e": [],
            "programmer": [],
            "captioning": [],
            "pos_tagging": [],
            "sem_tagging": [],
        }
        gt_ls = {
            "prop_s": [],
            "prop_e": [],
            "programmer": [],
            "captioning": [],
            "pos_tagging": [],
            "sem_tagging": [],
        }
        bs, _, _, caps_vocab_size = pred_caps.size()
        pos_vocab_size = pred_pos_seq.size(3)
        for n in range(bs):
            for i in range(gt_caps_count[n]):
                if self.training_captioning:
                    pred_ls["captioning"].append(
                        pred_caps[n, i, : gt_cap_lens[n, i]].reshape(
                            -1, caps_vocab_size
                        )
                    )
                    gt_ls["captioning"].append(
                        gt_caps[n, i, : gt_cap_lens[n, i]].flatten()
                    )

                if self.training_pos_tagging:
                    pred_ls["pos_tagging"].append(
                        pred_pos_seq[n, i, : gt_cap_lens[n, i]].reshape(
                            -1, pos_vocab_size
                        )
                    )
                    gt_ls["pos_tagging"].append(
                        gt_pos_seq[n, i, : gt_cap_lens[n, i]].flatten()
                    )

            if self.training_proposals:
                pred_ls["prop_s"].append(pred_prop_s[n, : gt_caps_count[n], :])
                gt_ls["prop_s"].append(gt_prop_s[n, : gt_caps_count[n], :])

                pred_ls["prop_e"].append(pred_prop_e[n, : gt_caps_count[n], :])
                gt_ls["prop_e"].append(gt_prop_e[n, : gt_caps_count[n], :])

            if self.training_sem_tagging:
                pred_ls["sem_tagging"].append(
                    pred_caps_sem_enc[n, : gt_caps_count[n], :]
                )
                gt_ls["sem_tagging"].append(gt_caps_sem_enc[n, : gt_caps_count[n], :])

            if self.training_programmer:
                pred_ls["programmer"].append(pred_program[n, : gt_prog_len[n], :])
                gt_ls["programmer"].append(gt_program[n, : gt_prog_len[n]])

        loss = {}
        if self.training_proposals:
            # event proposals loss
            s_pred_proposals = torch.cat(pred_ls["prop_s"])
            s_gt_proposals = torch.cat(gt_ls["prop_s"])
            s_prop_loss = self.s_prop_loss(s_pred_proposals, s_gt_proposals)

            e_pred_proposals = torch.cat(pred_ls["prop_e"])
            e_gt_proposals = torch.cat(gt_ls["prop_e"])
            e_prop_loss = self.e_prop_loss(e_pred_proposals, e_gt_proposals)

            loss["prop_loss"] = (s_prop_loss + e_prop_loss), s_prop_loss, e_prop_loss

        if self.training_programmer:
            # program instructions loss
            pred_program = torch.cat(pred_ls["programmer"])
            gt_program = torch.cat(gt_ls["programmer"])

            loss["prog_loss"] = self.programer_loss(pred_program, gt_program)  # CELoss

        if self.training_captioning:
            # captioning loss
            pred_caps = torch.cat(pred_ls["captioning"])
            gt_caps = torch.cat(gt_ls["captioning"])
            # cap_loss = self.captioning_loss(pred_caps, gt_caps, gt_cap_lens)  # length-weighted
            loss["cap_loss"] = self.captioning_loss(pred_caps, gt_caps)  # CELoss

        if self.training_pos_tagging:
            # pos-tagging loss
            pred_pos_seq = torch.cat(pred_ls["pos_tagging"])
            gt_pos_seq = torch.cat(gt_ls["pos_tagging"])
            # cap_loss = self.captioning_loss(pred_pos_seq, gt_pos_seq, gt_cap_lens)  # length-weighted
            loss["pos_loss"] = self.captioning_loss(pred_pos_seq, gt_pos_seq)  # CELoss

        if self.training_sem_tagging:
            # semantic tagging loss
            pred_caps_sem_enc = torch.cat(pred_ls["sem_tagging"])
            gt_caps_sem_enc = torch.cat(gt_ls["sem_tagging"])
            loss["sem_enc_loss"] = self.tagging_loss(pred_caps_sem_enc, gt_caps_sem_enc)

        loss["total_loss"] = sum(
            l if type(l) != tuple else l[0] for _, l in loss.items()
        )
        return loss
