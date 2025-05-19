import torch.nn.functional as F
import torch.nn as nn
import torch, math
from models.matcher import TripleMatcher, EntityMatcher, Ent_HeadTail_Matcher, TripleMatcher2
from scipy.special import logit

def binary_cross_entropy_(input, target, pad_value=0.0):
    num_no_pad = torch.sum((input != pad_value))
    loss = F.binary_cross_entropy(input, target, reduction='sum')
    loss = loss / num_no_pad
    return loss


class SetCriterion(nn.Module):
    """ This class computes the loss for Set_RE.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class, subject position and object position)
    """
    def __init__(self, num_classes, loss_weight, na_coef, losses, matcher, num_ent_types, ner_na_coef, ner_losses, hybrid=True):
        """ Create the criterion.
        Parameters:
            num_classes: number of relation categories
            matcher: module able to compute a matching between targets and proposals
            loss_weight: dict containing as key the names of the losses and as values their relative weight.
            na_coef: list containg the relative classification weight applied to the NA category and positional classification weight applied to the [SEP]
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.rel_matcher = TripleMatcher(loss_weight, matcher, boundary_softmax=True)
        self.ent_matcher = EntityMatcher(loss_weight, matcher, boundary_softmax=True)
        self.hybrid = hybrid

        self.re_losses = losses
        if losses:
            self.num_classes = num_classes
            rel_weight = torch.ones(num_classes + 1)
            rel_weight[-1] = na_coef
            self.register_buffer('rel_weight', rel_weight)

        self.ner_losses = ner_losses
        if ner_losses:
            self.num_ent_types = num_ent_types
            ner_weight = torch.ones(num_ent_types + 1)
            ner_weight[-1] = ner_na_coef
            self.register_buffer('ner_weight', ner_weight)


    def forward(self, outputs, targets, indices=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute all the requested losses
        losses = {}
        ent_indices = None
        rel_indices = None
        if self.re_losses:
            rel_indices = self.rel_matcher(outputs, targets) if indices is None else indices
            num_tgt = sum([len(t['relation']) for t in targets])    # e.g. 20 + 20 = 40
            for loss in self.re_losses:
                if loss in ["relation", "entity", "head_tail_part", "head_tail_type"] and self.empty_targets(targets, "relation"):
                    pass
                else:
                    losses.update(self.get_loss(loss, outputs, targets, rel_indices, num_tgt))

        if self.ner_losses:
            ent_indices = self.ent_matcher(outputs, targets) if indices is None else indices
            for loss in self.ner_losses:
                if loss in ["ner_span", "ner_part", "ent_have_rel"] and self.empty_targets(targets, "ent_type"):
                    pass
                else:
                    losses.update(self.get_loss(loss, outputs, targets, ent_indices))

        # for k, v in losses.items():
        #     print(k, v.item())
        # print()

        losses = sum(losses[k] * self.loss_weight[k] for k in losses.keys() if k in self.loss_weight and self.loss_weight[k]>0)
        return losses, ent_indices, rel_indices
    # indices格式：[(tensor11,tensor12),(tensor21,tensor22),(tensor31,tensor32)]    tensor: size([20])
    def relation_loss(self, outputs, targets, indices, num_tgt):
        """Classification loss (NLL)
        targets dicts must contain the key "relation" containing a tensor of dim [bsz]
        """
        src_logits = outputs['pred_rel_logits'] # [bsz, num_generated_triples, num_rel+1]   e.g.[3, 40, 2]
        idx = _get_src_permutation_idx(indices) # idx: (tensor1, tensor2)   tensor: size([num_generated_triples])
        target_classes_o = torch.cat([t["relation"][i] for t, (_, i) in zip(targets, indices)]) # size([num_generated_triple])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,         # [bsz, num_generated_triples]
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        weight = self.rel_weight.clone()
        if weight[-1] == -1:
            weight[-1] = num_tgt / (src_logits.size(0) * src_logits.size(1) - num_tgt)

        loss = F.cross_entropy(src_logits.flatten(0, 1), target_classes.flatten(0, 1), weight=weight)
        losses = {'relation': loss}
        # print(target_classes[2])
        # print(losses, flush=True)
        assert loss >= 0
        return losses

    def entity_loss(self, outputs, targets, indices, num_tgt):
        """Compute the losses related to the position of head entity or tail entity
        """
        idx = _get_src_permutation_idx(indices)
        selected_pred_head_start = outputs["head_start_logits"][idx]    # size([triple_query_num, seq_len])  e.g.size([40, 75])
        selected_pred_head_end = outputs["head_end_logits"][idx]
        selected_pred_tail_start = outputs["tail_start_logits"][idx]
        selected_pred_tail_end = outputs["tail_end_logits"][idx]
        # size([triple_query_num])    e.g.([40])
        target_head_start = torch.cat([t["head_start_index"][i] for t, (_, i) in zip(targets, indices)])    # zip函数：打包多个可迭代对象(按照最短长度)
        target_head_end = torch.cat([t["head_end_index"][i] for t, (_, i) in zip(targets, indices)])
        target_tail_start = torch.cat([t["tail_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_tail_end = torch.cat([t["tail_end_index"][i] for t, (_, i) in zip(targets, indices)])

        head_start_loss = F.cross_entropy(selected_pred_head_start, target_head_start)
        head_end_loss = F.cross_entropy(selected_pred_head_end, target_head_end)
        tail_start_loss = F.cross_entropy(selected_pred_tail_start, target_tail_start)
        tail_end_loss = F.cross_entropy(selected_pred_tail_end, target_tail_end)

        if self.hybrid:
            selected_pred_head_start = outputs["p_head_start"][idx]     # size([triple_query_num, seq_len])  e.g.size([40, 75])
            selected_pred_head_end = outputs["p_head_end"][idx]
            selected_pred_tail_start = outputs["p_tail_start"][idx]
            selected_pred_tail_end = outputs["p_tail_end"][idx]
            # size([triple_query_num, seq_len])     e.g. size([40, 75])
            head_left_onehot = torch.zeros([target_head_start.size(0), selected_pred_head_start.size(1)], dtype=torch.float32).to(device=selected_pred_head_start.device)
            head_left_onehot.scatter_(1, target_head_start.unsqueeze(1), 1)   # tensor._scatter参数： (要修改的维度，要修改的参照索引，要填充的值)
            head_right_onehot = torch.zeros([target_head_end.size(0), selected_pred_head_end.size(1)], dtype=torch.float32).to(device=selected_pred_head_end.device)
            head_right_onehot.scatter_(1, target_head_end.unsqueeze(1), 1)

            tail_left_onehot = torch.zeros([target_tail_start.size(0), selected_pred_tail_start.size(1)], dtype=torch.float32).to(device=selected_pred_tail_start.device)
            tail_left_onehot.scatter_(1, target_tail_start.unsqueeze(1), 1)
            tail_right_onehot = torch.zeros([target_tail_end.size(0), selected_pred_tail_end.size(1)], dtype=torch.float32).to(device=selected_pred_tail_end.device)
            tail_right_onehot.scatter_(1, target_tail_end.unsqueeze(1), 1)

            head_start_loss = (head_start_loss + binary_cross_entropy_(selected_pred_head_start, head_left_onehot))
            head_end_loss = (head_end_loss + binary_cross_entropy_(selected_pred_head_end, head_right_onehot))
            tail_start_loss = (tail_start_loss + binary_cross_entropy_(selected_pred_tail_start, tail_left_onehot))
            tail_end_loss = (tail_end_loss + binary_cross_entropy_(selected_pred_tail_end, tail_right_onehot))

        losses = {'head_entity': head_start_loss + head_end_loss, "tail_entity": tail_start_loss + tail_end_loss}
        # print(losses, flush=True)
        return losses

    def head_tail_part_loss(self, outputs, targets, indices, num_tgt):
        """Compute the losses related to the entity part detection
        """
        idx = _get_src_permutation_idx(indices)
        selected_pred_head_part = outputs["head_part_logits"][idx]
        selected_pred_tail_part = outputs["tail_part_logits"][idx]

        target_head_part = torch.cat([t["head_part_labels"][i] for t, (_, i) in zip(targets, indices)]).unsqueeze(-1)
        target_tail_part = torch.cat([t["tail_part_labels"][i] for t, (_, i) in zip(targets, indices)]).unsqueeze(-1)

        head_part_loss = F.binary_cross_entropy_with_logits(selected_pred_head_part, target_head_part)
        tail_part_loss = F.binary_cross_entropy_with_logits(selected_pred_tail_part, target_tail_part)
        losses = {'head_part': head_part_loss, 'tail_part': tail_part_loss}
        # print(losses, flush=True)
        return losses

    def head_tail_type_loss(self, outputs, targets, indices, num_tgt):
        """Compute the losses related to the entity part detection
        """
        idx = _get_src_permutation_idx(indices)
        selected_pred_head_type = outputs["head_type_logits"][idx]
        selected_pred_tail_type = outputs["tail_type_logits"][idx]

        target_head_type = torch.cat([t["head_type"][i] for t, (_, i) in zip(targets, indices)])
        target_tail_type = torch.cat([t["tail_type"][i] for t, (_, i) in zip(targets, indices)])

        head_type_loss = F.cross_entropy(selected_pred_head_type, target_head_type)
        tail_type_loss = F.cross_entropy(selected_pred_tail_type, target_tail_type)
        losses = {'head_tail_type': head_type_loss + tail_type_loss}
        # print(losses, flush=True)
        return losses

    def ner_type_loss(self, outputs, targets, indices):
        """Compute the losses related to NER
        """
        src_logits = outputs['ent_type_logits'] # [bsz, num_generated_entities, num_ent_type+1]
        idx = _get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["ent_type"][i] for t, (_, i) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_ent_types,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss = F.cross_entropy(src_logits.flatten(0, 1), target_classes.flatten(0, 1), weight=self.ner_weight)
        losses = {'ent_type': loss}
        return losses

    def ner_span_loss(self, outputs, targets, indices):
        """Compute the losses related to the span position of NER
        """
        idx = _get_src_permutation_idx(indices)
        selected_pred_ent_start = outputs["ent_start_logits"][idx]
        selected_pred_ent_end = outputs["ent_end_logits"][idx]

        target_ent_start = torch.cat([t["ent_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_ent_end = torch.cat([t["ent_end_index"][i] for t, (_, i) in zip(targets, indices)])

        ent_start_loss = F.cross_entropy(selected_pred_ent_start, target_ent_start)
        ent_end_loss = F.cross_entropy(selected_pred_ent_end, target_ent_end)
        losses = {'ent_span': (ent_start_loss + ent_end_loss)}
        # print(losses)
        return losses

    def ner_part_loss(self, outputs, targets, indices):
        """Compute the losses related to the entity part detection
        """
        idx = _get_src_permutation_idx(indices)
        selected_pred_ent_part = outputs["ent_part_logits"][idx]

        target_ent_part = torch.cat([t["ent_part_labels"][i] for t, (_, i) in zip(targets, indices)]).unsqueeze(-1)

        ent_part_loss = F.binary_cross_entropy_with_logits(selected_pred_ent_part, target_ent_part)
        losses = {'ent_part': ent_part_loss}
        # print(losses)
        return losses

    def ent_have_rel_loss(self, outputs, targets, indices):
        """Compute the losses on whether the entity has relation
        """
        idx = _get_src_permutation_idx(indices)
        selected_pred = outputs["ent_have_rel_logits"][idx]
        target = torch.cat([t["ent_have_rel"][i] for t, (_, i) in zip(targets, indices)])

        loss = F.cross_entropy(selected_pred, target)
        losses = {'ent_have_rel': loss}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty triples
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_rel_logits = outputs['pred_rel_logits']
        device = pred_rel_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_rel_logits.argmax(-1) != pred_rel_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_tgt, **kwargs):
        loss_map = {
            'relation': self.relation_loss,
            'cardinality': self.loss_cardinality,
            'entity': self.entity_loss,
            'ner_type': self.ner_type_loss,
            'ner_span': self.ner_span_loss,
            'ner_part': self.ner_part_loss,
            'head_tail_part': self.head_tail_part_loss,
            'ent_have_rel': self.ent_have_rel_loss,
            'head_tail_type': self.head_tail_type_loss
        }
        return loss_map[loss](outputs, targets, indices, num_tgt, **kwargs)

    @staticmethod
    def empty_targets(targets, filed):
        flag = True
        for target in targets:
            if len(target[filed]) != 0:
                flag = False
                break
        return flag


class SetCriterion2(nn.Module):
    """ Same as SetCriterion, but using BCE loss for boundary prediction.
    """
    def __init__(self, num_classes, loss_weight, na_coef, losses, matcher, num_ent_types, ner_na_coef, ner_losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of relation categories
            matcher: module able to compute a matching between targets and proposals
            loss_weight: dict containing as key the names of the losses and as values their relative weight.
            na_coef: list containg the relative classification weight applied to the NA category and positional classification weight applied to the [SEP]
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.rel_matcher = TripleMatcher(loss_weight, matcher, boundary_softmax=False)
        self.ent_matcher = EntityMatcher(loss_weight, matcher, boundary_softmax=False)

        self.re_losses = losses
        if losses:
            self.num_classes = num_classes
            rel_weight = torch.ones(num_classes + 1)
            rel_weight[-1] = na_coef
            self.register_buffer('rel_weight', rel_weight)

        self.ner_losses = ner_losses
        if ner_losses:
            self.num_ent_types = num_ent_types
            ner_weight = torch.ones(num_ent_types + 1)
            ner_weight[-1] = ner_na_coef
            self.register_buffer('ner_weight', ner_weight)

    def forward(self, outputs, targets, indices=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute all the requested losses
        losses = {}
        ent_indices = None
        rel_indices = None
        if self.re_losses:
            rel_indices = self.rel_matcher(outputs, targets) if indices is None else indices
            num_tgt = sum([len(t['relation']) for t in targets])
            for loss in self.re_losses:
                if loss in ["entity", "head_tail_part", "head_tail_type"] and self.empty_targets(targets, "relation"):
                    pass
                else:
                    losses.update(self.get_loss(loss, outputs, targets, rel_indices, num_tgt))

        if self.ner_losses:
            ent_indices = self.ent_matcher(outputs, targets) if indices is None else indices  # size: [3][2][50]
            num_tgt = sum([len(t['ent_type']) for t in targets])  # [50,50,50]
            for loss in self.ner_losses:
                if loss in ["ner_span", "ner_part", "ent_have_rel"] and self.empty_targets(targets, "ent_type"):
                    pass
                else:
                    losses.update(self.get_loss(loss, outputs, targets, ent_indices, num_tgt))

        # for k, v in losses.items():
        #     print(k, v.item())
        # print()

        losses = sum(losses[k] * self.loss_weight[k] for k in losses.keys() if k in self.loss_weight and self.loss_weight[k]>0)
        return losses, ent_indices, rel_indices

    def relation_loss(self, outputs, targets, indices, num_tgt):
        """Classification loss (NLL)
        targets dicts must contain the key "relation" containing a tensor of dim [bsz]
        """
        src_logits = outputs['pred_rel_logits'] # [bsz, num_generated_triples, num_rel+1]
        idx = _get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["relation"][i] for t, (_, i) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        weight = self.rel_weight.clone()
        if weight[-1] == -1:
            weight[-1] = num_tgt / (src_logits.size(0) * src_logits.size(1) - num_tgt)

        loss = F.cross_entropy(src_logits.flatten(0, 1), target_classes.flatten(0, 1), weight=weight)
        losses = {'relation': loss}
        assert loss >= 0
        return losses

    def entity_loss(self, outputs, targets, indices, num_tgt):
        """Compute the losses related to the position of head entity or tail entity
        """
        idx = _get_src_permutation_idx(indices)
        selected_pred_head_start = outputs["p_head_start"][idx]
        selected_pred_head_end = outputs["p_head_end"][idx]
        selected_pred_tail_start = outputs["p_tail_start"][idx]
        selected_pred_tail_end = outputs["p_tail_end"][idx]

        target_head_start = torch.cat([t["head_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_head_end = torch.cat([t["head_end_index"][i] for t, (_, i) in zip(targets, indices)])
        target_tail_start = torch.cat([t["tail_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_tail_end = torch.cat([t["tail_end_index"][i] for t, (_, i) in zip(targets, indices)])

        # head_start_loss = F.cross_entropy(selected_pred_head_start, target_head_start)
        # head_end_loss = F.cross_entropy(selected_pred_head_end, target_head_end)
        # tail_start_loss = F.cross_entropy(selected_pred_tail_start, target_tail_start)
        # tail_end_loss = F.cross_entropy(selected_pred_tail_end, target_tail_end)

        head_left_onehot = torch.zeros([target_head_start.size(0), selected_pred_head_start.size(1)], dtype=torch.float32).to(device=selected_pred_head_start.device)
        head_left_onehot.scatter_(1, target_head_start.unsqueeze(1), 1)
        head_right_onehot = torch.zeros([target_head_end.size(0), selected_pred_head_end.size(1)], dtype=torch.float32).to(device=selected_pred_head_end.device)
        head_right_onehot.scatter_(1, target_head_end.unsqueeze(1), 1)

        tail_left_onehot = torch.zeros([target_tail_start.size(0), selected_pred_tail_start.size(1)], dtype=torch.float32).to(device=selected_pred_tail_start.device)
        tail_left_onehot.scatter_(1, target_tail_start.unsqueeze(1), 1)
        tail_right_onehot = torch.zeros([target_tail_end.size(0), selected_pred_tail_end.size(1)], dtype=torch.float32).to(device=selected_pred_tail_end.device)
        tail_right_onehot.scatter_(1, target_tail_end.unsqueeze(1), 1)

        head_start_loss = binary_cross_entropy_(selected_pred_head_start, head_left_onehot)
        head_end_loss = binary_cross_entropy_(selected_pred_head_end, head_right_onehot)
        tail_start_loss = binary_cross_entropy_(selected_pred_tail_start, tail_left_onehot)
        tail_end_loss = binary_cross_entropy_(selected_pred_tail_end, tail_right_onehot)

        losses = {'head_entity': 1/2*(head_start_loss + head_end_loss), "tail_entity": 1/2*(tail_start_loss + tail_end_loss)}
        # print(losses)
        return losses

    def head_tail_part_loss(self, outputs, targets, indices, num_tgt):
        """Compute the losses related to the entity part detection
        """
        idx = _get_src_permutation_idx(indices)
        selected_pred_head_part = outputs["p_head_part"][idx]
        selected_pred_tail_part = outputs["p_tail_part"][idx]

        target_head_part = torch.cat([t["head_part_labels"][i] for t, (_, i) in zip(targets, indices)]).unsqueeze(-1)
        target_tail_part = torch.cat([t["tail_part_labels"][i] for t, (_, i) in zip(targets, indices)]).unsqueeze(-1)

        head_part_loss = binary_cross_entropy_(selected_pred_head_part, target_head_part)
        tail_part_loss = binary_cross_entropy_(selected_pred_tail_part, target_tail_part)
        losses = {'head_part': head_part_loss, 'tail_part': tail_part_loss}
        # print(losses)
        return losses

    def head_tail_type_loss(self, outputs, targets, indices, num_tgt):
        """Compute the losses related to the entity part detection
        """
        idx = _get_src_permutation_idx(indices)
        selected_pred_head_type = outputs["head_type_logits"][idx]
        selected_pred_tail_type = outputs["tail_type_logits"][idx]

        target_head_type = torch.cat([t["head_type"][i] for t, (_, i) in zip(targets, indices)])
        target_tail_type = torch.cat([t["tail_type"][i] for t, (_, i) in zip(targets, indices)])

        head_type_loss = F.cross_entropy(selected_pred_head_type, target_head_type)
        tail_type_loss = F.cross_entropy(selected_pred_tail_type, target_tail_type)
        losses = {'head_tail_type': head_type_loss + tail_type_loss}
        # print(losses, flush=True)
        return losses

    def ner_type_loss(self, outputs, targets, indices, num_tgt):
        """Compute the losses related to NER
        """
        src_logits = outputs['ent_type_logits'] # [bsz, num_generated_entities, num_ent_type+1]
        idx = _get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["ent_type"][i] for t, (_, i) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_ent_types,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        weight = self.ner_weight.clone()
        if weight[-1] == -1:
            weight[-1] = num_tgt / (src_logits.size(0) * src_logits.size(1) - num_tgt)
        
        loss = F.cross_entropy(src_logits.flatten(0, 1), target_classes.flatten(0, 1), weight=weight)
        losses = {'ent_type': loss}
        # print(src_logits)
        # print(target_classes)
        # print(losses, flush=True)
        return losses

    def ner_span_loss(self, outputs, targets, indices, num_tgt):
        """Compute the losses related to the span position of NER
        """
        idx = _get_src_permutation_idx(indices)
        selected_pred_ent_start = outputs["p_left"][idx]
        selected_pred_ent_end = outputs["p_right"][idx]

        target_ent_start = torch.cat([t["ent_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_ent_end = torch.cat([t["ent_end_index"][i] for t, (_, i) in zip(targets, indices)])

        left_onehot = torch.zeros([target_ent_start.size(0), selected_pred_ent_start.size(1)], dtype=torch.float32).to(device=selected_pred_ent_start.device)
        left_onehot.scatter_(1, target_ent_start.unsqueeze(1), 1)
        right_onehot = torch.zeros([target_ent_end.size(0), selected_pred_ent_end.size(1)], dtype=torch.float32).to(device=selected_pred_ent_end.device)
        right_onehot.scatter_(1, target_ent_end.unsqueeze(1), 1)

        ent_start_loss = binary_cross_entropy_(selected_pred_ent_start, left_onehot)
        ent_end_loss = binary_cross_entropy_(selected_pred_ent_end, right_onehot)

        losses = {'ent_span': ent_start_loss + ent_end_loss}
        # print(losses, flush=True)
        return losses

    def ner_part_loss(self, outputs, targets, indices, num_tgt):
        """Compute the losses related to the entity part detection
        """
        idx = _get_src_permutation_idx(indices)
        selected_pred_ent_part = outputs["p_part"][idx]

        target_ent_part = torch.cat([t["ent_part_labels"][i] for t, (_, i) in zip(targets, indices)]).unsqueeze(-1)

        # ent_part_loss = F.binary_cross_entropy(selected_pred_ent_part, target_ent_part, reduction='mean')

        ent_part_loss = binary_cross_entropy_(selected_pred_ent_part, target_ent_part)

        losses = {'ent_part': ent_part_loss}
        # print(losses, flush=True)
        return losses

    def ent_have_rel_loss(self, outputs, targets, indices, num_tgt):
        """Compute the losses on whether the entity has relation
        """
        idx = _get_src_permutation_idx(indices)
        selected_pred = outputs["ent_have_rel_logits"][idx]
        target = torch.cat([t["ent_have_rel"][i] for t, (_, i) in zip(targets, indices)])
        loss = F.cross_entropy(selected_pred, target)
        # loss = F.binary_cross_entropy_with_logits(selected_pred.squeeze(), target)
        losses = {'ent_have_rel': loss}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty triples
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_rel_logits = outputs['pred_rel_logits']
        device = pred_rel_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_rel_logits.argmax(-1) != pred_rel_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_tgt, **kwargs):
        loss_map = {
            'relation': self.relation_loss,
            'cardinality': self.loss_cardinality,
            'entity': self.entity_loss,
            'ner_type': self.ner_type_loss,
            'ner_span': self.ner_span_loss,
            'ner_part': self.ner_part_loss,
            'head_tail_part': self.head_tail_part_loss,
            'ent_have_rel': self.ent_have_rel_loss,
            'head_tail_type': self.head_tail_type_loss
        }
        return loss_map[loss](outputs, targets, indices, num_tgt, **kwargs)

    @staticmethod
    def empty_targets(targets, filed):
        flag = True
        for target in targets:
            if len(target[filed]) != 0:
                flag = False
                break
        return flag

class ConsistencyLoss(nn.Module):
    """ Calculate the loss of consistency between Entity Set Generation and Relation Set Generation.
    """
    def __init__(self, type_consistency=True):
        """ Create the criterion.
        """
        super().__init__()
        self.matcher = Ent_HeadTail_Matcher()
        self.type_consistency = type_consistency

    def forward(self, ent_output, rel_output, ent_indices):
        ent_start_probs = [t[i] for t, (i, _) in zip(F.log_softmax(ent_output['ent_start_logits'], dim=-1), ent_indices)]
        ent_end_probs = [t[i] for t, (i, _) in zip(F.log_softmax(ent_output['ent_end_logits'], dim=-1), ent_indices)]
        ent_part_probs = [t[i] for t, (i, _) in zip(F.log_softmax(ent_output['ent_part_logits'].squeeze(-1), dim=-1), ent_indices)]
        ent_type_probs = [t[i] for t, (i, _) in zip(F.log_softmax(ent_output['ent_type_logits'][...,:-1], dim=-1), ent_indices)]

        valid_sample_index = [i for i, p in enumerate(ent_start_probs) if p.size(0) > 0]

        head_start_probs = F.log_softmax(rel_output['head_start_logits'], dim=-1)
        head_end_probs = F.log_softmax(rel_output['head_end_logits'], dim=-1)
        head_part_probs = F.log_softmax(rel_output['head_part_logits'].squeeze(-1), dim=-1)
        tail_start_probs = F.log_softmax(rel_output['tail_start_logits'], dim=-1)
        tail_end_probs = F.log_softmax(rel_output['tail_end_logits'], dim=-1)
        tail_part_probs = F.log_softmax(rel_output['tail_part_logits'].squeeze(-1), dim=-1)
        head_type_probs = F.log_softmax(rel_output['head_type_logits'], dim=-1)
        tail_type_probs = F.log_softmax(rel_output['tail_type_logits'], dim=-1)

        ent_head_match = self.matcher(ent_start_probs, ent_end_probs, ent_part_probs, head_start_probs, head_end_probs, head_part_probs, ent_type_probs, head_type_probs)
        ent_tail_match = self.matcher(ent_start_probs, ent_end_probs, ent_part_probs, tail_start_probs, tail_end_probs, tail_part_probs, ent_type_probs, tail_type_probs)

        head_selected_ent_start_probs = []
        head_selected_ent_end_probs = []
        head_selected_ent_part_probs = []
        tail_selected_ent_start_probs = []
        tail_selected_ent_end_probs = []
        tail_selected_ent_part_probs = []
        head_selected_ent_type_probs = []
        tail_selected_ent_type_probs = []
        for ent_start_prob, ent_end_prob, ent_part_prob, ent_type_prob, head_indices, tail_indices in zip(ent_start_probs, ent_end_probs, ent_part_probs, ent_type_probs, ent_head_match, ent_tail_match):
            if head_indices is None:
                continue
            head_selected_ent_start_probs.append(ent_start_prob[head_indices])
            head_selected_ent_end_probs.append(ent_end_prob[head_indices])
            head_selected_ent_part_probs.append(ent_part_prob[head_indices])
            tail_selected_ent_start_probs.append(ent_start_prob[tail_indices])
            tail_selected_ent_end_probs.append(ent_end_prob[tail_indices])
            tail_selected_ent_part_probs.append(ent_part_prob[tail_indices])

            head_selected_ent_type_probs.append(ent_type_prob[head_indices])
            tail_selected_ent_type_probs.append(ent_type_prob[tail_indices])

        if head_selected_ent_start_probs == []:
            return 0.0
        
        head_selected_ent_start_probs = torch.cat(head_selected_ent_start_probs)
        head_selected_ent_end_probs = torch.cat(head_selected_ent_end_probs)
        head_selected_ent_part_probs = torch.cat(head_selected_ent_part_probs)
        tail_selected_ent_start_probs = torch.cat(tail_selected_ent_start_probs)
        tail_selected_ent_end_probs = torch.cat(tail_selected_ent_end_probs)
        tail_selected_ent_part_probs = torch.cat(tail_selected_ent_part_probs)

        head_selected_ent_type_probs = torch.cat(head_selected_ent_type_probs)
        tail_selected_ent_type_probs = torch.cat(tail_selected_ent_type_probs)


        head_start_log_probs = head_start_probs[valid_sample_index].flatten(0, 1)
        head_end_log_probs = head_end_probs[valid_sample_index].flatten(0, 1)
        head_part_log_probs = head_part_probs[valid_sample_index].flatten(0, 1)
        tail_start_log_probs = tail_start_probs[valid_sample_index].flatten(0, 1)
        tail_end_log_probs = tail_end_probs[valid_sample_index].flatten(0, 1)
        tail_part_log_probs = tail_part_probs[valid_sample_index].flatten(0, 1)
        head_type_log_probs = head_type_probs[valid_sample_index].flatten(0, 1)
        tail_type_log_probs = tail_type_probs[valid_sample_index].flatten(0, 1)

        # print(head_selected_ent_start_probs.shape)
        # print(head_start_log_probs.shape)
        # print()

        ent_head_match_loss = F.kl_div(head_selected_ent_start_probs, head_start_log_probs, reduction='batchmean', log_target=True) + \
                                F.kl_div(head_selected_ent_end_probs, head_end_log_probs, reduction='batchmean', log_target=True)

        ent_head_match_loss += F.kl_div(head_selected_ent_part_probs, head_part_log_probs, reduction='batchmean', log_target=True)

        if self.type_consistency:
            ent_head_match_loss += F.kl_div(head_selected_ent_type_probs, head_type_log_probs, reduction='batchmean', log_target=True)

        ent_tail_match_loss = F.kl_div(tail_selected_ent_start_probs, tail_start_log_probs, reduction='batchmean', log_target=True) + \
                                F.kl_div(tail_selected_ent_end_probs, tail_end_log_probs, reduction='batchmean', log_target=True)

        ent_tail_match_loss += F.kl_div(tail_selected_ent_part_probs, tail_part_log_probs, reduction='batchmean', log_target=True)

        if self.type_consistency:
            ent_tail_match_loss += F.kl_div(tail_selected_ent_type_probs, tail_type_log_probs, reduction='batchmean', log_target=True)

        
        head_tail_start_probs = torch.cat((head_start_probs, tail_start_probs), dim=1)
        head_tail_end_probs = torch.cat((head_end_probs, tail_end_probs), dim=1)
        head_tail_part_probs = torch.cat((head_part_probs, tail_part_probs), dim=1)
        head_tail_type_probs = torch.cat((head_type_probs, tail_type_probs), dim=1)

        headtail_ent_match = self.matcher(head_tail_start_probs, head_tail_end_probs, head_tail_part_probs, ent_start_probs, ent_end_probs, ent_part_probs, head_tail_type_probs, ent_type_probs)

        ent_selected_headtail_start_probs = []
        ent_selected_headtail_end_probs = []
        ent_selected_headtail_part_probs = []
        ent_selected_headtail_type_probs = []
        for head_tail_start_prob, head_tail_end_prob, head_tail_part_prob, head_tail_type_prob, indices in zip(head_tail_start_probs, head_tail_end_probs, head_tail_part_probs, head_tail_type_probs, headtail_ent_match):
            if indices is None:
                continue
            ent_selected_headtail_start_probs.append(head_tail_start_prob[indices])
            ent_selected_headtail_end_probs.append(head_tail_end_prob[indices])
            ent_selected_headtail_part_probs.append(head_tail_part_prob[indices])
            ent_selected_headtail_type_probs.append(head_tail_type_prob[indices])

        ent_selected_headtail_start_probs = torch.cat(ent_selected_headtail_start_probs)
        ent_selected_headtail_end_probs = torch.cat(ent_selected_headtail_end_probs)
        ent_selected_headtail_part_probs = torch.cat(ent_selected_headtail_part_probs)
        ent_selected_headtail_type_probs = torch.cat(ent_selected_headtail_type_probs)

        ent_start_log_probs = torch.cat(ent_start_probs)
        ent_end_log_probs = torch.cat(ent_end_probs)
        ent_part_log_probs = torch.cat(ent_part_probs)
        ent_type_log_probs = torch.cat(ent_type_probs)

        # print(ent_selected_headtail_start_probs.shape)
        # print(ent_start_log_probs.shape)

        headtail_ent_match_loss = F.kl_div(ent_selected_headtail_start_probs, ent_start_log_probs, reduction='batchmean', log_target=True) + \
                                F.kl_div(ent_selected_headtail_end_probs, ent_end_log_probs, reduction='batchmean', log_target=True)

        headtail_ent_match_loss += F.kl_div(ent_selected_headtail_part_probs, ent_part_log_probs, reduction='batchmean', log_target=True)

        if self.type_consistency:
            headtail_ent_match_loss += F.kl_div(ent_selected_headtail_type_probs, ent_type_log_probs, reduction='batchmean', log_target=True)

        # print('ent_head_match_loss', ent_head_match_loss.item())
        # print('ent_tail_match_loss', ent_tail_match_loss.item())
        # print('headtail_ent_match_loss', headtail_ent_match_loss.item())

        return ent_head_match_loss + ent_tail_match_loss + headtail_ent_match_loss


class PIQNLoss(nn.Module):
    def __init__(self, num_classes, loss_weight, na_coef, losses, matcher, loss_type: str, boundary_softmax=False, hybrid=True):
        super().__init__()

        self.loss_type = loss_type

        if loss_type == 'RE':
            if boundary_softmax:
                self.criterion = SetCriterion(num_classes, loss_weight, na_coef, losses, matcher,
                                                num_ent_types=None, ner_na_coef=None, ner_losses=None, hybrid=hybrid)
            else:
                self.criterion = SetCriterion2(num_classes, loss_weight, na_coef, losses, matcher,
                                                num_ent_types=None, ner_na_coef=None, ner_losses=None)
        elif loss_type == 'NER':
            if boundary_softmax:
                self.criterion = SetCriterion(num_classes=None, loss_weight=loss_weight, na_coef=None, losses=None, matcher=matcher,
                                                num_ent_types=num_classes, ner_na_coef=na_coef, ner_losses=losses, hybrid=hybrid)
            else:
                self.criterion = SetCriterion2(num_classes=None, loss_weight=loss_weight, na_coef=None, losses=None, matcher=matcher,
                                                num_ent_types=num_classes, ner_na_coef=na_coef, ner_losses=losses)
        else:
            raise ValueError("Invalid loss_type.")

    def forward(self, outputs, targets, indices=None):
        losses = []

        # loss, ent_indices, rel_indices = self.criterion.forward(outputs[-1], targets, indices)
        # losses.append(loss)
        # indices = ent_indices if self.loss_type == 'NER' else rel_indices

        ent_indices_, rel_indices_ = [], []
        for out_dict in outputs:
            loss, ent_indices, rel_indices = self.criterion.forward(out_dict, targets, indices)
            losses.append(loss)
            ent_indices_.append(ent_indices)
            rel_indices_.append(rel_indices)

        return sum(losses) / len(losses), ent_indices_, rel_indices_


def loss_entID(p_entID, head_ent_scores, tail_ent_scores, targets, indices):
    """Compute the losses related to the entID of head entity or tail entity
    """
    idx = _get_src_permutation_idx(indices)
    selected_pred_head_entID = head_ent_scores[idx]
    selected_pred_tail_entID = tail_ent_scores[idx]

    if selected_pred_head_entID.size(0) == 0:
        return 0

    target_head_entID = torch.cat([t["head_entID"][i] for t, (_, i) in zip(targets, indices)])
    target_tail_entID = torch.cat([t["tail_entID"][i] for t, (_, i) in zip(targets, indices)])

    head_loss = F.cross_entropy(selected_pred_head_entID, target_head_entID)
    tail_loss = F.cross_entropy(selected_pred_tail_entID, target_tail_entID)

    # loss = head_loss + tail_loss

    selected_pred_entID = p_entID[idx]
    
    target_entID = torch.cat([t["rel_entID_labels"][i] for t, (_, i) in zip(targets, indices)]).unsqueeze(-1)
    entID_loss = binary_cross_entropy_(selected_pred_entID, target_entID)
    loss = entID_loss + head_loss + tail_loss
    
    return loss


def loss_relID(p_relID, p_relID_head, p_relID_tail, targets, indices):
    """Compute the BCE losses related to relIDs associated with each entity
    """
    if p_relID.size(2) == 0:
        return 0
    
    idx = _get_src_permutation_idx(indices)
    selected_pred_relID_head = p_relID_head[idx]
    selected_pred_relID_tail = p_relID_tail[idx]

    target_relID_head = torch.cat([t["relID_head_labels"][i] for t, (_, i) in zip(targets, indices)]).unsqueeze(-1)
    target_relID_tail = torch.cat([t["relID_tail_labels"][i] for t, (_, i) in zip(targets, indices)]).unsqueeze(-1)

    relID_head_loss = binary_cross_entropy_(selected_pred_relID_head, target_relID_head)
    relID_tail_loss = binary_cross_entropy_(selected_pred_relID_tail, target_relID_tail)

    # loss = relID_head_loss + relID_tail_loss

    selected_pred_relID = p_relID[idx]
    target_relID = torch.cat([t["relID_labels"][i] for t, (_, i) in zip(targets, indices)]).unsqueeze(-1)
    relID_loss = binary_cross_entropy_(selected_pred_relID, target_relID)
    loss = relID_loss + relID_head_loss + relID_tail_loss

    return loss

# indices: e.g. 20 20    20 20    0 0         format: [(tensor11, tensor12), (tensor21, tensor22), (tensor31, tensor32)]
def _get_src_permutation_idx(indices):
    # permute predictions following indices     e.g. tensor([0,...,0,1,...,1])  /  tensor([0,...,0,1,...,1,2,...,2])
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
