from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
from models.set_criterion import PIQNLoss, ConsistencyLoss, loss_relID, loss_entID
from models.encoder_decoder import EntityAwareBertConfig, SeqEncoder, MatchPredictor
from utils.functions import generate_triple, generate_entity, generate_biorelex_preds, generate_ade_preds, set_project_entities
from utils.data import padded_stack


class SetPred4RE(nn.Module):

    def __init__(self, args, num_classes, num_ent_types, RE=True):
        super(SetPred4RE, self).__init__()
        self.args = args
        self.RE = RE

        config = EntityAwareBertConfig.from_pretrained(args.bert_directory, hidden_dropout_prob=args.dropout, num_generated_triples=args.num_generated_triples,
                    entity_queries_num = args.entity_queries_num, mask_ent2tok = args.mask_ent2tok,  mask_tok2ent = args.mask_tok2ent, mask_ent2ent = args.mask_ent2ent, mask_entself = args.mask_entself, entity_aware_attention = args.entity_aware_attention, entity_aware_selfout = args.entity_aware_selfout, entity_aware_intermediate = args.entity_aware_intermediate, entity_aware_output = args.entity_aware_output, use_entity_pos = args.use_entity_pos, use_entity_common_embedding = args.use_entity_common_embedding)

        self.encoder = SeqEncoder.from_pretrained(args.bert_directory,
                                                config = config,
                                                fix_bert_embeddings=args.fix_bert_embeddings,
                                                relation_type_count = num_classes,
                                                # piqn model parameters
                                                entity_type_count = num_ent_types,
                                                prop_drop = args.prop_drop,
                                                pos_size = args.pos_size,
                                                char_lstm_layers = args.char_lstm_layers,
                                                char_lstm_drop = args.char_lstm_drop, 
                                                char_size = args.char_size, 
                                                use_glove = args.use_glove, 
                                                use_pos = args.use_pos, 
                                                use_char_lstm = args.use_char_lstm,
                                                lstm_layers = args.lstm_layers,
                                                pool_type = args.pool_type,
                                                word_mask_tok2ent = args.word_mask_tok2ent,
                                                word_mask_ent2tok = args.word_mask_ent2tok,
                                                word_mask_ent2ent = args.word_mask_ent2ent,
                                                word_mask_entself = args.word_mask_entself,
                                                share_query_pos = args.share_query_pos,
                                                use_token_level_encoder = args.use_token_level_encoder,
                                                num_token_ent_rel_layer = args.num_token_ent_rel_layer,
                                                num_token_ent_layer = args.num_token_ent_layer,
                                                num_token_rel_layer = args.num_token_rel_layer,
                                                num_token_head_tail_layer = args.num_token_head_tail_layer,
                                                use_entity_attention = args.use_entity_attention,
                                                use_aux_loss =  args.use_aux_loss,
                                                use_lstm = args.use_lstm,
                                                RE = RE)

        self.num_classes = num_classes
        self.num_ent_types = num_ent_types

        self.re_criterion = PIQNLoss(num_classes, loss_weight=self.get_loss_weight(args), na_coef=args.na_rel_coef, losses=["entity", "relation", "head_tail_part", "head_tail_type"],
                                        matcher=args.matcher, loss_type='RE', boundary_softmax=True, hybrid=args.hybrid)
        
        if self.RE:
            self.ner_criterion = PIQNLoss(num_ent_types, loss_weight=self.get_loss_weight(args), na_coef=args.na_ent_coef, losses=["ner_type", "ner_span", "ner_part", "ent_have_rel"],
                                            matcher=args.matcher, loss_type='NER', boundary_softmax=False)
        else:
            self.ner_criterion = PIQNLoss(num_ent_types, loss_weight=self.get_loss_weight(args), na_coef=args.na_ent_coef, losses=["ner_type", "ner_span", "ner_part"],
                                            matcher=args.matcher, loss_type='NER', boundary_softmax=False)

        self.consistency_criterion = ConsistencyLoss(type_consistency=args.head_tail_type_loss_weight>0)

        if self.RE:
            self.head_entity_predictor = MatchPredictor(config)
            self.tail_entity_predictor = MatchPredictor(config)
            self.triple_predictor = MatchPredictor(config, squeeze=False)
            self.triple_head_predictor = MatchPredictor(config, squeeze=False)
            self.triple_tail_predictor = MatchPredictor(config, squeeze=False)
            self.entity_predictor = MatchPredictor(config, squeeze=False)

        self.device_id = 0

    def set_device_id(self, device_id):
        self.device_id = device_id
        
    def set_RE(self):
        if self.RE == True:
            return
        self.RE = True
        self.ner_criterion = PIQNLoss(self.num_ent_types, loss_weight=self.get_loss_weight(self.args), na_coef=self.args.na_ent_coef, losses=["ner_type", "ner_span", "ner_part", "ent_have_rel"],
                                            matcher=self.args.matcher, loss_type='NER', boundary_softmax=False)
        self.encoder.set_RE(self.encoder.config)
        self.head_entity_predictor = MatchPredictor(self.encoder.config)
        self.tail_entity_predictor = MatchPredictor(self.encoder.config)
        self.triple_predictor = MatchPredictor(self.encoder.config, squeeze=False)
        self.triple_head_predictor = MatchPredictor(self.encoder.config, squeeze=False)
        self.triple_tail_predictor = MatchPredictor(self.encoder.config, squeeze=False)
        self.entity_predictor = MatchPredictor(self.encoder.config, squeeze=False)

    def forward(self, input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, targets=None, info=None, epoch=None):

        h_token, ent_outputs, rel_outputs, projects = self.encoder(input_ids, attention_mask, seg_encoding, context2token_masks, token_masks)

        if targets is not None:
            if epoch == self.args.start_ent_have_rel_epoch:
                self.ner_criterion.criterion.loss_weight['ent_have_rel'] = self.args.ent_have_rel_loss_weight
            if epoch == self.args.stop_ent_have_rel_epoch:
                self.ner_criterion.criterion.loss_weight['ent_have_rel'] = 0
            
            ner_loss, ent_indices_, _ = self.ner_criterion(ent_outputs[-2:], targets)

            if self.RE:
                re_loss, _, rel_indices_ = self.re_criterion(rel_outputs[-2:], targets)
                loss = re_loss + ner_loss

                compute_link_loss = True
                if compute_link_loss:
                    entID_loss, relID_loss = 0, 0
                    for (project_ent, project_rel, project_head, project_tail), ent_indices, rel_indices in zip(projects, ent_indices_, rel_indices_):

                        project_ent_, entities_pad_mask = set_project_entities(project_ent, ent_indices=ent_indices, entID2entities=info['entID2entities'])

                        p_entID, head_ent_scores, tail_ent_scores = self.get_head_tail_scores(project_ent_, project_rel, project_head, project_tail, entities_pad_mask)
                        entID_loss += loss_entID(p_entID, head_ent_scores, tail_ent_scores, targets, rel_indices)

                        project_rel_, triples_pad_mask = set_project_entities(project_rel, ent_indices=rel_indices, entID2entities=info['relID2triples'])
                        project_head_, _ = set_project_entities(project_head, ent_indices=rel_indices, entID2entities=info['relID2triples'], output_mask=False)
                        project_tail_, _ = set_project_entities(project_tail, ent_indices=rel_indices, entID2entities=info['relID2triples'], output_mask=False)

                        p_relID, p_relID_head, p_relID_tail = self.get_p_rel(project_rel_, project_head_, project_tail_, project_ent, triples_pad_mask)
                        relID_loss += loss_relID(p_relID, p_relID_head, p_relID_tail, targets, ent_indices)

                    entID_loss /= len(projects)
                    relID_loss /= len(projects)

                    loss += entID_loss * self.args.head_tail_entID_loss_weight + relID_loss * self.args.relID_loss_weight

                if epoch >= self.args.start_consistency_epoch:
                    consistency_loss = self.consistency_criterion(ent_outputs[-1], rel_outputs[-1], ent_indices_[-1])

                    loss += consistency_loss * self.args.consistency_loss_weight
                    
                # print(flush=True)
            else:
                loss = ner_loss

            return loss, rel_outputs[-1] if self.RE else None, ent_outputs[-1]
        else:
            return rel_outputs[-1] if self.RE else None, ent_outputs[-1]

    def get_head_tail_scores(self, project_ent, project_rel, project_head, project_tail, entities_pad_mask):
        _, p_entID = self.entity_predictor.forward(project_ent, project_rel, entities_pad_mask)
        head_ent_scores, _ = self.head_entity_predictor.forward(project_ent, project_head, entities_pad_mask)
        tail_ent_scores, _ = self.tail_entity_predictor.forward(project_ent, project_tail, entities_pad_mask)
        return p_entID, head_ent_scores, tail_ent_scores
    
    def get_p_rel(self, project_rel, project_head, project_tail, project_ent, triples_pad_mask):
        _, p_relID = self.triple_predictor.forward(project_rel, project_ent, triples_pad_mask)
        _, p_relID_head = self.triple_head_predictor.forward(project_head, project_ent, triples_pad_mask)
        _, p_relID_tail = self.triple_tail_predictor.forward(project_tail, project_ent, triples_pad_mask)
        return p_relID, p_relID_head, p_relID_tail

    def default_predict(self, input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, info, ent_type_alphabet):
        with torch.no_grad():
            rel_output, ent_output = self.forward(input_ids, attention_mask, seg_encoding, context2token_masks, token_masks)
            entity_idxes, entities = generate_entity(ent_output, info, self.args, self.num_ent_types, ent_type_alphabet, map_index=True)
            pred_triple = None
            if self.RE:
                pred_triple = generate_triple(rel_output, info, self.args, self.num_classes, map_index=True)
                # project_ent_, entities_pad_mask = set_project_entities(project_ent, entity_idxes=entity_idxes)
                # head_ent_scores, tail_ent_scores = self.get_head_tail_scores(project_ent_, project_head, project_tail, entities_pad_mask)
                # pred_triple = generate_triple(rel_output, info, self.args, self.num_classes, entities, head_ent_scores, tail_ent_scores)
        return pred_triple, entities

    def biorelex_predict(self, input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, info, ent_type_alphabet, relation_alphabet):
        with torch.no_grad():
            rel_output, ent_output = self.forward(input_ids, attention_mask, seg_encoding, context2token_masks, token_masks)
            preds = generate_biorelex_preds(ent_output, rel_output, info, self.args, ent_type_alphabet, relation_alphabet)
        return preds

    def ade_predict(self, input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, info, ent_type_alphabet, relation_alphabet):
        with torch.no_grad():
            rel_output, ent_output = self.forward(input_ids, attention_mask, seg_encoding, context2token_masks, token_masks)
            preds = generate_ade_preds(ent_output, rel_output, info, self.args, ent_type_alphabet, relation_alphabet)
        return preds

    def batchify(self, batch_list, is_test=False):
        batch_size = len(batch_list)
        sent_idx = [ele[0] for ele in batch_list]
        sent_ids = [ele[1] for ele in batch_list]
        targets = [ele[2] for ele in batch_list]
        bio_labels = [ele[3] for ele in batch_list]
        text = [ele[4] for ele in batch_list]
        subword_to_word = [ele[5] for ele in batch_list]
        seg_encoding = [ele[6] for ele in batch_list]
        context2token_masks = [ele[7] for ele in batch_list]
        token_masks = [ele[8] for ele in batch_list]
        inst = [ele[9] for ele in batch_list] if self.args.dataset_name in ['biorelex', 'ade'] and is_test else None
        entID2entities = [ele[9] for ele in batch_list] if not is_test else None
        relID2triples = [ele[10] for ele in batch_list] if not is_test else None
        max_num_entity = max([len(x) for x in entID2entities]) if not is_test else None
        max_num_triple = max([len(x) for x in relID2triples]) if not is_test else None
        # print(relID2triples)
        # print(max_num_triple)
        ori_pos = [ele[11] for ele in batch_list] if any(x in self.args.dataset_name for x in ['child', 'cancer']) else None
        
        sent_lens = list(map(len, sent_ids))
        max_sent_len = max(sent_lens)
        input_ids = torch.zeros((batch_size, max_sent_len), requires_grad=False).long()
        attention_mask = torch.zeros((batch_size, max_sent_len), requires_grad=False, dtype=torch.float32)
        for idx, (seq, seqlen) in enumerate(zip(sent_ids, sent_lens)):
            input_ids[idx, :seqlen] = torch.LongTensor(seq)
            attention_mask[idx, :seqlen] = 1


        seg_encoding = [torch.tensor(x, dtype=torch.long) for x in seg_encoding]
        seg_encoding = padded_stack(seg_encoding)
        if context2token_masks[0] is not None:
            context2token_masks = padded_stack(context2token_masks)
            max_sent_len = context2token_masks.size(1)
        else:
            context2token_masks = None
        token_masks = [torch.tensor(x, dtype=torch.bool) for x in token_masks]
        token_masks = padded_stack(token_masks)


        targets_ = copy.deepcopy(targets)
        if not is_test:
            for t in targets_:
                for k in t.keys():
                    max_num = max_sent_len
                    if k in ['relID_labels', 'relID_head_labels', 'relID_tail_labels']:
                        max_num = max_num_triple
                    elif k in ['rel_entID_labels']:
                        max_num = max_num_entity

                    if 'labels' in k:
                        padded = []
                        for labels in t[k]:
                            padded.append(labels + [0] * (max_num - len(labels)))
                        t[k] = padded

        if self.args.use_gpu:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            seg_encoding = seg_encoding.cuda()
            context2token_masks = context2token_masks.cuda() if context2token_masks is not None else None
            token_masks = token_masks.cuda()

            if not is_test:
                targets_ = [{k: torch.tensor(v, dtype=(torch.float if 'labels' in k else torch.long), requires_grad=False).cuda()
                for k, v in t.items() if 'mention' not in k} for t in targets_]
            
        else:
            if not is_test:
                targets_ = [{k: torch.tensor(v, dtype=(torch.float if 'labels' in k else torch.long), requires_grad=False)
                for k, v in t.items() if 'mention' not in k} for t in targets_]
                
        info = {"seq_len": sent_lens, "sent_idx": sent_idx, "text": text, "subword_to_word": subword_to_word, "inst": inst, "ori_pos": ori_pos}
        info.update({'entID2entities': entID2entities})
        info.update({'relID2triples': relID2triples})

        return input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, targets_, bio_labels, info

    def get_CL_sample(self, projections, target_labels, indices, type_embeddings):
        idx = self._get_src_permutation_idx(indices)
        projections = projections[idx]
        target_labels = torch.cat([t[i] for t, (_, i) in zip(target_labels, indices)])
        
        instances = [[] for type_embedding in type_embeddings]
        min_instance = projections.size(0)
        for projection, target_label in zip(projections, target_labels):
            instances[target_label].append(projection)

        for instance in instances:
            num = len(instance)
            if num > 1:
                min_instance = min(min_instance, num)

        instances_ = []
        labels_ = []
        for i, instance in enumerate(instances):
            num = len(instance)
            if num > 1:
                # random.shuffle(instance)
                # instances_.append(torch.stack(instance[:min_instance]))
                # labels_.append(i)

                num_split = num // min_instance
                splits = [instance[k*min_instance: (k+1)*min_instance] for k in range(num_split)]
                for split in splits:
                    instances_.append(torch.stack(split))
                    labels_.append(i)

        instances_ = torch.stack(instances_)
        instances_ = F.normalize(instances_, dim=-1)
        labels_ = torch.LongTensor(labels_)
        # print(target_labels)
        # print('instances_.shape', instances_.shape)
        
        return instances_, labels_
        
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def get_loss_weight(args):
        return {"relation": args.rel_loss_weight,
                "head_tail_entID": args.head_tail_entID_loss_weight,
                "head_entity": args.head_ent_loss_weight,
                "tail_entity": args.tail_ent_loss_weight,
                "ent_type": args.ent_type_loss_weight,
                "ent_span": args.ent_span_loss_weight,
                "ent_part": args.ent_part_loss_weight,
                "head_part": args.head_part_loss_weight,
                "tail_part": args.tail_part_loss_weight,
                "head_tail_type": args.head_tail_type_loss_weight,
                "ent_have_rel": 0,
            }
