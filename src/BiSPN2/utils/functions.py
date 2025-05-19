import torch, collections, random, numpy

def filtration(prediction, relational_alphabet, remove_overlap=True):
    prediction = [(relational_alphabet.get_instance(ele.pred_rel), ele.head_mention, ele.tail_mention,
                    ele.head_start_index, ele.head_end_index,
                    ele.tail_start_index, ele.tail_end_index,
                    ele.rel_prob + 0.5*(ele.head_start_prob + ele.head_end_prob) + 0.5*(ele.tail_start_prob + ele.tail_end_prob)) for ele in prediction]

    prediction = sorted(prediction, key=lambda x: x[-1], reverse=True)

    res = []
    for pred in prediction:
        # if '[Inverse]_' in pred[0]:
        #     pred = (pred[0][len('[Inverse]_'):], pred[2], pred[1], pred[5], pred[6], pred[3], pred[4], pred[7])

        remove = False
        for ele in res:
            if remove_overlap and max(ele[3], pred[3]) <= min(ele[4], pred[4]) and \
                max(ele[5], pred[5]) <= min(ele[6], pred[6]):
                remove = True
            elif ele[3] == pred[3] and ele[4] == pred[4] and ele[5] == pred[5] and ele[6] == pred[6]:
                remove = True
        if not remove:
            res.append(pred)
    
    return res

def filtration_ent(prediction, ent_type_alphabet, remove_overlap=True, remove_same_span=True, span_in_triple=[]):
    entity_idxes, entities_ = [], {}
    
    for sent_id, entities in prediction.items():
        entities = [(entity_id,
            (ent_type_alphabet.get_instance(ele.pred_type), ele.start_index, ele.end_index,
            ele.entity_mention,
            (ele.type_prob + ele.start_prob + ele.end_prob)/3)
            ) for entity_id, ele in entities]

        entities = sorted(entities, key=lambda x: x[-1][-1], reverse=True)

        res = []
        entity_idx = []
        for entity_id, pred in entities:
            remove = False
            for ele in res:
                if remove_overlap and ele[0] == pred[0] and max(ele[1], pred[1]) <= min(ele[2], pred[2]):
                    if pred[1:3] not in span_in_triple:
                        remove = True
                elif remove_same_span and ele[1] == pred[1] and ele[2] == pred[2]:
                    remove = True
            if not remove:
                res.append(pred)
                entity_idx.append(entity_id)
        
        entity_idxes.append(entity_idx)
        entities_[sent_id] = res

    return entity_idxes, entities_

def filtration_ent_(prediction, ent_type_alphabet, remove_overlap=True, span_in_triple=[]):
    prediction = [(ent_type_alphabet.get_instance(ele.pred_type), ele.start_index, ele.end_index,
                    ele.type_prob + ele.start_prob + ele.end_prob) for _, ele in prediction]

    prediction = sorted(prediction, key=lambda x: x[-1], reverse=True)

    res = []
    for pred in prediction:
        remove = False
        for ele in res:
            if remove_overlap and ele[0] == pred[0] and max(ele[1], pred[1]) <= min(ele[2], pred[2]):
                if pred[1:3] not in span_in_triple:
                    remove = True
            elif ele[1] == pred[1] and ele[2] == pred[2]:
                remove = True
        if not remove:
            res.append(pred)

    return res


def get_span(start_indexes, end_indexes, start_probs, end_probs, subword_to_word, text, args, map_index):
    found = False
    for start_index in start_indexes:
        if start_probs[start_index] < args.boundary_ths:
            break
        if found:
            break
        for end_index in end_indexes:
            if end_probs[end_index] < args.boundary_ths:
                break
            if end_index < start_index or end_index - start_index + 1 > args.max_span_length:
                continue

            start_index_, end_index_ = start_index.item(), end_index.item()

            if subword_to_word is not None and (start_index_ not in subword_to_word or end_index_ not in subword_to_word):
                continue

            # English
            if isinstance(text, list):
                if subword_to_word is None:
                    start_index_mapped, end_index_mapped = start_index_, end_index_
                else:
                    start_index_mapped, end_index_mapped = subword_to_word[start_index_], subword_to_word[end_index_]
                mention = ' '.join(text[start_index_mapped: end_index_mapped+1])
            # Chinese
            else:
                start_index_mapped, end_index_mapped = subword_to_word[start_index_][0], subword_to_word[end_index_][-1]
                mention = text[start_index_mapped: end_index_mapped+1]
            
            start = start_index_mapped if map_index else start_index_
            end = end_index_mapped if map_index else end_index_
            start_prob = start_probs[start_index].item()
            end_prob = end_probs[end_index].item()
            found = True
            break

    if found:
        return (start, end, start_prob, end_prob, mention)
    else:
        return None
    
def generate_relation(pred_rel_logits, info, args):
    rel_probs, pred_rels = torch.max(pred_rel_logits.softmax(-1), dim=2)
    rel_probs = rel_probs.cpu().tolist()
    pred_rels = pred_rels.cpu().tolist()
    sent_idxes = info["sent_idx"]
    output = {}
    _Prediction = collections.namedtuple(
        "Prediction", ["pred_rel", "rel_prob"]
    )
    for (rel_prob, pred_rel, sent_idx) in zip(rel_probs, pred_rels, sent_idxes):
        output[sent_idx] = {}
        for triple_id in range(args.num_generated_triples):
            output[sent_idx][triple_id] = _Prediction(
                            pred_rel=pred_rel[triple_id],
                            rel_prob=rel_prob[triple_id])
    return output


def generate_triple(output, info, args, num_classes, map_index=False):
    _Pred_Triple = collections.namedtuple(
        "Pred_Triple", ["pred_rel", "rel_prob", "head_start_index", "head_end_index", "head_start_prob", "head_end_prob", "tail_start_index", "tail_end_index", "tail_start_prob", "tail_end_prob", "head_mention", "tail_mention"]
    )

    sent_idxes = info["sent_idx"]
    texts = info["text"]
    subword_to_words = info["subword_to_word"]
    ori_positions = info["ori_pos"]
    h_start_probs = output["head_start_logits"].softmax(-1)
    h_end_probs = output["head_end_logits"].softmax(-1)
    t_start_probs = output["tail_start_logits"].softmax(-1)
    t_end_probs = output["tail_end_logits"].softmax(-1)
    type_probs, pred_types = torch.max(output['pred_rel_logits'].softmax(-1), dim=2)
    type_probs = type_probs.cpu().tolist()
    pred_types = pred_types.cpu().tolist()

    output = {}
    for (h_start_prob, h_end_prob, t_start_prob, t_end_prob, type_prob, pred_type, sent_idx, text, subword_to_word, ori_pos) in \
    zip(h_start_probs, h_end_probs, t_start_probs, t_end_probs, type_probs, pred_types, sent_idxes, texts, subword_to_words, ori_positions):
        output[sent_idx] = {}
        K = min(h_start_prob.size(-1), args.n_best_size)
        for query_id in range(args.num_generated_triples):
            predictions = []
            # discard invalid pred
            if pred_type[query_id] == num_classes or type_prob[query_id] < args.rel_class_ths or \
            torch.max(h_start_prob[query_id]) < args.boundary_ths or torch.max(h_end_prob[query_id]) < args.boundary_ths or \
            torch.max(t_start_prob[query_id]) < args.boundary_ths or torch.max(t_end_prob[query_id]) < args.boundary_ths:
                output[sent_idx][query_id] = predictions
                continue

            h_start_indexes = torch.topk(h_start_prob[query_id], K).indices
            h_end_indexes = torch.topk(h_end_prob[query_id], K).indices
            t_start_indexes = torch.topk(t_start_prob[query_id], K).indices
            t_end_indexes = torch.topk(t_end_prob[query_id], K).indices

            try:
                head_start, head_end, head_start_prob, head_end_prob, head_mention = get_span(
                    h_start_indexes,
                    h_end_indexes,
                    h_start_prob[query_id],
                    h_end_prob[query_id],
                    subword_to_word, text, args, map_index
                )
                tail_start, tail_end, tail_start_prob, tail_end_prob, tail_mention = get_span(
                    t_start_indexes,
                    t_end_indexes,
                    t_start_prob[query_id],
                    t_end_prob[query_id],
                    subword_to_word, text, args, map_index
                )
                if ori_pos is not None:
                    head_start += ori_pos[0]
                    head_end += ori_pos[0]
                    tail_start += ori_pos[0]
                    tail_end += ori_pos[0]
            except:
                output[sent_idx][query_id] = predictions
                continue

            predictions.append(
                _Pred_Triple(
                    pred_rel=pred_type[query_id],
                    rel_prob=type_prob[query_id],
                    head_start_index=head_start,
                    head_end_index=head_end,
                    head_start_prob=head_start_prob,
                    head_end_prob=head_end_prob,
                    tail_start_index=tail_start,
                    tail_end_index=tail_end,
                    tail_start_prob=tail_start_prob,
                    tail_end_prob=tail_end_prob,
                    head_mention=head_mention,
                    tail_mention=tail_mention
                )
            )
            output[sent_idx][query_id] = predictions

    triples = {}
    for sent_idx in output:
        triples[sent_idx] = []
        for triple_id in range(args.num_generated_triples):
            pred = output[sent_idx][triple_id]
            if len(pred) > 0:
                triple = pred[0]
                triples[sent_idx].append(triple)
    # print('generate_triple')
    return triples

def generate_triple_with_etype(output, info, args, num_classes, num_etypes, relation_alphabet, map_index=False):
    _Pred_Triple = collections.namedtuple(
        "Pred_Triple", ["pred_rel", "rel_prob", "head_start_index", "head_end_index", "head_start_prob", "head_end_prob", "tail_start_index", "tail_end_index", "tail_start_prob", "tail_end_prob", "head_mention", "tail_mention", "head_type", "tail_type", "h_type_prob", "t_type_prob"]
    )

    sent_idxes = info["sent_idx"]
    texts = info["text"]
    subword_to_words = info["subword_to_word"]
    ori_positions = info["ori_pos"]
    categories = info["category"] if info["category"] else [None for _ in texts]
    h_start_probs = output["head_start_logits"].softmax(-1)
    h_end_probs = output["head_end_logits"].softmax(-1)
    t_start_probs = output["tail_start_logits"].softmax(-1)
    t_end_probs = output["tail_end_logits"].softmax(-1)
    type_probs, pred_types = torch.max(output['pred_rel_logits'].softmax(-1), dim=2)
    type_probs = type_probs.cpu().tolist()
    pred_types = pred_types.cpu().tolist()
    h_type_probs, h_pred_types = torch.max(output['head_type_logits'].softmax(-1), dim=2)
    h_type_probs = h_type_probs.cpu().tolist()
    h_pred_types = h_pred_types.cpu().tolist()
    t_type_probs, t_pred_types = torch.max(output['tail_type_logits'].softmax(-1), dim=2)
    t_type_probs = t_type_probs.cpu().tolist()
    t_pred_types = t_pred_types.cpu().tolist()

    output = {}
    for (h_start_prob, h_end_prob, t_start_prob, t_end_prob, type_prob, pred_type, sent_idx, text, subword_to_word, ori_pos, h_pred_type, t_pred_type, h_type_prob, t_type_prob, category) in \
    zip(h_start_probs, h_end_probs, t_start_probs, t_end_probs, type_probs, pred_types, sent_idxes, texts, subword_to_words, ori_positions, h_pred_types, t_pred_types, h_type_probs, t_type_probs, categories):
        output[sent_idx] = {}
        K = min(h_start_prob.size(-1), args.n_best_size)
        for query_id in range(args.num_generated_triples):
            predictions = []
            # discard invalid pred
            if pred_type[query_id] == num_classes or type_prob[query_id] < args.rel_class_ths or \
            torch.max(h_start_prob[query_id]) < args.boundary_ths or torch.max(h_end_prob[query_id]) < args.boundary_ths or \
            torch.max(t_start_prob[query_id]) < args.boundary_ths or torch.max(t_end_prob[query_id]) < args.boundary_ths:
                output[sent_idx][query_id] = predictions
                continue
            if h_pred_type[query_id] == num_etypes or t_pred_type[query_id] == num_etypes or \
            h_type_prob[query_id] < args.ent_class_ths or t_type_prob[query_id] < args.ent_class_ths:
                output[sent_idx][query_id] = predictions
                continue
            if category and category not in relation_alphabet.get_instance(pred_type[query_id]):
                output[sent_idx][query_id] = predictions
                continue

            h_start_indexes = torch.topk(h_start_prob[query_id], K).indices
            h_end_indexes = torch.topk(h_end_prob[query_id], K).indices
            t_start_indexes = torch.topk(t_start_prob[query_id], K).indices
            t_end_indexes = torch.topk(t_end_prob[query_id], K).indices

            try:
                head_start, head_end, head_start_prob, head_end_prob, head_mention = get_span(
                    h_start_indexes,
                    h_end_indexes,
                    h_start_prob[query_id],
                    h_end_prob[query_id],
                    subword_to_word, text, args, map_index
                )
                tail_start, tail_end, tail_start_prob, tail_end_prob, tail_mention = get_span(
                    t_start_indexes,
                    t_end_indexes,
                    t_start_prob[query_id],
                    t_end_prob[query_id],
                    subword_to_word, text, args, map_index
                )
                if ori_pos is not None:
                    head_start += ori_pos[0]
                    head_end += ori_pos[0]
                    tail_start += ori_pos[0]
                    tail_end += ori_pos[0]
            except:
                output[sent_idx][query_id] = predictions
                continue

            predictions.append(
                _Pred_Triple(
                    pred_rel=pred_type[query_id],
                    rel_prob=type_prob[query_id],
                    head_start_index=head_start,
                    head_end_index=head_end,
                    head_start_prob=head_start_prob,
                    head_end_prob=head_end_prob,
                    tail_start_index=tail_start,
                    tail_end_index=tail_end,
                    tail_start_prob=tail_start_prob,
                    tail_end_prob=tail_end_prob,
                    head_mention=head_mention,
                    tail_mention=tail_mention,
                    head_type=h_pred_type[query_id],
                    tail_type=t_pred_type[query_id],
                    h_type_prob=h_type_prob[query_id],
                    t_type_prob=t_type_prob[query_id]
                )
            )
            output[sent_idx][query_id] = predictions

    triples = {}
    for sent_idx in output:
        triples[sent_idx] = []
        for triple_id in range(args.num_generated_triples):
            pred = output[sent_idx][triple_id]
            if len(pred) > 0:
                triple = pred[0]
                triples[sent_idx].append(triple)
    # print('generate_triple')
    return triples

def formulate_gold(target, info, relational_alphabet, ent_type_alphabet):
    sent_idxes = info["sent_idx"]
    categories = info["category"]
    gold = {}
    for i in range(len(sent_idxes)):
        gold[sent_idxes[i]] = []
        try:
            category = categories[i]
        except:
            category = None
        for j in range(len(target[i]["relation"])):
            rel_type = relational_alphabet.get_instance(target[i]["relation"][j])
            if category and category not in rel_type:
                continue
            triple = (
                rel_type,
                (target[i]["head_start_index"][j], target[i]["head_end_index"][j],
                target[i]["head_mention"][j], ent_type_alphabet.get_instance(target[i]["head_type"][j])),
                (target[i]["tail_start_index"][j], target[i]["tail_end_index"][j],
                target[i]["tail_mention"][j], ent_type_alphabet.get_instance(target[i]["tail_type"][j])),
            )
            # print(triple)
            if triple not in gold[sent_idxes[i]]:
                gold[sent_idxes[i]].append(triple)
    return gold

def formulate_gold_(target, info, relational_alphabet):
    sent_idxes = info["sent_idx"]
    gold = {}
    for i in range(len(sent_idxes)):
        gold[sent_idxes[i]] = []
        for j in range(len(target[i]["relation"])):
            triple = (relational_alphabet.get_instance(target[i]["relation"][j]), target[i]["head_mention"][j], target[i]["tail_mention"][j])
            if triple not in gold[sent_idxes[i]]:
                gold[sent_idxes[i]].append(triple)
    # print('formulate_gold_', gold)
    return gold


def generate_ent_type(pred_type_logits, info, args):
    type_probs, pred_types = torch.max(pred_type_logits.softmax(-1), dim=2)
    type_probs = type_probs.cpu().tolist()
    pred_types = pred_types.cpu().tolist()
    sent_idxes = info["sent_idx"]
    output = {}
    _Prediction = collections.namedtuple(
        "Prediction", ["pred_type", "type_prob"]
    )
    for (type_prob, pred_type, sent_idx) in zip(type_probs, pred_types, sent_idxes):
        output[sent_idx] = {}
        for entity_id in range(args.entity_queries_num):
            output[sent_idx][entity_id] = _Prediction(
                            pred_type=pred_type[entity_id],
                            type_prob=type_prob[entity_id])
    return output

def generate_entity(output, info, args, num_classes, ent_type_alphabet, map_index=False, filtrate=True):
    _Pred_Entity = collections.namedtuple(
        "Pred_Triple", ["pred_type", "type_prob", "start_index", "end_index", "start_prob", "end_prob", "entity_mention"]
    )

    sent_idxes = info["sent_idx"]
    texts = info["text"]
    subword_to_words = info["subword_to_word"]
    ori_positions = info["ori_pos"] if info["ori_pos"] else [None for _ in texts]
    start_probs = output["ent_start_logits"].softmax(-1)
    end_probs = output["ent_end_logits"].softmax(-1)
    type_probs, pred_types = torch.max(output['ent_type_logits'].softmax(-1), dim=2)
    type_probs = type_probs.cpu().tolist()
    pred_types = pred_types.cpu().tolist()
            
    output = {}
    for (start_prob, end_prob, type_prob, pred_type, sent_idx, text, subword_to_word, ori_pos) in zip(start_probs, end_probs, type_probs, pred_types, sent_idxes, texts, subword_to_words, ori_positions):
        output[sent_idx] = {}
        K = min(start_prob.size(-1), args.n_best_size)
        for query_id in range(args.entity_queries_num):
            predictions = []
            # discard invalid pred
            if pred_type[query_id] == num_classes or type_prob[query_id] < args.ent_class_ths or \
            torch.max(start_prob[query_id]) < args.boundary_ths or torch.max(end_prob[query_id]) < args.boundary_ths:
                output[sent_idx][query_id] = predictions
                continue

            start_indexes = torch.topk(start_prob[query_id], K).indices
            end_indexes = torch.topk(end_prob[query_id], K).indices

            try:
                start, end, start_prob_, end_prob_, mention = get_span(
                    start_indexes,
                    end_indexes,
                    start_prob[query_id],
                    end_prob[query_id],
                    subword_to_word, text, args, map_index
                )
                if ori_pos is not None:
                    start += ori_pos[0]
                    end += ori_pos[0]
                predictions.append(
                    _Pred_Entity(
                        pred_type=pred_type[query_id], type_prob=type_prob[query_id],
                        start_index=start, end_index=end,
                        start_prob=start_prob_, end_prob=end_prob_,
                        entity_mention=mention
                    )
                )
            except:
                pass
            output[sent_idx][query_id] = predictions

    entities = {}
    for sent_idx in output:
        entities[sent_idx] = []
        for entity_id in range(args.entity_queries_num):
            pred = output[sent_idx][entity_id]
            if len(pred) > 0:
                ent = pred[0]
                entities[sent_idx].append((entity_id, ent))

    entity_idxes, entities = filtration_ent(entities, ent_type_alphabet, remove_overlap=filtrate, remove_same_span=filtrate)
    
    return entity_idxes, entities


def formulate_gold_ent(target, info, ent_type_alphabet):
    sent_idxes = info["sent_idx"]
    texts = info["text"]
    subword_to_words = info["subword_to_word"]
    gold = {}
    for sent_idx, text, subword_to_word, t in zip(sent_idxes, texts, subword_to_words, target):
        # print(target[i])
        gold[sent_idx] = set()
        for j in range(len(t["ent_type"])):
            label, start, end = ent_type_alphabet.get_instance(t["ent_type"][j]), t["ent_start_index"][j], t["ent_end_index"][j]
            # English
            if isinstance(text, list):
                mention = ' '.join(text[start: end+1])
            # Chinese
            else:
                start, end = subword_to_word[start][0], subword_to_word[end][-1]
                mention = text[start: end+1]
            ent = (label, start, end, mention)
            gold[sent_idx].add(ent)
    # print('formulate_gold_ent', gold)
    return gold

def generate_biorelex_preds(ent_output, rel_output, info, args, ent_type_alphabet, relation_alphabet):
    num_ent_type = ent_type_alphabet.size()
    num_rel_type = relation_alphabet.size()

    _, pred_entites = generate_entity(ent_output, info, args, num_ent_type, ent_type_alphabet, filtrate=False)
    pred_triples = generate_triple(rel_output, info, args, num_rel_type)

    preds = {}

    for id, text, subword_to_word in zip(pred_entites.keys(), info['text'], info['subword_to_word']):
        # print(text)
        entities = pred_entites[id]
        triples = pred_triples[id]

        triples = filtration(triples, relation_alphabet, remove_overlap=True)
        span_in_triple = [t[3:5] for t in triples] + [t[5:7] for t in triples]
        entities = filtration_ent_(entities, ent_type_alphabet, remove_overlap=True, span_in_triple=span_in_triple)

        mention_starts, mention_ends, mention_labels = [e[1] for e in entities], [e[2] for e in entities], [e[0] for e in entities]
        nb_mentions = len(mention_labels)

        # Build loc2label
        loc2label = {}
        for i in range(nb_mentions):
            loc2label[(mention_starts[i], mention_ends[i])] = mention_labels[i]

        # Initialize sample to be returned
        interactions, entities = [], []
        sample = {
            'id': id, 'text': text,
            'interactions': interactions, 'entities': entities
        }

        # Get clusters
        predicted_clusters, mention_to_predicted = [], {}
        for m_start, m_end in zip(mention_starts, mention_ends):
            if not (m_start, m_end) in mention_to_predicted:
                singleton_cluster = [(m_start, m_end)]
                predicted_clusters.append(singleton_cluster)
                mention_to_predicted[(m_start, m_end)] = singleton_cluster

        # Populate entities
        mention2entityid = {}
        for entityid, cluster in enumerate(predicted_clusters):
            assert len(cluster) == 1
            mentions, entity_labels = [], []
            for start_token, end_token in cluster:
                start_char, end_char = subword_to_word[start_token][0], subword_to_word[end_token][-1]+1
                mentions.append((start_char, end_char))
                # entity_labels.append(loc2label[(start_token, end_token)])
                mention2entityid[(start_token, end_token)] = entityid
            entity_name = text[start_char: end_char]
            entity_label = loc2label[(start_token, end_token)]

            # print(start_token, end_token)
            # print(entity_name)
            entities.append({
                'label': entity_label,
                'names': {
                    entity_name: {
                        'is_mentioned': True,
                        'mentions': mentions,
                    }
                },
                'is_mentioned': True
            })
        
        # Populate interactions
        for triple in triples:
            loci = triple[3: 5]
            locj = triple[5: 7]
            label = triple[0]
            if loci not in mention2entityid or locj not in mention2entityid:
                continue
            ent_i = mention2entityid[loci]
            ent_j = mention2entityid[locj]
            interactions.append({
                'participants': [ent_i, ent_j],
                'label': label
            })

        preds[id] = sample
    return preds


def generate_ade_preds(ent_output, rel_output, info, args, ent_type_alphabet, relation_alphabet):
    num_ent_type = ent_type_alphabet.size()
    num_rel_type = relation_alphabet.size()

    _, pred_entites = generate_entity(ent_output, info, args, num_ent_type, ent_type_alphabet, filtrate=False)
    pred_triples = generate_triple(rel_output, info, args, num_rel_type)

    preds = {}

    for id, text, subword_to_word in zip(pred_entites.keys(), info['text'], info['subword_to_word']):
        # print(text)
        entities = pred_entites[id]
        triples = pred_triples[id]

        triples = filtration(triples, relation_alphabet, remove_overlap=True)
        span_in_triple = [t[3:5] for t in triples] + [t[5:7] for t in triples]
        entities = filtration_ent_(entities, ent_type_alphabet, remove_overlap=True, span_in_triple=span_in_triple)
        entities = set([tuple(ele[:3]) for ele in entities])

        # Initialize sample to be returned
        triples_, entities_ = [], []
        sample = {
            'id': id, 'text': text,
            'interactions': triples_, 'entities': entities_
        }

        loc2label = dict()
        for ent_type, start_token, end_token in entities:
            start_char, end_char = subword_to_word[start_token][0], subword_to_word[end_token][-1]+1
            entities_.append((start_char, end_char, ent_type))
            loc2label[(start_token, end_token)] = ent_type
        
        # Populate interactions
        for triple in triples:
            loci = triple[3: 5]
            locj = triple[5: 7]
            label = triple[0]
            
            h_start, h_end = subword_to_word[loci[0]][0], subword_to_word[loci[1]][-1]+1
            t_start, t_end = subword_to_word[locj[0]][0], subword_to_word[locj[1]][-1]+1
            triples_.append({
                'head': text[h_start: h_end],
                'head_type': 'Adverse-Effect',
                'tail': text[t_start: t_end],
                'tail_type': 'Drug',
                'type': label
            })

        preds[id] = sample
    return preds


def set_project_entities(project_ent, ent_indices=None, entID2entities=None, entity_idxes=None, output_mask=True, reverse_mask=True):
        project_ent_ = []
        batch_size, num_gen_entities = project_ent.size(0), project_ent.size(1)
        
        if ent_indices is not None:
            # print([len(indices[0]) for indices in ent_indices])
            num_entities = [len(x) if x is not None else 0 for x in entID2entities]
            max_num_entities = max(num_entities)

            for i, (src, tgt) in enumerate(ent_indices):
                num_ent = num_entities[i]
                batch_idx = src.new_full((max_num_entities,), fill_value=i)

                if num_ent == 0:
                    assert len(tgt) == 0
                    src_idx = torch.arange(max_num_entities, device=src.device)

                else:
                    _, sorted_indices = torch.sort(tgt, dim=0)

                    chosen_src = []
                    entID2entities_ = entID2entities[i]

                    for entID in range(num_ent):
                        # chosen_ent = entID2entities_[entID][torch.randint(len(entID2entities_[entID]), (1,))]
                        chosen_ent = random.choice(entID2entities_[entID])
                        # chosen_ent = numpy.random.choice(entID2entities_[entID])
                        chosen_src.append(src[sorted_indices[chosen_ent]])

                    chosen_src = torch.stack(chosen_src)

                    # for entities_ in entID2entities_.values():

                    pad_full_idx = [idx for idx in range(num_gen_entities) if idx not in chosen_src]
                    k = (max_num_entities-num_ent) // len(pad_full_idx)
                    v = (max_num_entities-num_ent) % len(pad_full_idx)
                    pad_full_idx = pad_full_idx * k + pad_full_idx[:v]
                    pad_full_idx = torch.LongTensor(pad_full_idx, device=src.device)

                    src_idx = torch.cat([chosen_src, pad_full_idx])

                tmp = project_ent[batch_idx, src_idx]
                # print(tmp.shape)
                project_ent_.append(tmp)
        
        else:
            num_entities = [len(entity_idx) for entity_idx in entity_idxes]
            max_num_entities = max(num_entities)
            for i, entity_idx in enumerate(entity_idxes):
                # print(i, entity_idx)
                pad_full_idx = [idx for idx in range(num_gen_entities) if idx not in entity_idx]
                pad_full_idx = pad_full_idx[:1] * (max_num_entities-len(entity_idx))

                entity_idx_ = entity_idx + pad_full_idx
                project_ent_.append(project_ent[i, entity_idx_])
        
        project_ent_ = torch.stack(project_ent_)
        entities_pad_mask = None
        if output_mask:
            if reverse_mask:
                mask = torch.arange(max_num_entities).expand(batch_size, -1) >= torch.LongTensor(num_entities).unsqueeze(1)
            else:
                mask = (torch.arange(max_num_entities).expand(batch_size, -1) < torch.LongTensor(num_entities).unsqueeze(1)).float()
            entities_pad_mask = mask.to(project_ent_.device)

        return project_ent_, entities_pad_mask
