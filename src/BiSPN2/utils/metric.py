import json
import numpy as np


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
            if remove_overlap and max(ele[3], pred[3]) <= min(ele[4], pred[4]) \
                and max(ele[5], pred[5]) <= min(ele[6], pred[6]):
                remove = True
            elif ele[3] == pred[3] and ele[4] == pred[4] and ele[5] == pred[5] and ele[6] == pred[6]:
                remove = True
        if not remove:
            res.append(pred)
    
    return res  



def metric_(pred, gold,  list_text, relational_alphabet, log_fn, print_pred, remove_overlap=True):
    assert pred.keys() == gold.keys()
    gold_num = 0
    rel_num = 0
    ent_num = 0
    right_num = 0
    pred_num = 0
    pred_ent_num, gold_ent_num = 0, 0
    pred_rel_num, gold_rel_num = 0, 0

    if print_pred:
        log_file = open(log_fn + '.rel', 'w', encoding='utf-8')
        
    for i, sent_idx in enumerate(pred):
        if print_pred:
            print("Sample-ID:", sent_idx, file=log_file)
            print(list_text[i], file=log_file)

        gold_num += len(gold[sent_idx])
        prediction = filtration(pred[sent_idx], relational_alphabet, remove_overlap=remove_overlap)
        prediction = set([tuple(ele[:3]) for ele in prediction])
        pred_num += len(prediction)
        gold_rel_set = set([e[0] for e in gold[sent_idx]])
        pred_rel_set = set([ele[0] for ele in prediction])
        gold_ent_set = set([e[1:] for e in gold[sent_idx]])
        pred_ent_set = set([ele[1:] for ele in prediction])
        for ele in prediction:
            if ele in gold[sent_idx]:
                right_num += 1

            # label, head, tail = ele
            # for ele_ in gold[sent_idx]:
            #     label_, head_, tail_ = ele_
            #     if label==label_ and LCS(head, head_) >= max(len(head), len(head_))/2 and LCS(tail, tail_) >= max(len(tail), len(tail_))/2:
            #         right_num += 1
            #         break

        rel_num += len(pred_rel_set & gold_rel_set)
        ent_num += len(pred_ent_set & gold_ent_set)
        pred_ent_num += len(pred_ent_set)
        pred_rel_num += len(pred_rel_set)
        gold_ent_num += len(gold_ent_set)
        gold_rel_num += len(gold_rel_set)
        if print_pred:
            print("Gold:", file=log_file)
            print([e[:3] for e in gold[sent_idx]], file=log_file)
            print("Pred:", file=log_file)
            print([e[:3] for e in prediction], file=log_file)
            print('', file=log_file)

    if pred_num == 0:
        precision = -1
        r_p = -1
        e_p = -1
    else:
        precision = (right_num + 0.0) / pred_num
        e_p = (ent_num + 0.0) / pred_ent_num
        r_p = (rel_num + 0.0) / pred_rel_num

    if gold_num == 0:
        recall = -1
        r_r = -1
        e_r = -1
    else:
        recall = (right_num + 0.0) / gold_num
        e_r = ent_num / gold_ent_num
        r_r = rel_num / gold_rel_num

    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    if (e_p == -1) or (e_r == -1) or (e_p + e_r) <= 0.:
        e_f = -1
    else:
        e_f = 2 * e_r * e_p / (e_p + e_r)

    if (r_p == -1) or (r_r == -1) or (r_p + r_r) <= 0.:
        r_f = -1
    else:
        r_f = 2 * r_p * r_r / (r_r + r_p)

    print("gold_num = ", gold_num, " pred_num = ", pred_num, " right_num = ", right_num)
    print("precision = ", precision, " recall = ", recall, " f1_value = ", f_measure)
    print("rel_precision = ", r_p, " rel_recall = ", r_r, " rel_f1_value = ", r_f)
    print("head_tail_precision = ", e_p, " head_tail_recall = ", e_r, " head_tail_f1 = ", e_f)

    if print_pred:
        print("gold_num = ", gold_num, " pred_num = ", pred_num, " right_num = ", right_num, file=log_file)
        print("precision = ", precision, " recall = ", recall, " f1_value = ", f_measure, file=log_file)
        print("rel_precision = ", r_p, " rel_recall = ", r_r, " rel_f1_value = ", r_f, file=log_file)
        print("ent_precision = ", e_p, " ent_recall = ", e_r, " ent_f1_value = ", e_f, file=log_file)
    return {"precision": precision, "recall": recall, "f1": f_measure}


def filtration_with_etype(prediction, relational_alphabet, ent_type_alphabet, remove_overlap=True):
    # for ele in prediction:
    #     print(f'rel_prob:{ele.rel_prob} tail_start_prob:{ele.tail_start_prob} tail_end_prob:{ele.tail_end_prob} t_type_prob:{ele.t_type_prob}')
    # print()

    prediction = [(relational_alphabet.get_instance(e.pred_rel), e.head_mention, e.tail_mention, 
                    e.head_start_index, e.head_end_index,
                    e.tail_start_index, e.tail_end_index,
                    ent_type_alphabet.get_instance(e.head_type), ent_type_alphabet.get_instance(e.tail_type),
                    e.rel_prob + 0.5*(e.head_start_prob + e.head_end_prob) + 0.5*(e.tail_start_prob + e.tail_end_prob) + 0.5*(e.h_type_prob + e.t_type_prob)
                    ) for e in prediction]

    prediction = sorted(prediction, key=lambda x: x[-1], reverse=True)

    res = []
    for pred in prediction:
        # if '[Inverse]_' in pred[0]:
        #     pred = (pred[0][len('[Inverse]_'):], pred[2], pred[1], pred[5], pred[6], pred[3], pred[4], pred[7])

        remove = False
        for ele in res:
            if remove_overlap and max(ele[3], pred[3]) <= min(ele[4], pred[4]) \
                and max(ele[5], pred[5]) <= min(ele[6], pred[6]):
                remove = True
            elif ele[3] == pred[3] and ele[4] == pred[4] and ele[5] == pred[5] and ele[6] == pred[6]:
                remove = True
        if not remove:
            res.append(pred)
    
    return res  





def overlap(span1, span2):
    return not (span1[1] <= span2[0] or span2[1] <= span1[0])

def LCS(text1: str, text2: str) -> int:
    n = len(text1)
    m = len(text2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])

    return dp[n][m]

def calculate_iou(pred_start, pred_end, true_start, true_end):
    """计算 IOU"""
    overlap_length = max(0, min(pred_end, true_end) - max(pred_start, true_start))
    union_length = (pred_end - pred_start) + (true_end - true_start) - overlap_length
    iou = overlap_length / union_length if union_length > 0 else 0
    return iou

def sample_score(pred_entities, true_entities):
    """
    计算单个样本的度量值
    :param pred_entities: 预测实体列表，每个实体为 (start, end, type)
    :param true_entities: 真实实体列表，每个实体为 (start, end, type)
    :param association_matrix: 实体类型关联度矩阵
    :return: 样本的度量值（0 到 1 之间）
    """
    n = len(pred_entities)
    m = len(true_entities)
    if n == 0 or m == 0:
        return 0.0, 0.0  # 如果没有预测或真实实体，得分为 0

    # 初始化加权 IOU 矩阵
    weighted_ious_p2t = np.zeros((n, m))
    weighted_ious_t2p = np.zeros((m, n))

    # 计算每对实体的加权 IOU
    for i, (p_type, p_start, p_end, _) in enumerate(pred_entities):
        for j, (t_type, t_start, t_end, _) in enumerate(true_entities):
            iou1 = calculate_iou(p_start, p_end, t_start, t_end)
            #weighted_ious_p2t[i, j] = iou1 * entityLinkMatrix[p_type][t_type]
            if p_type != t_type:
                continue
            weighted_ious_p2t[i, j] = iou1

    for i, (t_type, t_start, t_end, _) in enumerate(true_entities):
        for j, (p_type, p_start, p_end, _) in enumerate(pred_entities):
            iou2 = calculate_iou(p_start, p_end, t_start, t_end)
            #weighted_ious_t2p[i, j] = iou2 * entityLinkMatrix[t_type][p_type]
            if p_type != t_type:
                continue
            weighted_ious_t2p[i, j] = iou2

    matches_p2t = np.max(weighted_ious_p2t, axis = 1)
    matches_t2p = np.max(weighted_ious_t2p, axis = 1)
                

    # 计算样本的最终度量值
    return sum(matches_p2t), sum(matches_t2p)

# 对所有batch的样本进行最后的统计计算
def ent_metric_overlap(pred, gold, list_text, log_fn, print_pred):
    assert pred.keys() == gold.keys()
    gold_num = 0
    right_num_pred = 0
    right_num_gold = 0
    pred_num = 0
    
    samples = []
    if print_pred:
        log_file = open(log_fn + '.ent', 'w', encoding='utf-8')
        
    for i, sent_idx in enumerate(pred):
        # if print_pred:
        #     print("Sample-ID:", sent_idx, file=log_file)
        #     print(list_text[i], file=log_file)

        gold_num += len(gold[sent_idx])
        prediction = pred[sent_idx]
        #prediction = set([ele[:4] for ele in prediction])
        prediction = list([ele[:4] for ele in prediction])
        pred_num += len(prediction)
        '''for ele in prediction:
            if ele in gold[sent_idx]:
                right_num += 1'''
        match_p2t, match_t2p = sample_score(prediction, gold[sent_idx])
        right_num_pred += match_p2t
        right_num_gold += match_t2p
        

            # label, start, end, mention = ele
            # for ele_ in gold[sent_idx]:
            #     label_, start_, end_, mention_ = ele_
            #     if label == label_ and LCS(mention, mention_) >= max(len(mention), len(mention_))/2:
            #         right_num += 1
            #         break

        # if print_pred:
        prediction = sorted(list(prediction), key=lambda x: x[1:3])
        gt = sorted(list(gold[sent_idx]), key=lambda x: x[1:3])
        sample = {
            "doc_key": sent_idx,
            "sent_id": i,
            "text": list_text[i],
            "entities": prediction,
            "gold": gt
        }
        samples.append(sample)
        # print("Gold:", file=log_file)
        # gold_entities = sorted(list(gold[sent_idx]), key=lambda x: x[1:3])
        # tmp = json.dumps(gold_entities, ensure_ascii=False)
        # print(tmp, file=log_file)
        # print("Pred:", file=log_file)
        # pred_entities = sorted(list(prediction), key=lambda x: x[1:3])
        # tmp = json.dumps(pred_entities, ensure_ascii=False)
        # print(tmp, file=log_file)
        # print('', file=log_file)

    if pred_num == 0:
        precision = -1
    else:
        precision = (right_num_pred + 0.0) / pred_num

    if gold_num == 0:
        recall = -1
    else:
        recall = (right_num_gold + 0.0) / gold_num

    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    print("-------------Overlap----------------")
    print("num gold entity = ", gold_num, " num pred entity = ", pred_num, " num correct pred_entity=", right_num_pred, " num correct gold_entity=", right_num_gold)
    print("entity precision = ", precision, " entity recall = ", recall, " entity f1 = ", f_measure)

    if print_pred:
        json.dump(samples, log_file, ensure_ascii=False, indent=4)
        # print("num gold entity = ", gold_num, "; num pred entity = ", pred_num, "; num correct entity", right_num, file=log_file)
        # print("entity precision = ", precision, "; entity recall = ", recall, "; entity f1 = ", f_measure, file=log_file)

    return {"entity_precision": precision, "entity_recall": recall, "entity_f1": f_measure, "ent_pred": samples}

# 对所有batch的样本进行最后的统计计算
def ent_metric(pred, gold, list_text, log_fn, print_pred):
    assert pred.keys() == gold.keys()
    gold_num = 0
    #right_num_pred = 0
    #right_num_gold = 0
    right_num = 0
    pred_num = 0
    
    samples = []
    if print_pred:
        log_file = open(log_fn + '.ent', 'w', encoding='utf-8')
        
    for i, sent_idx in enumerate(pred):
        # if print_pred:
        #     print("Sample-ID:", sent_idx, file=log_file)
        #     print(list_text[i], file=log_file)

        gold_num += len(gold[sent_idx])
        prediction = pred[sent_idx]
        #prediction = set([ele[:4] for ele in prediction])
        prediction = list([ele[:4] for ele in prediction])
        pred_num += len(prediction)
        for ele in prediction:
            if ele in gold[sent_idx]:
                right_num += 1
        #match_p2t, match_t2p = sample_score(prediction, gold[sent_idx])
        #right_num_pred += match_p2t
        #right_num_gold += match_t2p
        

            # label, start, end, mention = ele
            # for ele_ in gold[sent_idx]:
            #     label_, start_, end_, mention_ = ele_
            #     if label == label_ and LCS(mention, mention_) >= max(len(mention), len(mention_))/2:
            #         right_num += 1
            #         break

        # if print_pred:
        prediction = sorted(list(prediction), key=lambda x: x[1:3])
        gt = sorted(list(gold[sent_idx]), key=lambda x: x[1:3])
        sample = {
            "doc_key": sent_idx,
            "sent_id": i,
            "text": list_text[i],
            "entities": prediction,
            "gold": gt
        }
        samples.append(sample)
        # print("Gold:", file=log_file)
        # gold_entities = sorted(list(gold[sent_idx]), key=lambda x: x[1:3])
        # tmp = json.dumps(gold_entities, ensure_ascii=False)
        # print(tmp, file=log_file)
        # print("Pred:", file=log_file)
        # pred_entities = sorted(list(prediction), key=lambda x: x[1:3])
        # tmp = json.dumps(pred_entities, ensure_ascii=False)
        # print(tmp, file=log_file)
        # print('', file=log_file)

    if pred_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / pred_num

    if gold_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / gold_num

    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    print("-----------No Overlap------------")
    print("num gold entity = ", gold_num, " num pred entity = ", pred_num, " num correct entity=", right_num)
    print("entity precision = ", precision, " entity recall = ", recall, " entity f1 = ", f_measure)

    #if print_pred:
        #json.dump(samples, log_file, ensure_ascii=False, indent=4)
        # print("num gold entity = ", gold_num, "; num pred entity = ", pred_num, "; num correct entity", right_num, file=log_file)
        # print("entity precision = ", precision, "; entity recall = ", recall, "; entity f1 = ", f_measure, file=log_file)

    return {"entity_precision": precision, "entity_recall": recall, "entity_f1": f_measure, "ent_pred": samples}

def sample_score_rel(pred_triples, true_triples):
    """
    计算单个样本的度量值
    :param pred_entities: 预测实体列表，每个实体为 (start, end, type)
    :param true_entities: 真实实体列表，每个实体为 (start, end, type)
    :return: 样本的度量值（0 到 1 之间）
    """
    n = len(pred_triples)
    m = len(true_triples)
    if n == 0 or m == 0:
        return 0.0, 0.0  # 如果没有预测或真实三元组，得分为 0

    # 初始化加权 IOU 矩阵
    weighted_ious_p2t = np.zeros((n, m))
    weighted_ious_t2p = np.zeros((m, n))

    
    for i, (p_type, p_head, p_tail) in enumerate(pred_triples):
        for j, (t_type, t_head, t_tail) in enumerate(true_triples):
            if p_type != t_type:
                weighted_ious_p2t[i, j] = 0
                continue
            p_head_type, p_head_start, p_head_end, _ = p_head
            p_tail_type, p_tail_start, p_tail_end, _ = p_tail
            t_head_type, t_head_start, t_head_end, _ = t_head
            t_tail_type, t_tail_start, t_tail_end, _ = t_tail

            '''head_score = calculate_iou(int(p_head_start), int(p_head_end), int(t_head_start), int(t_head_end)) * entityLinkMatrix[p_head_type][t_head_type]
            tail_score = calculate_iou(int(p_tail_start), int(p_tail_end), int(t_tail_start), int(t_tail_end)) * entityLinkMatrix[p_tail_type][t_tail_type]'''
            if p_head_type != t_head_type or p_tail_type != t_tail_type:
                continue
            head_score = calculate_iou(int(p_head_start), int(p_head_end), int(t_head_start), int(t_head_end)) 
            tail_score = calculate_iou(int(p_tail_start), int(p_tail_end), int(t_tail_start), int(t_tail_end))           
            weighted_ious_p2t[i][j] = head_score * tail_score

    for i, (t_type, t_head, t_tail) in enumerate(true_triples):
        for j, (p_type, p_head, p_tail) in enumerate(pred_triples):
            if p_type != t_type:
                weighted_ious_t2p[i, j] = 0
                continue
            p_head_type, p_head_start, p_head_end, _ = p_head
            p_tail_type, p_tail_start, p_tail_end, _ = p_tail
            t_head_type, t_head_start, t_head_end, _ = t_head
            t_tail_type, t_tail_start, t_tail_end, _ = t_tail

            '''head_score = calculate_iou(int(p_head_start), int(p_head_end), int(t_head_start), int(t_head_end)) * entityLinkMatrix[t_head_type][p_head_type]
            tail_score = calculate_iou(int(p_tail_start), int(p_tail_end), int(t_tail_start), int(t_tail_end)) * entityLinkMatrix[t_tail_type][p_tail_type]'''
            if p_head_type != t_head_type or p_tail_type != t_tail_type:
                continue
            head_score = calculate_iou(int(p_head_start), int(p_head_end), int(t_head_start), int(t_head_end))
            tail_score = calculate_iou(int(p_tail_start), int(p_tail_end), int(t_tail_start), int(t_tail_end))
            weighted_ious_t2p[i][j] = head_score * tail_score

    matches_p2t = np.max(weighted_ious_p2t, axis = 1)
    matches_t2p = np.max(weighted_ious_t2p, axis = 1)

    # 计算样本的最终度量值
    return sum(matches_p2t), sum(matches_t2p)

def metric_overlap(pred, gold, pred_ents, list_text, relational_alphabet, ent_type_alphabet, log_fn, print_pred, remove_overlap=True, list_category=None):
    assert pred.keys() == gold.keys()
    gold_num = 0
    #rel_num = 0
    #ent_num = 0
    right_num_pred = 0
    right_num_gold = 0
    pred_num = 0
    #pred_ent_num, gold_ent_num = 0, 0
    #pred_rel_num, gold_rel_num = 0, 0

    samples = []
    if print_pred:
        log_file = open(log_fn + '.rel', 'w', encoding='utf-8')
    # 对于每一个seq    
    for i, sent_idx in enumerate(pred):
        gt = gold[sent_idx]

        gt = list([(ele[0], (ele[1][3], ele[1][0], ele[1][1], ele[1][2]), (ele[2][3], ele[2][0], ele[2][1], ele[2][2]))for ele in gt])


        pred_ent = pred_ents[sent_idx]
        pred_ent = list([ele[:4] for ele in pred_ent])        

        gold_num += len(gt)
        prediction = filtration_with_etype(pred[sent_idx], relational_alphabet, ent_type_alphabet, remove_overlap=remove_overlap)
        prediction = list([(ele[0], (ele[7], ele[3], ele[4], ele[1]), (ele[8], ele[5], ele[6], ele[2])) for ele in prediction])

        filtered_prediction  = []
        for pred_rel in prediction:
            if pred_rel[1] in pred_ent and pred_rel[2] in pred_ent:
                filtered_prediction.append(pred_rel)
    
        #pred_num += len(prediction)
        pred_num += len(filtered_prediction)
 
        '''gold_rel_set = set([e[0] for e in gt])
        pred_rel_set = set([ele[0] for ele in prediction])
        gold_ent_set = set([e[1:] for e in gt])
        pred_ent_set = set([ele[1:] for ele in prediction])'''

        '''for ele in prediction:          # right_num : prediction和gt中一致的triple数目
            if ele in gt:
                right_num += 1'''

        #match_p2t, match_t2p = sample_score_rel(prediction, gt)
        match_p2t, match_t2p = sample_score_rel(filtered_prediction, gt)
        right_num_pred += match_p2t
        right_num_gold += match_t2p  

        '''rel_num += len(pred_rel_set & gold_rel_set)     # 关系类型一致的数目
        ent_num += len(pred_ent_set & gold_ent_set)     # 实体一致的数目
        pred_ent_num += len(pred_ent_set)
        pred_rel_num += len(pred_rel_set)
        gold_ent_num += len(gold_ent_set)
        gold_rel_num += len(gold_rel_set)'''
        # if print_pred:
        #prediction = sorted(list(prediction), key=lambda x: (x[1][0], x[1][1], x[2][0], x[2][1]))
        filtered_prediction = sorted(list(filtered_prediction), key=lambda x: (x[1][0], x[1][1], x[2][0], x[2][1]))
        gt = sorted(list(gt), key=lambda x: (x[1][0], x[1][1], x[2][0], x[2][1]))
        sample = {
            "doc_key": sent_idx,
            "sent_id": i,
            "text": list_text[i],
            #"category": list_category[i],
            "category": None,
            "relations": filtered_prediction,
            #"relations": prediction,
            "gold": gt,
        }
        if list_category:
            sample["category"] = list_category[i]
        samples.append(sample)
        # print("Gold:", file=log_file)
        # print([e[:3] for e in gt], file=log_file)
        # print("Pred:", file=log_file)
        # print([e[:3] for e in prediction], file=log_file)
        # print('', file=log_file)

    if pred_num == 0:
        precision = -1
        #r_p = -1
        #e_p = -1
    else:
        precision = (right_num_pred + 0.0) / pred_num
        #e_p = (ent_num + 0.0) / pred_ent_num
        #r_p = (rel_num + 0.0) / pred_rel_num

    if gold_num == 0:
        recall = -1
        #r_r = -1
        #e_r = -1
    else:
        recall = (right_num_gold + 0.0) / gold_num
        #e_r = ent_num / gold_ent_num
        #r_r = rel_num / gold_rel_num

    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    '''if (e_p == -1) or (e_r == -1) or (e_p + e_r) <= 0.:
        e_f = -1
    else:
        e_f = 2 * e_r * e_p / (e_p + e_r)

    if (r_p == -1) or (r_r == -1) or (r_p + r_r) <= 0.:
        r_f = -1
    else:
        r_f = 2 * r_p * r_r / (r_r + r_p)'''

    print('-----------Overlap-------------')
    print("gold_num = ", gold_num, " pred_num = ", pred_num, " right_num_pred = ", right_num_pred, " right_num_gold = ", right_num_gold)
    print("precision = ", precision, " recall = ", recall, " f1_value = ", f_measure)
    #print("rel_precision = ", r_p, " rel_recall = ", r_r, " rel_f1_value = ", r_f)
    #print("head_tail_precision = ", e_p, " head_tail_recall = ", e_r, " head_tail_f1 = ", e_f)

    if print_pred:
        json.dump(samples, log_file, ensure_ascii=False, indent=4)
        # print("gold_num = ", gold_num, " pred_num = ", pred_num, " right_num = ", right_num, file=log_file)
        # print("precision = ", precision, " recall = ", recall, " f1_value = ", f_measure, file=log_file)
        # print("rel_precision = ", r_p, " rel_recall = ", r_r, " rel_f1_value = ", r_f, file=log_file)
        # print("ent_precision = ", e_p, " ent_recall = ", e_r, " ent_f1_value = ", e_f, file=log_file)
    return {"precision": precision, "recall": recall, "f1": f_measure, "rel_pred": samples}


def metric(pred, gold, pred_ents, list_text, relational_alphabet, ent_type_alphabet, log_fn, print_pred, remove_overlap=True, list_category=None):
    assert pred.keys() == gold.keys()
    gold_num = 0
    rel_num = 0
    ent_num = 0
    right_num = 0
    #right_num_pred = 0
    #right_num_gold = 0
    pred_num = 0
    #pred_ent_num, gold_ent_num = 0, 0
    #pred_rel_num, gold_rel_num = 0, 0


    samples = []
    if print_pred:
        log_file = open(log_fn + '.rel', 'w', encoding='utf-8')
    # 对于每一个seq    
    for i, sent_idx in enumerate(pred):
        gt = gold[sent_idx]
        
        gt = list([(ele[0], (ele[1][3], ele[1][0], ele[1][1], ele[1][2]), (ele[2][3], ele[2][0], ele[2][1], ele[2][2]))for ele in gt])


        # if print_pred:
        #     print("Sample-ID:", sent_idx, file=log_file)
        #     print(list_text[i], file=log_file)
        pred_ent = pred_ents[sent_idx]
        pred_ent = list([ele[:4] for ele in pred_ent]) 

        gold_num += len(gt)
        prediction = filtration_with_etype(pred[sent_idx], relational_alphabet, ent_type_alphabet, remove_overlap=remove_overlap)
        prediction = list([(ele[0], (ele[7], ele[3], ele[4], ele[1]), (ele[8], ele[5], ele[6], ele[2])) for ele in prediction])

        filtered_prediction  = []
        for pred_rel in prediction:
            if pred_rel[1] in pred_ent and pred_rel[2] in pred_ent:
                filtered_prediction.append(pred_rel)

        #prediction = convert(prediction)
        pred_num += len(filtered_prediction)
        #pred_num += len(prediction)

        '''gold_rel_set = set([e[0] for e in gt])
        pred_rel_set = set([ele[0] for ele in prediction])
        gold_ent_set = set([e[1:] for e in gt])
        pred_ent_set = set([ele[1:] for ele in prediction])'''

        for ele in filtered_prediction:          # right_num : prediction和gt中一致的triple数目
        #for ele in prediction: 
            if ele in gt:
                right_num += 1

        #match_p2t, match_t2p = sample_score_rel(prediction, gold[sent_idx])
        #right_num_pred += match_p2t
        #right_num_gold += match_t2p  

        '''rel_num += len(pred_rel_set & gold_rel_set)     # 关系类型一致的数目
        ent_num += len(pred_ent_set & gold_ent_set)     # 实体一致的数目
        pred_ent_num += len(pred_ent_set)
        pred_rel_num += len(pred_rel_set)
        gold_ent_num += len(gold_ent_set)
        gold_rel_num += len(gold_rel_set)'''
        # if print_pred:
        filtered_prediction = sorted(list(filtered_prediction), key=lambda x: (x[1][0], x[1][1], x[2][0], x[2][1]))
        #prediction = sorted(list(prediction), key=lambda x: (x[1][0], x[1][1], x[2][0], x[2][1]))
        gt = sorted(list(gt), key=lambda x: (x[1][0], x[1][1], x[2][0], x[2][1]))
        sample = {
            "doc_key": sent_idx,
            "sent_id": i,
            "text": list_text[i],
            #"category": list_category[i],
            "category": None,
            "relations": filtered_prediction,
            #"relations": prediction,
            "gold": gt,
        }
        if list_category:
            sample["category"] = list_category[i]
        samples.append(sample)
        # print("Gold:", file=log_file)
        # print([e[:3] for e in gt], file=log_file)
        # print("Pred:", file=log_file)
        # print([e[:3] for e in prediction], file=log_file)
        # print('', file=log_file)

    if pred_num == 0:
        precision = -1
        #r_p = -1
        #e_p = -1
    else:
        precision = (right_num + 0.0) / pred_num
        #e_p = (ent_num + 0.0) / pred_ent_num
        #r_p = (rel_num + 0.0) / pred_rel_num

    if gold_num == 0:
        recall = -1
        r_r = -1
        e_r = -1
    else:
        recall = (right_num + 0.0) / gold_num
        #e_r = ent_num / gold_ent_num
        #r_r = rel_num / gold_rel_num

    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    '''if (e_p == -1) or (e_r == -1) or (e_p + e_r) <= 0.:
        e_f = -1
    else:
        e_f = 2 * e_r * e_p / (e_p + e_r)

    if (r_p == -1) or (r_r == -1) or (r_p + r_r) <= 0.:
        r_f = -1
    else:
        r_f = 2 * r_p * r_r / (r_r + r_p)'''

    print("---------No Overlap------------")
    print("gold_num = ", gold_num, " pred_num = ", pred_num, " right_num = ", right_num)
    print("precision = ", precision, " recall = ", recall, " f1_value = ", f_measure)
    #print("rel_precision = ", r_p, " rel_recall = ", r_r, " rel_f1_value = ", r_f)
    #print("head_tail_precision = ", e_p, " head_tail_recall = ", e_r, " head_tail_f1 = ", e_f)

    '''if print_pred:
        json.dump(samples, log_file, ensure_ascii=False, indent=4)'''
        # print("gold_num = ", gold_num, " pred_num = ", pred_num, " right_num = ", right_num, file=log_file)
        # print("precision = ", precision, " recall = ", recall, " f1_value = ", f_measure, file=log_file)
        # print("rel_precision = ", r_p, " rel_recall = ", r_r, " rel_f1_value = ", r_f, file=log_file)
        # print("ent_precision = ", e_p, " ent_recall = ", e_r, " ent_f1_value = ", e_f, file=log_file)
    return {"precision": precision, "recall": recall, "f1": f_measure, "rel_pred": samples}






def is_normal_triplet(triplets):
    entities = set()
    for triplet in triplets:
        head_entity = (triplet[1], triplet[2])
        tail_entity = (triplet[3], triplet[4])
        entities.add(head_entity)
        entities.add(tail_entity)
    return len(entities) == 2 * len(triplets)


def is_multi_label(triplets):
    if is_normal_triplet(triplets):
        return False
    entity_pair = [(triplet[1], triplet[2], triplet[3], triplet[4]) for triplet in triplets]
    return len(entity_pair) != len(set(entity_pair))


def is_overlapping(triplets):
    if is_normal_triplet(triplets):
        return False
    entity_pair = [(triplet[1], triplet[2], triplet[3], triplet[4]) for triplet in triplets]
    entity_pair = set(entity_pair)
    entities = []
    for pair in entity_pair:
        entities.append((pair[0], pair[1]))
        entities.append((pair[2], pair[3]))
    entities = set(entities)
    return len(entities) != 2 * len(entity_pair)


def get_key_val(dict_1, list_1):
    dict_2 = dict()
    for ele in list_1:
        dict_2.update({ele: dict_1[ele]})
    return dict_2
