import os, json
import time
import argparse
from collections import defaultdict, Counter

def offset_entity(entity, offset):
    etype, st, ed, mention = entity
    return (etype, st-offset, ed-offset, mention)

def offset_relation(relation, offset):
    rel, h, t = relation
    return (rel, (h[0]-offset, h[1]-offset, h[2], h[3]), (t[0]-offset, t[1]-offset, t[2], t[3]))

def overlap_ent(ent1, ent2):
    return max(ent1[1], ent2[1]) <= min(ent1[2], ent2[2])

def within_ent(ent1, ent2):
    return (ent1[1] < ent2[1] <= ent2[2] < ent1[2]) or (ent2[1] < ent1[1] <= ent1[2] < ent2[2])

def overlap_span(span1, span2):
    return max(span1[0], span2[0]) <= min(span1[1], span2[1])

def overlap_rel(rel1, rel2):
    return overlap_span(rel1[2], rel2[2]) and (overlap_span(rel1[1], rel2[1]) or rel1[0]==rel2[0])

def conflict(ele1, ele2, ele_type):
    if ele_type == 'ent':
        return overlap_ent(ele1, ele2) and not within_ent(ele1, ele2)
    elif ele_type == 'rel':
        return overlap_rel(ele1, ele2)
    else:
        raise ValueError(f'Unsupported ele_type: {ele_type}')

def can_replace(ele1, ele2, ele_type):
    if ele_type == 'ent':
        return ele1[0] == ele2[0] and ele1[1] <= ele2[1] <= ele2[2] <= ele1[2] and len(ele1) >= len(ele2)*2
    elif ele_type == 'rel':
        rel1, h1, t1 = ele1
        rel2, h2, t2 = ele2
        return rel1 == rel2 and h1[0] <= h2[0] <= h2[1] <= h1[1] and t1[0] <= t2[0] <= t2[1] <= t1[1]
    else:
        return False



def voting(eles, ths, ele_type):
    ele_cnt = Counter(eles)
    new_eles = set()
    for ele in sorted(ele_cnt, key=lambda x: ele_cnt[x], reverse=True):
        cnt = ele_cnt[ele]
        if cnt < ths:
            break
        
        valid = True
        for ele_ in list(new_eles):
            if conflict(ele, ele_, ele_type):
                if can_replace(ele_, ele, ele_type):
                    new_eles.remove(ele_)
                    continue
                valid = False
                break
        if valid:
            new_eles.add(ele)

    new_eles = list(new_eles)
    if ele_type == 'ent':
        return sorted(new_eles, key=lambda x: x[1:3])
    elif ele_type == 'rel':
        return sorted(new_eles, key=lambda x: (x[1][0], x[1][1], x[2][0], x[2][1]))
    else:
        raise ValueError(f'Unsupported ele_type: {ele_type}')
    
def get_ent_in_rel(relations):
    entities = set()
    for (rel, h, t) in relations:
        h_etype, t_etype = rel[1:-1].split('，', 1)
        # if h[3] != h_etype and t[3] != t_etype:
        #     print((rel, h, t), 'fff')
        #     continue
        # if h[3] != h_etype or t[3] != t_etype:
        #     print((rel, h, t))
        entities.add((h_etype,) + h[:3])
        entities.add((t_etype,) + t[:3])
    return list(entities)

KNOWN_ENTITIES = {
    "定性结果": ["阴性", "部分+", "网状纤维+"],
    "肝硬化镜检": ['周围肝组织未见结节性肝硬化', '周围肝组织结节性肝硬化', '周围肝组织 未见结节性肝硬化', '周围肝组织 结节性肝硬化', '肝组织结节性肝硬化'],
    "切缘累及": ['肝切缘未见癌累及', '两切缘未见癌累及'],
    "诊断名称": ['乙肝', '丙肝', "结节性肝硬化", '肝硬化', '食管胃底静脉曲张', '食管胃底静脉曲', '肝腺癌', '腺癌', '局灶性结节状增生', "肝细胞癌", "肝小囊肿", "结肠性肿瘤", "(肝右叶) 肝细胞肝癌", "肝细胞肝癌", "胆管细胞癌", "原发性肝MT"],
    "手术名称": ['支架', '肝MT术', '胃镜下治疗'],
    "家族病史": ['家族史'],
    "药物类型": ['抑酸护胃', "抑酸", '降压', '利尿', '降钾', "降糖", "抗病毒", "止痛", 'FOLFOX4化疗', 'XELOX'],
    "症状体征": ['精神、食纳及睡眠可', '大小便正常', '体重无明显下降', '体重未见明显改变', '乏力纳差', '纳差乏力', '乏力', '胸痛', '头晕', '水肿', '黑便', '呕血', '夜眠可', "呼吸困难", "精神可", "夜眠佳"],
    "医院名称": ['我科', '瑞金医院', '张家港市第六人民医院'],
    "状态词-否定": ['未见', '否认'],
    "蛋白质说明": ["未检测到突变"],
    "持续时间": ["一小时", "2天"],
    "临床发现途径": ["体检"],
    "药物名称": ["耐信", '吉西他滨'],
    "组织类型补充说明": ['细梁型，粗梁型，假腺管型，团片型', '细梁型，粗梁型，假腺管型', '细梁型，粗梁型', '细梁型'],
    "放射治疗": ["TOMO放疗", '放疗'],
    "治疗次数": ["6个疗程"],
    "脂肪变性": ['肝细胞脂肪变性'],
}

def clean_ent(entities, valid_classes):
    entities_ = []
    for ent in entities:
        ent = list(ent)
        try:
            time.strptime(ent[-1], '%Y-%m-%d')
            ent[0] = '时间'
        except:
            pass

        for ent_class, mentions in KNOWN_ENTITIES.items():
            if ent[-1] in mentions:
                ent[0] = ent_class
                break

        if ent[-1][0] == 'I' and ent[-1][-1] == '级':
            ent[0] = "病理分级分化"
        if ent[-1][-1] == '术' and ent[0] != '手术名称':
            if len(ent[-1]) >= 15:
                continue
            ent[0] = "手术名称"
        if '年余' in ent[-1] and '时间' not in ent[0]:
            continue
        if '未见结节性肝硬化' in ent[-1]:
            ent[0] = "肝硬化镜检"
        if '乙肝病毒DNA' in ent[-1]:
            ent[0] = "临床发现"
        if '肝穿刺' in ent[-1]:
            ent[0] = "手术名称"
        if ent[-1] == '及':
            continue
        if ent[-1] == '无' and ent[0] != '状态词-否定':
            continue
        if ent[0] not in valid_classes:
            continue
        entities_.append(ent)
    return entities_


def clean_rel(relations, valid_classes, doc_type):
    interval_ths = 36 if doc_type == '现病史' else 22
    relations_ = []
    for rel in relations:
        if doc_type == '现病史' and rel[1][0] - rel[2][1] > 10:
            continue
        if min(abs(rel[1][0] - rel[2][1]), abs(rel[1][1] - rel[2][0])) > interval_ths:
            continue
        if rel[0] not in valid_classes:
            continue
        if rel[1][:2] == rel[2][:2]:
            continue
        relations_.append(rel)
    return relations_

def find_entities(text: str, entities, valid_classes):
    found_entities = []
    for ent_class, mentions in KNOWN_ENTITIES.items():
        if ent_class not in valid_classes:
            continue
        for mention in mentions:
            st = text.find(mention)
            if st < 0:
                continue
            ent = (ent_class, st, st+len(mention)-1, mention)
            if any(overlap_ent(e, ent) for e in entities):
                continue
            if any(overlap_ent(e, ent) for e in found_entities):
                continue
            found_entities.append(ent)
            # print(text)
            # print(ent)
            # print()
    return found_entities

def assemble_ent(list_ent_preds, ths, doc_key_2_entities, doc_type_2_classes):
    for preds in list_ent_preds:
        for d in preds:
            doc_key = d["doc_key"]
            doc_type = doc_key.split('.')[0]
            text = d["text"]
            prefix = f'【{doc_type}】 '
            offset = len(prefix)
            assert text[:offset] == prefix
            text = text[offset:]
            suffix = f' 【{doc_type}】'
            if text[-len(suffix):] == suffix:
                text = text[:-len(suffix)]
            entities = [x for x in d['entities'] if x[0] in doc_type_2_classes[doc_type]]
            entities = clean_ent(entities, doc_type_2_classes[doc_type])
            entities = [offset_entity(x, offset) for x in entities]
            doc_key_2_entities[doc_key].extend(entities)
            # find entities
            found_entities = find_entities(text, entities, doc_type_2_classes[doc_type])
            doc_key_2_entities[doc_key].extend(found_entities)


    doc_key_2_entities_ = dict()
    for doc_key, entities in doc_key_2_entities.items():
        doc_type = doc_key.split('.')[0]
        entities_ = voting(entities, ths, 'ent')
        entities_ = clean_ent(entities_, doc_type_2_classes[doc_type])
        doc_key_2_entities_[doc_key] = entities_

    return doc_key_2_entities_

def assemble_rel(list_rel_preds, ths, doc_type_2_classes):
    doc_key_2_text = dict()
    doc_key_2_relations = defaultdict(list)
    doc_key_2_entities = dict()
    for preds in list_rel_preds:
        for d in preds:
            doc_key = d["doc_key"]
            doc_type = doc_key.split('.')[0]
            text = d["text"]
            prefix = f'【{doc_type}】 '
            offset = len(prefix)
            assert text[:offset] == prefix
            text = text[offset:]
            suffix = f' 【{doc_type}】'
            if text[-len(suffix):] == suffix:
                text = text[:-len(suffix)]
            relations = [x for x in d['relations'] if x[0] in doc_type_2_classes[doc_type]]
            relations = clean_rel(relations, doc_type_2_classes[doc_type], doc_type)
            relations = [offset_relation(x, offset) for x in relations]

            doc_key_2_text[doc_key] = text
            doc_key_2_relations[doc_key].extend(relations)
            
    doc_key_2_relations_ = dict()
    for doc_key, relations in doc_key_2_relations.items():
        # print(relations)
        relations_ = voting(relations, ths, 'rel')
        entities = get_ent_in_rel(relations_)
        relations_ = [(rel, h[:3], t[:3]) for (rel, h, t) in relations_]
        doc_key_2_relations_[doc_key] = relations_
        doc_key_2_entities[doc_key] = entities

    return doc_key_2_text, doc_key_2_relations_, doc_key_2_entities

def reconcile_rel(doc_key_2_relations, doc_key_2_entities, doc_type_2_classes):
    mention_2_etype = dict()
    for ent_class, mentions in KNOWN_ENTITIES.items():
        for mention in mentions:
            mention_2_etype[mention] = ent_class
    
    doc_key_2_relations_ = dict()
    doc_key_2_entities_ = dict()
    for doc_key, relations in doc_key_2_relations.items():
        doc_type = doc_key.split('.')[0]
        valid_classes = doc_type_2_classes[doc_type]
        entities = doc_key_2_entities[doc_key]
        span_2_etype = dict((tuple(e[1:3]), e[0]) for e in entities)

        relations_ = []
        for r in relations:
            h_mention, t_mention = r[1][2], r[2][2]
            h_span, t_span = tuple(r[1][:2]), tuple(r[2][:2])
            if h_span not in span_2_etype or t_span not in span_2_etype:
                continue
            rel_type = f"（{span_2_etype[h_span]}，{span_2_etype[t_span]}）"
            if rel_type != r[0]:
                # print(doc_key)
                # print(r, rel_type)
                # print()
                h_type, t_type = r[0][1:-1].split('，')
                if h_mention in mention_2_etype and mention_2_etype[h_mention] != h_type:
                    continue
                if t_mention in mention_2_etype and mention_2_etype[t_mention] != t_type:
                    continue
                if rel_type not in valid_classes:
                    rel_type = r[0]
                    span_2_etype[h_span], span_2_etype[t_span] = h_type, t_type
            relations_.append((rel_type, r[1], r[2]))
        doc_key_2_relations_[doc_key] = relations_

        entities_ = []
        for e in entities:
            etype = span_2_etype[tuple(e[1:3])]
            e_ = [etype,] + e[1:]
            entities_.append(e_)
        doc_key_2_entities_[doc_key] = entities_
    return doc_key_2_relations_, doc_key_2_entities_

def ensemble(in_dir, model_ids, ent_ths, rel_ths, doc_type_2_classes, list_rel_preds=None, list_ent_preds=None):
    if list_rel_preds is None:
        list_rel_preds = []
        list_ent_preds = []
        for model_id in model_ids:
            rel_pred = json.load(open(os.path.join(in_dir, model_id+'.rel')))
            ent_pred = json.load(open(os.path.join(in_dir, model_id+'.ent')))
            list_rel_preds.append(rel_pred)
            list_ent_preds.append(ent_pred)

    doc_key_2_text, doc_key_2_relations, doc_key_2_entities = assemble_rel(
        list_rel_preds, rel_ths, doc_type_2_classes['rel']
    )
    doc_key_2_entities = assemble_ent(
        list_ent_preds, ent_ths, doc_key_2_entities, doc_type_2_classes['ent']
    )
    doc_key_2_relations, doc_key_2_entities = reconcile_rel(doc_key_2_relations, doc_key_2_entities, doc_type_2_classes['rel'])
    return doc_key_2_text, doc_key_2_relations, doc_key_2_entities

def pred_2_json(doc_key_2_text, doc_key_2_relations, doc_key_2_entities, ent_json_name, rel_json_name):
    ent_ensemble = []
    rel_ensemble = []
    for doc_key, text in doc_key_2_text.items():
        entities = doc_key_2_entities[doc_key]
        relations = doc_key_2_relations[doc_key]
        ent_ensemble.append(
            {
                "doc_key": doc_key,
                "text": text,
                "entities": entities,
            }
        )
        rel_ensemble.append(
            {
                "doc_key": doc_key,
                "text": text,
                "relations": relations,
            }
        )
    json.dump(ent_ensemble, open(ent_json_name, 'w'), ensure_ascii=False, indent=2)
    json.dump(rel_ensemble, open(rel_json_name, 'w'), ensure_ascii=False, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default="../predictions")
    parser.add_argument('--class_fn', type=str, default="baidu_cancer_classes.json")
    parser.add_argument('--model_ids', type=str, default="['01-31-17-56-41','01-31-23-09-13','02-01-01-11-22','01-31-17-18-21']")
    parser.add_argument('--ent_ths', type=int, default=2)
    parser.add_argument('--rel_ths', type=int, default=1)
    args, unparsed = parser.parse_known_args()

    model_ids = eval(args.model_ids)
    doc_type_2_classes = json.load(open(args.class_fn))
    doc_key_2_text, doc_key_2_relations, doc_key_2_entities = ensemble(
        args.in_dir, model_ids, args.ent_ths, args.rel_ths, doc_type_2_classes
    )
    pred_2_json(doc_key_2_text, doc_key_2_relations, doc_key_2_entities, "pred_ent__.json", "pred_rel__.json")
