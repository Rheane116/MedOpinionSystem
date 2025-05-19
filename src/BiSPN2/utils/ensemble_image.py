import os, json
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

def overlap_span(span1, span2):
    return max(span1[0], span2[0]) <= min(span1[1], span2[1])

def overlap_rel(rel1, rel2):
    return (
        overlap_span(rel1[1], rel2[1]) and overlap_span(rel1[2], rel2[2])
        or rel1[0]==rel2[0] and (
            overlap_span(rel1[1], rel2[1]) and abs(rel1[2][0] - rel2[2][0]) > 26
            or overlap_span(rel1[2], rel2[2]) and abs(rel1[1][0] - rel2[1][0]) > 26
        )
    )

def conflict(ele1, ele2, ele_type):
    if ele_type == 'ent':
        return overlap_ent(ele1, ele2)
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
        return rel1 == rel2 and (h1[0] <= h2[0] <= h2[1] <= h1[1] and t1[0] <= t2[0] <= t2[1] <= t1[1]
            or t1[0] < t2[0] and abs(h2[0] - t2[0]) < abs(h1[0] - t1[0]))
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
        entities.add((h_etype,) + h[:3])
        entities.add((t_etype,) + t[:3])
    return list(entities)

KNOWN_ENTITIES = {
    "状态词-疑似": ['考虑', '类似', '似'],
    "状态词-否定": ['未见', '无'],
}

def clean_ent(entities):
    entities_ = []
    for ent in entities:
        ent = list(ent)
        for ent_class, mentions in KNOWN_ENTITIES.items():
            if ent[-1] in mentions:
                ent[0] = ent_class
                break
        if ent[-1].endswith('mm'):
            ent[0] = "数值"
        entities_.append(ent)
    return entities_

def clean_rel(relations, valid_classes):
    relations_ = []
    for rel in relations:
        if min(abs(rel[1][0] - rel[2][1]), abs(rel[1][1] - rel[2][0])) > 64:
            continue
        if rel[0] not in valid_classes:
            continue
        if overlap_span(rel[1][:2], rel[2][:2]):
            continue
        relations_.append(rel)
    return relations_

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
            entities = [x for x in d['entities'] if x[0] in doc_type_2_classes[doc_type]]
            entities = [offset_entity(x, offset) for x in entities]
            doc_key_2_entities[doc_key].extend(entities)

    doc_key_2_entities_ = dict()
    for doc_key, entities in doc_key_2_entities.items():
        entities_ = voting(entities, ths, 'ent')
        entities_ = clean_ent(entities_)
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
            relations = [x for x in d['relations'] if x[0] in doc_type_2_classes[doc_type]]
            relations = clean_rel(relations, doc_type_2_classes[doc_type])
            relations = [offset_relation(x, offset) for x in relations]

            doc_key_2_text[doc_key] = text
            doc_key_2_relations[doc_key].extend(relations)
            
    doc_key_2_relations_ = dict()
    for doc_key, relations in doc_key_2_relations.items():
        # print(relations)
        relations_ = voting(relations, ths, 'rel')
        entities = get_ent_in_rel(relations_)
        doc_key_2_relations_[doc_key] = relations_
        doc_key_2_entities[doc_key] = entities

    return doc_key_2_text, doc_key_2_relations_, doc_key_2_entities

def reconcile_rel(doc_key_2_relations, doc_key_2_entities, doc_type_2_classes):    
    doc_key_2_relations_ = dict()
    doc_key_2_entities_ = dict()
    for doc_key, relations in doc_key_2_relations.items():
        doc_type = doc_key.split('.')[0]
        valid_classes = doc_type_2_classes[doc_type]
        entities = doc_key_2_entities[doc_key]
        span_2_etype = dict((tuple(e[1:3]), e[0]) for e in entities)
        span_2_mention = dict((tuple(e[1:3]), e[3]) for e in entities)

        relations_ = []
        for r in relations:
            h_mention, t_mention = r[1][2], r[2][2]
            h_span, t_span = tuple(r[1][:2]), tuple(r[2][:2])
            if h_span not in span_2_etype:
                for span in span_2_etype:
                    if overlap_span(h_span, span):
                        h_span = span
                        h_mention = span_2_mention[span]
            if t_span not in span_2_etype:
                for span in span_2_etype:
                    if overlap_span(t_span, span):
                        t_span = span
                        t_mention = span_2_mention[span]

            if h_span not in span_2_etype or t_span not in span_2_etype:
                continue
            rel_type = f"（{span_2_etype[h_span]}，{span_2_etype[t_span]}）"
            if rel_type != r[0]:
                # print(doc_key)
                # print(r, rel_type)
                # print()
                h_type, t_type = r[0][1:-1].split('，', 1)
                if rel_type not in valid_classes:
                    rel_type = r[0]
                    span_2_etype[h_span], span_2_etype[t_span] = h_type, t_type
            relations_.append((rel_type, h_span+(h_mention,span_2_etype[h_span]), t_span+(t_mention,span_2_etype[t_span])))
        doc_key_2_relations_[doc_key] = relations_

        entities_ = []
        for e in entities:
            etype = span_2_etype[tuple(e[1:3])]
            e_ = [etype,] + e[1:]
            entities_.append(e_)
        doc_key_2_entities_[doc_key] = entities_
    return doc_key_2_relations_, doc_key_2_entities_

def ensemble_image(in_dir, model_ids, ent_ths, rel_ths, doc_type_2_classes, list_rel_preds=None, list_ent_preds=None):
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


# in_dir = "../predictions"
# class_fn = "baidu_image_classes.json"
# model_ids = ["01-25-18-14-12", "01-25-17-01-48", "01-25-17-01-24", "01-25-18-13-52", "01-25-14-59-40", "01-25-15-08-01"]
# # model_ids = ["01-25-18-14-12", "01-25-17-01-48", "01-25-17-01-24", "01-25-18-13-52", "02-02-17-31-56", "02-02-19-01-35"]
# ent_ths = 1
# rel_ths = 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default="../predictions")
    parser.add_argument('--class_fn', type=str, default="baidu_image_classes.json")
    parser.add_argument('--model_ids', type=str, default="['01-25-18-14-12','01-25-17-01-48','01-25-17-01-24','01-25-18-13-52','01-25-14-59-40','01-25-15-08-01']")
    parser.add_argument('--ent_ths', type=int, default=1)
    parser.add_argument('--rel_ths', type=int, default=2)
    args, unparsed = parser.parse_known_args()

    model_ids = eval(args.model_ids)
    doc_type_2_classes = json.load(open(args.class_fn))
    doc_key_2_text, doc_key_2_relations, doc_key_2_entities = ensemble_image(
        args.in_dir, model_ids, args.ent_ths, args.rel_ths, doc_type_2_classes
    )
    pred_2_json(doc_key_2_text, doc_key_2_relations, doc_key_2_entities, "ent__.json", "rel__.json")
