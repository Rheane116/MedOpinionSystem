import os, pickle, copy, sys, copy, json, itertools
import torch
from utils.alphabet import Alphabet
from transformers import BertTokenizer, BertTokenizerFast, AlbertTokenizer

def list_index(list1: list, list2: list) -> list:
    start = [i for i, x in enumerate(list2) if x == list1[0]]
    end = [i for i, x in enumerate(list2) if x == list1[-1]]
    if len(start) == 1 and len(end) == 1:
        return start[0], end[0]
    else:
        for i in start:
            for j in end:
                if i <= j:
                    if list2[i:j+1] == list1:
                        index = (i, j)
                        break
        return index[0], index[1]
        

def medical_data_process(input_doc, relational_alphabet, entity_type_alphabet, tokenizer, evaluate, repeat_gt_entities=-1, repeat_gt_triples=-1, max_len=495, max_ent=80):
    samples = []
    total_triples = 0
    total_entities = 0
    max_triples = 0
    max_entities = 0
    max_len_mention = 0
    num_samples = 0
    avg_len = 0
    print(input_doc)

    avg_no_entity_len = 0
    num_no_entity = 0

    with open(input_doc) as f:
        lines = json.load(f)

    for idx, line in enumerate(lines):
        doc_key = line["doc_key"]
        sent_id = line["sent_id"]
        text = line["text"]
        ori_pos = line['original_position'] if 'original_position' in line else (0, len(text))
        category = line['category'] if 'category' in line else None
        doc_type = line['doc_type'] if 'doc_type' in line else None
        idx = doc_key + '.' + str(sent_id)

        if doc_type and doc_type in ['病理报告', '个人史', '既往史', '婚育史', '家族史', '现病史']:
            text += ' 【' + doc_type + '】'
        
        enc = tokenizer(text, add_special_tokens=True)
        sent_id = enc['input_ids']
        if category:
            appended = tokenizer('【' + category + '】', add_special_tokens=False)['input_ids']
            sent_id += appended

        if not evaluate and len(line['entities']) == 0:
            avg_no_entity_len += len(sent_id)
            num_no_entity += 1
            continue

        # if len(line['entities']) <= 1:
        #     continue

        if len(line['entities']) > max_ent:
            print(f"{text}\n# entities({len(line['entities'])}) > max_ent({max_ent})")
            continue

        if len(sent_id) > max_len:
            print(f"{text}\n# tokens({len(sent_id)}) > max_len({max_len})")
            continue

        avg_len += len(text)

        sent_seg_encoding = [0] * len(sent_id)
        context2token_masks = None
        token_masks = [1] * len(sent_id)

        char_to_bep = dict()
        bep_to_char = dict()
        for i in range(len(text)):
            bep_index = enc.char_to_token(i)
            char_to_bep[i] = bep_index
            if bep_index in bep_to_char:
                left, right = bep_to_char[bep_index][0], bep_to_char[bep_index][-1]
                bep_to_char[bep_index] = [left, max(right, i)]
            else:
                bep_to_char[bep_index] = [i, i]
        
        target = {"relation": [], "head_start_index": [], "head_end_index": [], "tail_start_index": [], "tail_end_index": [],
                    "head_mention": [], "tail_mention": [], 'head_part_labels': [], 'tail_part_labels': [], 'head_type': [], 'tail_type': [],
                    "head_entID": [], "tail_entID": [], "rel_entID_labels": [],
                    "ent_type": [], "ent_start_index": [], "ent_end_index": [], "ent_part_labels": [], 'ent_have_rel': [],
                    "relID_labels": [], "relID_head_labels": [], "relID_tail_labels": []
                }

        flag_invalid = False
        if evaluate:
            triples = line["relations"]
            for triple in triples:
                rel_type = triple[-1] if triple[-1] != "对应关系" else category
                relation_id = relational_alphabet.get_index(rel_type)
                h_start, h_end, h_mention, h_label = triple[0]
                t_start, t_end, t_mention, t_label = triple[1]

                assert text[h_start: h_end] == h_mention
                assert text[t_start: t_end] == t_mention

                max_len_mention = max(max_len_mention, len(h_mention))
                max_len_mention = max(max_len_mention, len(t_mention))

                target["relation"].append(relation_id)
                target["head_start_index"].append(h_start)
                target["head_end_index"].append(h_end-1)
                target["tail_start_index"].append(t_start)
                target["tail_end_index"].append(t_end-1)
                target["head_mention"].append(h_mention)
                target["tail_mention"].append(t_mention)
                target["head_type"].append(entity_type_alphabet.get_index(h_label))
                target["tail_type"].append(entity_type_alphabet.get_index(t_label))

            entities = line["entities"]
            for ent in entities:
                ent_type_id = entity_type_alphabet.get_index(ent[-1])
                ent_start, ent_end = ent[0], ent[1]
                ent_start_index, ent_end_index = char_to_bep[ent_start], char_to_bep[ent_end-1]
                if ent_start_index is None or ent_end_index is None:
                    print(text)
                    print(text[ent_start: ent_end])
                    flag_invalid = True
                    continue

                target["ent_type"].append(ent_type_id)
                target["ent_start_index"].append(ent_start_index)
                target["ent_end_index"].append(ent_end_index)

            if flag_invalid:
                exit(0)
                continue
            
            samples.append([idx, sent_id, target, None, text, bep_to_char, sent_seg_encoding, context2token_masks, token_masks, None, None, ori_pos, category])

        else:
            repeat_num = 1
            triples = line["relations"]
            entities = line["entities"]

            set_head_tail = set()

            span2entID = {}
            span2etype = {}
            for entID, ent in enumerate(entities):
                span2entID[tuple(ent)] = entID
                span2etype[tuple(ent)] = ent[-1]

            span2relID_head = {}
            span2relID_tail = {}
            span2relID = {}
            for span in span2entID.keys():
                span2relID_head[span] = [0] * len(triples)
                span2relID_tail[span] = [0] * len(triples)
                span2relID[span] = [0] * len(triples)
                for relID, triple in enumerate(triples):
                    if span == tuple(triple[0]):
                        span2relID_head[span][relID] = 1
                        span2relID[span][relID] = 1
                    if span == tuple(triple[1]):
                        span2relID_tail[span][relID] = 1
                        span2relID[span][relID] = 1

            triples_ = triples
            relID2triples = dict()
            if repeat_gt_triples != -1 and len(triples) > 0:
                repeat_gt_triples_ = max(repeat_gt_triples, len(triples))
                k = repeat_gt_triples_ // len(triples)
                m = repeat_gt_triples_ % len(triples)
                triples_ = triples * k

                relID2triples = dict([(i, [len(triples)*j + i for j in range(k)])
                                    for i in range(len(triples))])

            for triple in triples_:
                head, tail, rel_type = triple[0], triple[1], triple[2]
                rel_type = rel_type if rel_type != "对应关系" else category
                relation_id = relational_alphabet.get_index(rel_type)
                h_start, h_end, h_mention, h_label = head
                t_start, t_end, t_mention, t_label = tail

                assert text[h_start: h_end] == h_mention
                assert text[t_start: t_end] == t_mention
                assert h_label == span2etype[tuple(head)]
                assert t_label == span2etype[tuple(tail)]

                try:
                    head_start_index, head_end_index = char_to_bep[h_start], char_to_bep[h_end-1]
                    tail_start_index, tail_end_index = char_to_bep[t_start], char_to_bep[t_end-1]
                except:
                    print(line['doc_key'])
                    print(tokenizer.tokenize(text))
                    print(char_to_bep)
                if head_start_index is None or head_end_index is None or tail_start_index is None or tail_end_index is None:
                    print(text)
                    print(text[h_start: h_end], text[t_start: t_end])
                    flag_invalid = True
                    continue

                max_len_mention = max(max_len_mention, len(h_mention))
                max_len_mention = max(max_len_mention, len(t_mention))

                target["relation"].append(relation_id)
                target["head_start_index"].append(head_start_index)
                target["head_end_index"].append(head_end_index)
                target["tail_start_index"].append(tail_start_index)
                target["tail_end_index"].append(tail_end_index)
                target["head_mention"].append(h_mention)
                target["tail_mention"].append(t_mention)
                target["head_type"].append(entity_type_alphabet.get_index(h_label))
                target["tail_type"].append(entity_type_alphabet.get_index(t_label))

                h_entID = span2entID[tuple(head)]
                t_entID = span2entID[tuple(tail)]
                target["head_entID"].append(h_entID)
                target["tail_entID"].append(t_entID)

                rel_entID_labels = [0] * len(entities)
                rel_entID_labels[h_entID] = 1
                rel_entID_labels[t_entID] = 1
                target["rel_entID_labels"].append(rel_entID_labels)

                head_part_labels = [0.0] * len(sent_id)
                for index in range(head_start_index, head_end_index+1):
                    head_part_labels[index] = 0.5
                head_part_labels[head_start_index] = 1.0
                head_part_labels[head_end_index] = 1.0
                target["head_part_labels"].append(head_part_labels)
                
                tail_part_labels = [0.0] * len(sent_id)
                for index in range(tail_start_index, tail_end_index+1):
                    tail_part_labels[index] = 0.5
                tail_part_labels[tail_start_index] = 1.0
                tail_part_labels[tail_end_index] = 1.0
                target["tail_part_labels"].append(tail_part_labels)

                set_head_tail.add(tuple(head))
                set_head_tail.add(tuple(tail))

            bio_labels = [0] * len(sent_id)

            entities_ = entities
            entID2entities = dict()
            if repeat_gt_entities != -1 and len(entities) > 0:
                repeat_gt_entities_ = max(repeat_gt_entities, len(entities))
                k = repeat_gt_entities_ // len(entities)
                m = repeat_gt_entities_ % len(entities)
                entities_ = entities * k
                entities_ += entities[:m]

                entID2entities = dict([(i, [len(entities)*j + i for j in range(k)])
                                    for i in range(len(entities))])
                for entID in entID2entities:
                    if entID < m:
                        entID2entities[entID].append(len(entities)*k + entID)

            for ent in entities_:
                ent_type = ent[-1]

                ent_type_id = entity_type_alphabet.get_index(ent_type)
                ent_start, ent_end = ent[0], ent[1]
                ent_start_index, ent_end_index = char_to_bep[ent_start], char_to_bep[ent_end-1]
                if ent_start_index is None or ent_end_index is None:
                    print(text)
                    print(text[ent_start: ent_end])
                    flag_invalid = True
                    continue

                if text[ent_start: ent_end] != ent[2]:
                    print('text[ent_start: ent_end] != ent[2]')
                    print(line)
                    exit(0)

                max_len_mention = max(max_len_mention, ent_end-ent_start)

                target["ent_type"].append(ent_type_id)
                target["ent_start_index"].append(ent_start_index)
                target["ent_end_index"].append(ent_end_index)
                target["relID_labels"].append(span2relID[tuple(ent)])
                target["relID_head_labels"].append(span2relID_head[tuple(ent)])
                target["relID_tail_labels"].append(span2relID_tail[tuple(ent)])

                ent_have_rel = 1 if tuple(ent) in set_head_tail else 0
                target["ent_have_rel"].append(ent_have_rel)

                bio_labels[ent_start_index] = 1
                for index in range(ent_start_index+1, ent_end_index+1):
                    bio_labels[index] = 2

                ent_part_labels = [0.0] * len(sent_id)
                for index in range(ent_start_index, ent_end_index+1):
                    ent_part_labels[index] = 0.5
                ent_part_labels[ent_start_index] = 1.0
                ent_part_labels[ent_end_index] = 1.0
                target["ent_part_labels"].append(ent_part_labels)

            if flag_invalid:
                exit(0)
                continue
            for _ in range(repeat_num):
                samples.append([idx, sent_id, target, bio_labels, text, bep_to_char, sent_seg_encoding, context2token_masks, token_masks, entID2entities, relID2triples, ori_pos, category])

        total_triples += len(triples)
        total_entities += len(entities)
        max_triples = max(max_triples, len(triples))
        max_entities = max(max_entities, len(entities))
        num_samples += 1

    print('[num samples]:', num_samples)
    print('[avg len]:', avg_len / num_samples)
    # print('[num triples]:', total_triples)
    # print('[avg triples]:', total_triples / num_samples)
    # print('[max triples]:', max_triples)
    # print('[num entities]:', total_entities)
    # print('[avg entities]:', total_entities / num_samples)
    # print('[max entities]:', max_entities)
    # print('[max len mention]:', max_len_mention)
    # print('[# samples without entity]:', num_no_entity)
    # print()
    # exit(0)

    return samples


class Data:
    def __init__(self):
        self.relational_alphabet = Alphabet("Relation", unkflag=False, padflag=False)
        self.train_loader = []
        self.valid_loader = []
        self.test_loader = []
        self.weight = {}

        self.entity_type_alphabet = Alphabet("Entity", unkflag=False, padflag=False)

    def show_data_summary(self):
        
        print("DATA SUMMARY START:")
        print("     Relation Alphabet Size: %s" % self.relational_alphabet.size())
        print(self.relational_alphabet.instances)
        print("     Ent Type Alphabet Size: %s" % self.entity_type_alphabet.size())
        print(self.entity_type_alphabet.instances)
        print("     Train  Instance Number: %s" % (len(self.train_loader)))
        print("     Valid  Instance Number: %s" % (len(self.valid_loader)))
        print("     Test   Instance Number: %s" % (len(self.test_loader)))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def generate_instance(self, args):
        if any(x in args.dataset_name for x in ['child', 'cancer', 'baidu']):
            tokenizer = BertTokenizerFast.from_pretrained(args.bert_directory, do_lower_case=True)
            self.train_loader = medical_data_process(args.train_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=False,
                                    repeat_gt_entities=args.repeat_gt_entities, repeat_gt_triples=args.repeat_gt_triples, max_len=args.max_len)
            self.weight = copy.deepcopy(self.relational_alphabet.index_num)
            self.test_loader = medical_data_process(args.test_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=True, max_len=args.max_len)
        else:
            raise ValueError(f'Unsupported dataset name {args.dataset_name}')

        self.relational_alphabet.close()
        self.entity_type_alphabet.close()


def build_data(args):
    data = Data()
    data.generate_instance(args)
    data.show_data_summary()
    return data


def save_data_setting(data, args):
    new_data = copy.deepcopy(data)
    data.show_data_summary()
    if not os.path.exists(args.generated_data_directory):
        os.makedirs(args.generated_data_directory)
    saved_path = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    with open(saved_path, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting is saved to file: ", saved_path)


def load_data_setting(args):

    saved_path = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    with open(saved_path, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting is loaded from file: ", saved_path)
    data.show_data_summary()
    return data


def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor

def padded_stack(tensors, padding=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked

def create_word_mask(start, end, context_size):
        mask = torch.zeros(context_size, dtype=torch.bool)
        mask[start:end] = 1
        return mask
