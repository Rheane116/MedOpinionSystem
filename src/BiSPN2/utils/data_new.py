from utils.alphabet import Alphabet
from transformers import BertTokenizer, BertTokenizerFast
import json
import copy
import os
import pickle
import sys

def review_data_process(input_doc, 
                        relational_alphabet, 
                        entity_type_alphabet, 
                        tokenizer, 
                        evaluate, 
                        repeat_gt_entities=-1, 
                        repeat_gt_triples=-1, 
                        max_len=495, max_ent=80):
    samples = []
    total_triples = 0
    total_entities = 0
    # 单个样本中triples/entities的最大数目
    max_triples = 0
    max_entities = 0
    # 文本中出现的实体的最大字符长度
    max_len_mention = 0
    num_samples = 0
    avg_len = 0 
    print(input_doc)

    avg_no_entity_len = 0
    num_no_entity = 0

    with open(input_doc) as f:
        lines = json.load(f)

    for idx, line in enumerate(lines):
        #print(line)
        doc_key = line["doc_key"]
        sent_id = line["sent_id"]
        text = line["text"]
        ori_pos = line['original_position'] if 'original_position' in line else (0, len(text))
        category = line['category'] if 'category' in line else None
        # 对Json文件中的每个样本的text进行分词，返回enc对象
        enc = tokenizer(text, add_special_tokens = True)
        # input_ids:文本中每个单词或子词(应该不是字符)在词汇表中的索引        
        sent_id = enc['input_ids']

        #if not evaluate and len(line['entities']) == 0:
        if len(line['entities']) == 0:
            avg_no_entity_len += len(sent_id)
            num_no_entity += 1
            continue
        if len(line['entities']) > max_ent:
            print(f"\n{line['text']}\n# entities({len(line['entities'])}) > max_ent({max_ent})")
            continue
        if len(sent_id) > max_len:
            # if 'test' not in input_doc and not category and doc_type:
            #     line, text, enc, sent_id = clip_text(line, tokenizer)
            # if len(sent_id) > max_len:
            #     print(f"{line['text']}\n# tokens({len(sent_id)}) > max_len({max_len})")
            continue

        avg_len += len(text)
        # 初始化一个与sent_id长度相同的列表，所有元素都设置为0。这个列表可能用于标记句子中每个词的某种状态或编码
        sent_seg_encoding = [0] * len(sent_id)
        # 初始化context2token_masks为None。这个变量可能用于存储某种与上下文相关的token掩码
        context2token_masks = None
        # 初始化一个与sent_id长度相同的列表，所有元素都设置为1。这个列表可能用于表示句子中每个词都是有效的或需要被处理的。
        token_masks = [1] * len(sent_id)

        # 字符索引 -> bep索引
        char_to_bep = dict()
        # bep索引 -> 字符范围(左边界，右边界)
        bep_to_char = dict()

        # i为text的字符索引
        for i in range(len(text)):
            bep_index = enc.char_to_token(i)
            char_to_bep[i] = bep_index
            if bep_index in bep_to_char:
                left, right = bep_to_char[bep_index][0], bep_to_char[bep_index][-1]
                bep_to_char[bep_index] = [left, max(right, i)]
            else:
                bep_to_char[bep_index] = [i, i]

        target = {  # 存储关系的类型(在rel_alphabet中的索引)
                    "relation" : [],
                    # 头/尾实体的开始/结束的索引
                        #(not_evaluate：bep索引， evaluate:char索引)
                    "head_start_index":[], "head_end_index":[], "tail_start_index":[], "tail_end_index":[],
                    # 头/尾实体的文本
                    "head_mention":[], "tail_mention":[],
                    # 头/尾实体的标记标签(以bep为单位)
                    "head_part_labels":[], "tail_part_labels":[],
                    # 头/尾实体的类型(在ent_alphabet中的索引)
                    "head_type": [], "tail_type": [],
                    # 头/尾实体的索引（当前样本中）
                    "head_entID": [], "tail_entID": [], 
                    # 存储关系涉及到的实体的索引(当前样本中,01列表)
                    "rel_entID_labels": [],

                    # 存储实体的类型(在ent_alphabet中的索引)
                    "ent_type": [], 
                    # 存储实体的起始/结束的bep索引
                    "ent_start_index": [], "ent_end_index": [], 
                    # 存储实体的标记标签
                    "ent_part_labels": [], 
                    # 标记实体是否有关联关系
                    'ent_have_rel': [],
                    # 用于存储实体所涉及到的关系（01列表）
                    "relID_labels": [], "relID_head_labels": [], "relID_tail_labels": []
                }   
        # 标记当前实体是否无效
        flag_invalid = False

        if evaluate:
            triples = line['relations']    
            for triple in triples:
                rel_type = triple[-1]    
                relation_id = relational_alphabet.get_index(rel_type)
                # 数据标注时， 结束index比实际上多1
                #print(line)
                #print(triple)
                h_start, h_end, h_mention, h_label = triple[0]
                t_start, t_end, t_mention, t_label = triple[1]

                #assert text[h_start:h_end] == h_mention
                #assert text[t_start: t_end] == t_mention
                '''if text[h_start:h_end + 1] != h_mention:
                    continue
                if text[t_start: t_end + 1] != t_mention:
                    continue'''
                if text[h_start:h_end] != h_mention:
                    continue
                if text[t_start: t_end] != t_mention:
                    continue

                max_len_mention = max(max_len_mention, len(h_mention))
                max_len_mention = max(max_len_mention, len(t_mention))

                target["relation"].append(relation_id)
                target["head_start_index"].append(h_start)
                target["head_end_index"].append(h_end - 1)
                target["tail_start_index"].append(t_start)
                target["tail_end_index"].append(t_end - 1)
                target["head_mention"].append(h_mention)
                target["tail_mention"].append(t_mention)
                target["head_type"].append(entity_type_alphabet.get_index(h_label))
                target["tail_type"].append(entity_type_alphabet.get_index(t_label))

            entities = line['entities']
            for ent in entities:
                ent_type_id = entity_type_alphabet.get_index(ent[-1])
                ent_start, ent_end = ent[0], ent[1]
                '''ent_start_index, ent_end_index = char_to_bep[ent_start], char_to_bep[ent_end]'''
                ent_start_index, ent_end_index = char_to_bep[ent_start], char_to_bep[ent_end - 1]

                if ent_start_index is None or ent_end_index is None:
                    print("\nent_start_index is None or ent_end_index is None")
                    print(text)
                    print(text[ent_start: ent_end])
                    flag_invalid = True
                    continue

                target["ent_type"].append(ent_type_id)
                target["ent_start_index"].append(ent_start_index)
                target["ent_end_index"].append(ent_end_index)

            if flag_invalid:
                #exit(0)
                continue
            samples.append([idx, sent_id, target, None, text, bep_to_char, sent_seg_encoding, context2token_masks, token_masks, None, None, ori_pos, category])

        else:
            repeat_num = 1
            triples = line['relations']       
            entities = line['entities']

            set_head_tail = set()

            span2entID = {}     # 映射： 实体tuple -> 实体ID
            span2etype = {}     # 映射： 实体tuple -> 实体类型

            for i, ent in enumerate(entities):
                span2entID[tuple(ent)] = i
                span2etype[tuple(ent)] = ent[-1]

            span2relID_head = {}    # 映射： 实体tuple -> 作为头实体涉及的关系
            span2relID_tail = {}    # 映射： 实体tuple -> 作为尾实体涉及的关系
            span2relID = {}         # 映射： 实体tuple -> 涉及的所有关系

            #print(text)
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
            relID2triples = dict()  # 映射：关系ID -> 重复后的关系ID

            if repeat_gt_triples != -1 and len(triples) > 0:
                repeat_gt_triples_ = max(repeat_gt_triples, len(triples))
                k = repeat_gt_triples_ // len(triples)
                m = repeat_gt_triples_ % len(triples)
                triples_ = triples * k
                relID2triples = dict([(i, [len(triples) * j + i for j in range(k)])
                                for i in range(len(triples))])
            
            for triple in triples_:
                head, tail, rel_type = triple[0], triple[1], triple[2]
                relation_id = relational_alphabet.get_index(rel_type)
                h_start, h_end, h_mention, h_label = head
                t_start, t_end, t_mention, t_label = tail

                #print(f"sample{idx}:\n\ttriple:{triple}")
                #assert text[h_start : h_end] == h_mention
                #assert text[t_start : t_end] == t_mention
                #assert span2etype[tuple(head)] == h_label
                #assert span2etype[tuple(tail)] == t_label
                if text[h_start : h_end] != h_mention:
                    continue
                if text[t_start : t_end] != t_mention:
                    continue
                if span2etype[tuple(head)] != h_label:
                    continue
                if span2etype[tuple(tail)] != t_label:
                    continue


                try:
                    '''head_start_index, head_end_index = char_to_bep[h_start], char_to_bep[h_end]
                    tail_start_index, tail_end_index = char_to_bep[t_start], char_to_bep[t_end]'''
                    head_start_index, head_end_index = char_to_bep[h_start], char_to_bep[h_end - 1]
                    tail_start_index, tail_end_index = char_to_bep[t_start], char_to_bep[t_end - 1]
                except:
                    # 如果转换失败，则打印相关信息进行调试
                    print(line['doc_key'])
                    print(tokenizer.tokenize(text))
                    print(char_to_bep)
                if head_start_index is None or head_end_index is None or tail_start_index is None or tail_end_index is None:
                    print("\nhead_start_index is None or head_end_index is None or tail_start_index is None or tail_end_index is None")
                    print(text)
                    print(text[h_start: h_end], text[t_start: t_end])
                    flag_invalid = True
                    continue

                max_len_mention = max(max_len_mention, len(h_mention))
                max_len_mention = max(max_len_mention, len(t_mention))

                target['relation'].append(relation_id)
                target['head_start_index'].append(head_start_index)
                target['head_end_index'].append(head_end_index)
                target['tail_start_index'].append(tail_start_index)
                target['tail_end_index'].append(tail_end_index)
                target['head_mention'].append(h_mention)
                target['tail_mention'].append(t_mention)
                target['head_type'].append(entity_type_alphabet.get_index(h_label))
                target['tail_type'].append(entity_type_alphabet.get_index(t_label))

                h_entID = span2entID[tuple(head)]
                t_entID = span2entID[tuple(tail)]
                target['head_entID'].append(h_entID)
                target['tail_entID'].append(t_entID)

                rel_entID_labels = [0] * len(entities)
                rel_entID_labels[h_entID] = 1
                rel_entID_labels[t_entID] = 1
                target['rel_entID_labels'].append(rel_entID_labels)

                head_part_labels = [0.0] * len(sent_id)
                for index in range(head_start_index, head_end_index + 1):
                    head_part_labels[index] = 0.5
                head_part_labels[head_start_index] = 1.0
                head_part_labels[head_end_index] = 1.0
                target['head_part_labels'].append(head_part_labels)

                tail_part_labels = [0.0] * len(sent_id)
                for index in range(tail_start_index, tail_end_index + 1):
                    tail_part_labels[index] = 0.5
                tail_part_labels[tail_start_index] = 1.0
                tail_part_labels[tail_end_index] = 1.0
                target['tail_part_labels'].append(tail_part_labels)

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

                entID2entities = dict([(i, [len(entities) * j + i for j in range(k)]) 
                            for i in range(len(entities))])
                for entID in entID2entities:
                    if entID < m:
                        entID2entities[entID].append(len(entities)*k + entID)
            
            for ent in entities_:
                ent_type = ent[-1]
                ent_type_id = entity_type_alphabet.get_index(ent_type)
                ent_start, ent_end = ent[0], ent[1]
                ent_start_index, ent_end_index = char_to_bep[ent_start], char_to_bep[ent_end - 1]

                if ent_start_index is None or ent_end_index is None:
                    print("\nent_start_index is None or ent_end_index is None:")
                    print(text)
                    print(text[ent_start: ent_end + 1])
                    flag_invalid = True
                    continue

                if text[ent_start: ent_end] != ent[2]:
                    print('\ntext[ent_start: ent_end + 1] != ent[2]')
                    print(line)
                    #exit(0)
                    continue

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
                for index in range(ent_start_index+1, ent_end_index + 1):
                    bio_labels[index] = 2

                ent_part_labels = [0.0] * len(sent_id)
                for index in range(ent_start_index, ent_end_index + 1):
                    ent_part_labels[index] = 0.5
                ent_part_labels[ent_start_index] = 1.0
                ent_part_labels[ent_end_index] = 1.0
                target["ent_part_labels"].append(ent_part_labels)

            if flag_invalid:
                #exit(0)
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
    #print('[num triples]:', total_triples)
    print('[avg triples]:', total_triples / num_samples)
    print('[max triples]:', max_triples)
    #print('[num entities]:', total_entities)
    print('[avg entities]:', total_entities / num_samples)
    print('[max entities]:', max_entities)
    print('[max len mention]:', max_len_mention)
    print('[# samples without entity]:', num_no_entity)
    print()
    # exit(0)

    return samples

class Data:
    def __init__(self):
        self.relational_alphabet = Alphabet("Relation", unkflag=False, padflag=False)
        self.train_loader = []
        self.valid_loader = []
        self.test_loader = []
        self.test_val_loader = []
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
        #if any(x in args.dataset_name for x in ['child', 'cancer', 'baidu']):
        tokenizer = BertTokenizerFast.from_pretrained(args.bert_directory, do_lower_case=True)
        self.train_loader = review_data_process(args.train_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=False,
            repeat_gt_entities=args.repeat_gt_entities, repeat_gt_triples=args.repeat_gt_triples, max_len=args.max_len, max_ent=args.max_ent)
        self.weight = copy.deepcopy(self.relational_alphabet.index_num)
        self.valid_loader = review_data_process(args.valid_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=True)
        self.test_loader = review_data_process(args.test_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=True)
        #self.test_val_loader = review_data_process(args.test_val_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=True)


        self.relational_alphabet.close()
        self.entity_type_alphabet.close()
    
def build_data(args):

    file = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    # if os.path.exists(file) and not args.refresh:
    #     data = load_data_setting(args)
    # else:
    data = Data()
    data.generate_instance(args)
    save_data_setting(data, args)
    # exit(0)
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