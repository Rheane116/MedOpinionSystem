#from data_convert import *
import argparse
import json
import os
import random
from utils.file_utils import *

label_id_text_map = {}
object_id_detail_map = {}
data_list = []
data_num = 0

parser = argparse.ArgumentParser(description="Arguments of dataset construction.")
parser.add_argument('--version', type=str, default="_")
#parser.add_argument('--agency', type=str, default="医院")
args = parser.parse_args()


def label_extraction(allData, output_file):
    labellist = {'ent':[],'rel':[]}
    ent_label_list = allData['label_list']
    rel_label_list = allData['relation_list']
    for ent_label in ent_label_list:
        entity = {'id':None, 'text':None}
        id = ent_label['id']
        text = ent_label['text']
        entity['id'] = id
        entity['text'] = text
        labellist['ent'].append(entity)
        label_id_text_map[id] = text
    for rel_label in rel_label_list:
        relation = {'id':None, 'text':None}
        id = rel_label['id']
        text = rel_label['text']
        relation['id'] = id
        relation['text'] = text
        labellist['rel'].append(relation)
        label_id_text_map[id] = text
    
    save_json(labellist, output_file)


def trans_label(input_label_file):
    output_label_file_path = output_folder_path + 'review_labels.json'
    labellist_new = {'ent':[], 'rel':[]}
    labellist = read_json(input_label_file)

    ent_idx = labellist['ent'][0]['id']
    ent_text_new_set = []
    for i, ent in enumerate(labellist['ent']):
        ent_label = {}
        ent_text = ent['text'].split('-')
        if len(ent_text) == 1:
            ent_label['id'] = ent_idx
            ent_label['text'] = ent_text[0]
            ent_idx += 1
            labellist_new['ent'].append(ent_label)
        elif not ent_text[0] in ent_text_new_set:
            ent_text_new_set.append(ent_text[0])
            ent_label['id'] = ent_idx
            ent_label['text'] = ent_text[0]
            ent_idx += 1
            labellist_new['ent'].append(ent_label)
        #ent_idx += 1
        #labellist_new['ent'].append(ent_label)

    rel_idx = labellist['rel'][0]['id']
    labellist_new['rel'].append({'id':rel_idx, 'text':'评价-正'})
    labellist_new['rel'].append({'id':rel_idx + 1, 'text':'评价-负'})

    save_json(labellist_new, input_label_file)

def ER_extraction(allData, output_file):
    global data_num
    allData_list = allData['data']
    for d in allData_list:
        data = {'doc_key':None, 'sent_id' : 0, 'text':None, 
                'original_position':[0,0],
                'entities':[], 'relations':[]}

        data['doc_key'] = d['id']
        data['text'] = d['text']

        objects = d['myObjects']
        conns = d['connections']
        annotations = d['annotations']

        for object in objects:
            if object['connection']:
                continue
            ent = [None, None, None, None] 
            anno_id = object['annotation']
            ent[2] = object['text']

            for annotation in annotations:
                if annotation['id'] == anno_id:
                    ent[0] = annotation['start_offset']
                    #ent[1] = annotation['end_offset']
                    ent[1] = annotation['end_offset'] 
                    ent[3] = label_id_text_map[annotation['label']]
                    break
            data['entities'].append(ent)
            object_id_detail_map[object['id']] = ent

        for conn in conns:
            rel = [None, None, None]
            rel[0] = object_id_detail_map[conn['source']]
            rel[1] = object_id_detail_map[conn['to']]
            rel[2] = label_id_text_map[conn['relation']]

            data['relations'].append(rel)
        data_list.append(data)
        data_num += 1

    save_json(data_list, output_file)
    return data_list
    #print(f"{data_num}条数据已被写入{output_file}")
    #return data_list
    '''datalist_new = []
    for i, data in enumerate(data_list):
        data_new = {'doc_key': data['doc_key'],
                    'sent_id': data['sent_id'],
                    'text': data['text'],
                    'original_position': data['original_position'],
                    'entities': [],
                    'relations': []}   
        for ent in data['entities']:
            data_new['entities'].append([ent[0], ent[1], ent[2], (ent[3].split('-'))[0]])
        for rel in data['relations']:
            data_new['relations'].append([rel[0], [rel[1][0],rel[1][1],rel[1][2],rel[1][3].split('-')[0]], rel[-1] +'-'+rel[1][3].split('-')[-1]])
        datalist_new.append(data_new)

    #print('\n')
    #print(datalist_new[:2])
    save_json(datalist_new, output_file)
    print(f'已将提取后的结果写至{output_file}!')
    return datalist_new'''


def change_format(data, test=False):
    cnt = 0
    for i in range(len(data)):
        data[i]['doc_key'] = cnt
        cnt += 1
    return data

# 划分数据集
def split_data(data, output_folder_path, train_ratio=0.8, val_ratio=0.1):
    train_output_file = output_folder_path + 'train.json'
    val_output_file = output_folder_path + 'val.json'
    test_output_file = output_folder_path + 'test.json'
    val_test_output_file = output_folder_path + 'val_test.json'
    #test_raw_output_file = output_folder_path + 'test_raw.json'
    #test_val_output_file = output_folder_path + 'test_val.json'
    #test_train_output_file = output_folder_path + 'test_train.json'

    random.shuffle(data)
    total_size = len(data)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    print(f"训练集的大小为：{train_size}")
    print(f"验证集的大小为：{val_size}")
    print(f"测试集的大小为：{test_size}")
 
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    val_test_data = data[train_size:]

    train_data = change_format(train_data)
    save_json(train_data, train_output_file)
    val_data = change_format(val_data)
    save_json(val_data, val_output_file)
    test_data = change_format(test_data)
    save_json(test_data, test_output_file)
    val_test_data = change_format(val_test_data)
    save_json(val_test_data, val_test_output_file)
  

    #print('--------val_data--------')
    #print(val_data)
    #print('--------test_data--------')
    #print(test_data)
    #print('--------test_val_data--------')
    #print(test_val_data)
    #print('--------test_data_raw--------')
    #print(test_data)
 
    #save_json(train_data, train_output_file)
    #save_json(val_data, val_output_file)
    #save_json(test_data_raw, test_raw_output_file)
    #save_json(test_data, test_output_file)
    #save_json(test_val_data, test_val_output_file)
    #save_json(test_train_data, test_train_output_file)
    print(f"数据划分完成，并分别保存到 {train_output_file}, {val_output_file}, {test_output_file}")


if __name__== "__main__":
    '''input_file = '+++data/raw_data/file' + args.version + '.json'
    output_folder_path = '+++data/' + args.version + '/'
    #output_folder_path = '+++data/' + '_4_6_17_shuffle' + '/'
    output_data_file = output_folder_path + 'data.json'

            # 创建文件夹
    mkdir(output_folder_path)

    allData = read_json(input_file)
    allData_list = allData['data']

    label_extraction(allData, output_file = output_folder_path + 'review_labels.json')
    dataset = ER_extraction(allData, output_file = output_data_file)'''
    #trans_label(output_folder_path + 'review_labels.json')
    #dataset = replace_chars(dataset, old_char='：', new_char=':')
    #split_data(dataset, output_folder_path=output_folder_path)

    dataset = read_json('./+++data/_no_prompt/data_combine.json')
    split_data(dataset, output_folder_path='./+++data/_no_prompt/')
