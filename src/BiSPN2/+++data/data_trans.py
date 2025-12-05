import argparse

import os
from data import read_json, save_json, change_format, split_data
from utils.file_utils import *

parser = argparse.ArgumentParser(description="Arguments of data transformation.")
parser.add_argument('--version', type=str, default="_")
#parser.add_argument('--agency', type=str, default="医院")
args = parser.parse_args()

input_data_file = '+++data/' + args.version + '/data.json'
input_label_file = '+++data/' + args.version + '/review_labels.json'
output_folder_path = '+++data/' + args.version + '_new/'
output_data_file = output_folder_path + 'data.json'
train_output_file = output_folder_path + 'train.json'
val_output_file = output_folder_path + 'val.json'
test_output_file = output_folder_path + 'test.json'


        # 创建文件夹
if not os.path.exists(output_folder_path):  # 检查文件夹是否已存在
    os.makedirs(output_folder_path)         # 创建文件夹
    print(f"文件夹 '{output_folder_path}' 创建成功！")
else:
    print(f"文件夹 '{output_folder_path}' 已存在！")

def trans_label():
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

    save_json(labellist_new, output_label_file_path)


def trans_data():
    output_data_file_path = output_folder_path + 'data.json'

    datalist = read_json(input_data_file)
    datalist_new = []
    for i, data in enumerate(datalist):
        data_new = {'doc_key': data['doc_key'],
                    'sent_id': data['sent_id'],
                    'text': data['text'],
                    'original_position': data['original_position'],
                    'entities': [],
                    'relations': []}
        for ent in data['entities']:
            ent[3] = ent[3].split('-')[0]
            data_new['entities'].append(ent)
        for rel in data['relations']:
            pos_neg = rel[1][-1].split('-')[-1]
            rel[1][-1] = rel[1][-1].split('-')[0]
            rel[-1] = "评价-" + pos_neg
            data_new['relations'].append(rel)
        datalist_new.append(data_new)

    save_json(datalist_new, output_data_file_path)
    return datalist_new

trans_label()
datalist = trans_data()
split_data(datalist, version=args.version + '_new')