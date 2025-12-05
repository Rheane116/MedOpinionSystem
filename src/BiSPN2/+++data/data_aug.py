#from ..utils.file_utils import *
import random
import json

def read_json(path):
  with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
  return data
 
def save_json(data, path):
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)



train_datalist = read_json('./+++data/_addtrain/train_tmp.json')
#agency_list = list(set([train_data['text'].split(':')[0] for train_data in train_datalist]))
agency_list_false = ['A医院', 'B医院', 'C医院', 'D医院', 'E医院', 'F医院', 'G医院', 'H医院', 'I医院']
'''agency_list = []
for train_data in train_datalist:
  for ent in train_data['entities']:
    if ent[-1] == '医疗机构':
      agency_list.append(ent[-2])'''

datalist = read_json('./+++data/_addtrain/data.json')
agency_list = list(set([data['text'].split(':')[0] for data in datalist]))
agency_list = list(set(agency_list))


'''print(agency_list)
print(f'相异医疗机构的数量:{len(agency_list)}')'''

text_list = list(set([data['text'].split(':')[1] for data in train_datalist]))

#print(len(text_list))
#print(len(train_datalist))


add_sample_list = read_json('./+++data/_addtrain/train_tmp.json')

replace_sample_num = 0
add_num = 0

#print(train_datalist)
#print(agency_list)
#print(len(train_datalist))
for train_data in train_datalist:
  if train_data['entities'] == []:
    continue
  text = train_data['text']
  prompt = text.split(':')[0]
  agency_mentions = []
  for ent in train_data['entities']:
    if ent[3] == '医疗机构':
      agency_mentions.append(ent[2])
  #print(text)
  flag = False
  for agency in agency_list:
    if flag == True:
      break
    if agency == prompt or agency in agency_mentions or agency == '北大' or agency=='千佛山医院' or agency =='复旦大学中山医院' or agency=='西南医院':
      continue
    idx = text.find(agency)
    #print(text)
    #print(f'typeof(idx):{type(idx)}')
    if idx == -1 or idx <= len(prompt) + 1:
      continue
    print(f"text:{text}")
    print(f"agency:{agency}")
    replace_sample_num += 1
    replace_agency_list  = random.sample(agency_list_false, 4)
    for replace_agency in replace_agency_list:
      add_sample = {'doc_key': train_data['doc_key'], 'sent_id': 0, 'text':None, 'ori_pos':[0,0], 'entities':[], 'relations':[]}
      if replace_agency == prompt:
        continue
      add_num += 1
      add_sample['text'] = text[:idx] + replace_agency + text[idx + len(agency):]
      print(f"    replace_agency:{replace_agency}")
      print(f"   len_diff:{len(replace_agency) - len(agency)}")
      print(f"   idx:{idx}")
      for ent in train_data['entities']:
        if ent[0] >= idx:
          add_sample['entities'].append([ent[0]+len(replace_agency) - len(agency),ent[1] + len(replace_agency) - len(agency), ent[2], ent[3]])
        else:
          add_sample['entities'].append(ent)
      for rel in train_data['relations']:
        if rel[0][0] >= idx:
          head = [rel[0][0] + len(replace_agency) - len(agency), rel[0][1] + len(replace_agency) - len(agency), rel[0][2], rel[0][3]]
        else:
          head = [rel[0][0], rel[0][1], rel[0][2], rel[0][3]]
        if rel[1][0] >= idx:
          tail = [rel[1][0] + len(replace_agency) - len(agency), rel[1][1] + len(replace_agency) - len(agency), rel[1][2], rel[1][3]]
        else:
          tail = [rel[1][0], rel[1][1], rel[1][2], rel[1][3]]
        
        add_sample['relations'].append([head, tail, rel[-1]])
      add_sample_list.append(add_sample)
      flag = True
      


print('-------Final----------')
print(f'训练集总样本数：{len(train_datalist)}')
print(f'训练集中相异文本的数量：{len(text_list)}')
print(agency_list)
print(f'相异医疗机构的数量:{len(agency_list)}')
print("湘雅附一" in agency_list)
print(f'replace_sample_num={replace_sample_num}')
print(f'add_num={add_num}')
#print(add_sample_list)
save_json(add_sample_list, './+++data/_addtrain/train_new3.json')


      


