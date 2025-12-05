from utils.file_utils import *

#datalist = read_json('./+++data/_with_prompt/data_combine.json')
datalist = read_json('./+++data/_addtrain/train_new3.json')

def check_boundary():
  for data in datalist:
    text = data['text']
    for ent in data['entities']:
      if text[ent[0] : ent[1]] != ent[2]:
        print(f"\ndoc_key: {data['doc_key']}")
        print(f"ent_start: {ent[0]}")
        print(text[ent[0] : ent[1]])
        print(f'entity_mention:{ent[2]}')
        print(text)
    for rel in data['relations']:
      if text[rel[0][0] : rel[0][1]] != rel[0][2]:
        print(f"\ndoc_key: {data['doc_key']}")
        print(f"\nhead_ent_start: {rel[0][0]}")
        print(text[rel[0][0] : rel[0][1]])
        print(f'head_entity_mention:{rel[0][2]}')
        print(text)
      if text[rel[1][0] : rel[1][1]] != rel[1][2]:
        print(f"\ndoc_key: {data['doc_key']}")
        print(f"\ntail_ent_start: {rel[1][0]}")
        print(text[rel[1][0] : rel[1][1]])
        print(f'tail_entity_mention:{rel[1][2]}')
        print(text)

def change_boundary():
  datalist_new = []
  for data in datalist:
    text_split = data['text'].split('&')
    if len(text_split) >=2:
      offset = len(text_split[0]) + 1
      text_new = text_split[0] + ':' + text_split[1]
    else:
      offset = 0
      text_new = text_split[0]
    data_new = {'doc_key': data['doc_key'], 'sent_id':data['sent_id'], 'text':text_new,
                'entities':[], 'relations':[]}
    for ent in data['entities']:
      data_new['entities'].append([ent[0] + offset, ent[1] + offset, ent[2], ent[3]])
    for rel in data['relations']:
      data_new['relations'].append([[rel[0][0] + offset, rel[0][1] + offset, rel[0][2], rel[0][3]],[rel[1][0] + offset, rel[1][1] + offset, rel[1][2], rel[1][3]],rel[-1]])
    datalist_new.append(data_new)
  save_json(datalist_new, './+++data/_with_prompt/train.json')

#change_boundary()
check_boundary()

'''datalist = read_json('./+++data/_addtrain/data.json')
agency_list = []
for train_data in datalist:
  for ent in train_data['entities']:
    if ent[-1] == '医疗机构':
      agency_list.append(ent[-2])
agency_list = list(set(agency_list))
print(agency_list)'''




