from utils.file_utils import *
from collections import defaultdict

combine_list = read_json('./+++data/_with_prompt/test_split.json')
split_list = read_json('./+++data/_no_prompt/data_tmp.json')
new_list = []

merged_dict = defaultdict(list)
for sample in split_list:
    merged_dict[sample["text"].split(':')[1]].append(sample)

for combine in combine_list:
    text = combine['text']
    if text in merged_dict.keys():
        for sample in merged_dict[text]:
          new_list.append(sample)

save_json(new_list, './+++data/_with_prompt/test.json')