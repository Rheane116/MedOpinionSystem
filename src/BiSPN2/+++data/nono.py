from utils.file_utils import *
import json
from collections import defaultdict

def merge_and_adjust_samples(original_list, additional_list):
    """
    合并相同text的样本，并整合additional_list中的entities和relations
    
    参数:
        original_list: 原始样本列表
        additional_list: 提供额外entities和relations的列表
        
    返回:
        合并后的新样本列表
    """
    # 第一步：合并original_list中相同text的样本
    merged_dict = defaultdict(list)
    for sample in original_list:
        merged_dict[sample["text"].split(':')[1]].append(sample)
    
    # 创建基础合并样本
    merged_samples = []
    for text, samples in merged_dict.items():
        base_sample = samples[0].copy()  # 使用第一个样本作为基础
        base_sample["entities"] = []
        base_sample["relations"] = []
        
        # 合并所有相同text样本的entities和relations
        seen_entities = set()
        seen_relations = set()
        
        for sample in samples:
            for entity in sample["entities"]:
                entity_tuple = tuple(entity)
                if entity_tuple not in seen_entities:
                    seen_entities.add(entity_tuple)
                    base_sample["entities"].append(entity)
            
            for relation in sample["relations"]:
                # 将relation转为可哈希的元组
                rel_tuple = (
                    tuple(relation[0]), 
                    tuple(relation[1]), 
                    relation[2] if len(relation) > 2 else None
                )
                if rel_tuple not in seen_relations:
                    seen_relations.add(rel_tuple)
                    base_sample["relations"].append(relation)
        
        merged_samples.append(base_sample)
    
    # 第二步：整合additional_list中的entities和relations
    additional_dict = defaultdict(list)
    for sample in additional_list:
        additional_dict[sample["text"].split(':')[1]].append(sample)
    
    final_samples = []
    for sample in merged_samples:
        text = sample["text"].split(':')[1]
        if text not in additional_dict:
            final_samples.append(sample)
            continue
        
        # 计算文本前缀差异
        additional_samples = additional_dict[text]
        base_text = sample["text"]
        
        for add_sample in additional_samples:
            add_text = add_sample["text"]
            
            # 计算前缀差异长度
            prefix_diff = 0
            for i in range(min(len(base_text), len(add_text))):
                if base_text[i] != add_text[i]:
                    prefix_diff = len(add_text[:i])
                    break
            
            # 调整并合并entities
            for entity in add_sample["entities"]:
                adjusted_entity = [
                    entity[0] + prefix_diff,
                    entity[1] + prefix_diff,
                    entity[2],
                    entity[3]
                ]
                if adjusted_entity not in sample["entities"]:
                    sample["entities"].append(adjusted_entity)
            
            # 调整并合并relations
            for relation in add_sample["relations"]:
                adjusted_relation = [
                    [
                        relation[0][0] + prefix_diff,
                        relation[0][1] + prefix_diff,
                        relation[0][2],
                        relation[0][3]
                    ],
                    [
                        relation[1][0] + prefix_diff,
                        relation[1][1] + prefix_diff,
                        relation[1][2],
                        relation[1][3]
                    ],
                    relation[2] if len(relation) > 2 else None
                ]
                if adjusted_relation not in sample["relations"]:
                    sample["relations"].append(adjusted_relation)
        
        final_samples.append(sample)
    
    return final_samples

# 示例使用
if __name__ == "__main__":
 
    original_list = read_json('+++data/_addtrain/train_tmp.json')
    additional_list = read_json('+++data/_addtrain/data.json')
    
    # 合并并调整
    result = merge_and_adjust_samples(original_list, additional_list)
    
    save_json(result, '+++data/_no_prompt/train_merge.json')