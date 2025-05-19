import json
import os

def read_json(path):
  with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
  return data
 
def save_json(data, path):
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

def mkdir(path):
    if not os.path.exists(path):  # 检查文件夹是否已存在
        os.makedirs(path)         # 创建文件夹
        print(f"文件夹 '{path}' 创建成功！")
    else:
        print(f"文件夹 '{path}' 已存在！")

# 递归替换函数
def replace_chars(obj, old_char, new_char):
    if isinstance(obj, dict):
        return {k.replace(old_char, new_char): replace_chars(v, old_char, new_char) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_chars(item, old_char, new_char) for item in obj]
    elif isinstance(obj, str):
        return obj.replace(old_char, new_char)
    else:
        return obj

def replace_symbol_in_json(file_path, old_char,new_char):
    """
    将JSON文件中的所有字符串中的 old_char 替换为 new_char
    :param file_path: JSON文件路径
    :param old_char: 要替换的字符
    :param new_char: 替换后的字符
    """
    # 读取JSON文件
    data = read_json(file_path)
    # 执行替换
    modified_data = replace_chars(data, old_char, new_char)
    # 写回原文件
    save_json(modified_data, file_path)

