import re
import hashlib
from src.utils.file_utils import *
from config.path import *
from datetime import datetime
import Levenshtein
def levenshtein_similarity(s1, s2):
    """使用Levenshtein库计算相似度"""
    dist = Levenshtein.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1 - dist / max_len

def fuzzy_match(full, variant):
  if full.endswith("医院"):
    full = full[:-2]
  if variant.endswith("医院"):
    variant = variant[:-2]
  cnt = 0
  for c in variant:
    if c in full:
      cnt += 1
  return cnt / len(variant)
  

class MedicalTextCleaner:
  def __init__(self):
    self.keep_punct = r",，.。!！?？()（）、:：<>《》''’‘”“[] "
    self.ad_keywords = read_json(AD_LIST_PATH)
    self.seen_texts = set()

  def clean_text(self, text:str) -> str:
    if not text:
      return ""
    # 只保留中文、英文、数字和基本标点
    text = re.sub(
        rf"[^\u4e00-\u9fa5a-zA-Z0-9{re.escape(self.keep_punct)}\s]", 
        "", 
        text
    )
    
    # 合并连续空格
    text = re.sub(r"\s+", " ", text).strip()
    return text

  def is_ad(self, text: str) -> bool:
      """简单广告检测"""
      # 检查广告关键词
      if any(kw in text for kw in self.ad_keywords):
          return True
          
      # 检查联系方式
      if re.search(r"1[3-9]\d{9}|微信\w+|联系方式\w+|电话\w+", text):
          return True
          
      return False

  def remove_duplicate(self, text: str) -> bool:
      """基于MD5的去重检查"""
      text_md5 = hashlib.md5(text.encode()).hexdigest()
      if text_md5 in self.seen_texts:
          return True
      self.seen_texts.add(text_md5)
      return False
  
  def pipeline(self, raw_text: str) -> str:
      """完整处理流程"""
      if not raw_text or len(raw_text) < 5:  # 过滤短文本
          return None
          
      # 1. 基础清洗
      text = self.clean_text(raw_text)
      if not text:
          return None
          
      # 2. 去重检查
      if self.remove_duplicate(text):
          return None
          
      # 3. 广告过滤
      if self.is_ad(text):
          return None
          
      return text
  

class AgencyFilter:
  def __init__(self, agency_name=None, other_names=None):
    self.agency_name = agency_name
    self.patterns = [
      re.compile(r"(.+)省(?!.*中心医院)(.*)(中医院|妇幼保健院|中心)"),
      re.compile(r"(.+)省(.+)(医院)"),
      re.compile(r"(.+)市(.+)区(?!.*中心医院)(.*)(中医院|妇幼保健院|中心)"),
      re.compile(r"(.+)市(.+)区(.+)(医院)"),
      re.compile(r"(.+)(市|县)(?!.*中心医院)(.*)(中医院|妇幼保健院|中心)"),
      re.compile(r"(.+)(市|县)(.*)(医院)"),
      re.compile(r"(.+)(大学|学院)附属第([一二三四五六七八九])医院"),
      re.compile(r"(.+)(大学|学院)第([一二三四五六七八九])附属医院"),
      re.compile(r"(.+)(大学|学院)第([一二三四五六七八九])医院"),
      re.compile(r"(.+)(大学|学院)附属(.+)医院"),
      re.compile(r"(.+)(大学|学院)附属医院"),
      re.compile(r"(.+)(大学|学院)(.+)医院")
    ]
    write_agency_2_json(agency_name, other_names)
    self._generate_variants()
    try:
      self.agency_inv_list = read_json(AGENCY_INV_LIST_PATH)
      self.agency_list = read_json(AGENCY_LIST_PATH)
      self.agency_variants = self.agency_list[agency_name]
    except KeyError:
      self.agency_variants = []

  def check_conflict(self, variant, text):
    try:
      for full in self.agency_inv[variant]:
        check_list = self.agency_list[full]+ [full]
        for check in check_list:
          if check in text and check != variant:
            return True
      return False
    except Exception:
      return
  
  def agency_filter(self, text, thresh=0.8):
    variants = self.agency_variants
    if len(text) > 512 or not text:
      return -1, -1, [], []
    if_variant = 0
    if len(variants) > 0:
      sims = [99] * len(variants)
      for i, variant in enumerate(variants):
        if variant in text and not self.check_conflict(variant, text):
          if_variant = 1
          sims[i] = fuzzy_match(self.agency_name, variant)
    if self.agency_name in text:
      return 1, if_variant, sims, variants
    else:
      return 0, if_variant, sims, variants
    
      
  def _generate_variants(self):
    variant = set()
    for i, pat in enumerate(self.patterns):
      m = pat.match(self.agency_name)
      if not m:
        continue
      parts = m.groups()
      if i == 0:      # (.+)省(.*)(中医院|妇幼保健院|中心)
        variant.add(parts[0]+parts[1]+parts[2])
        if "妇幼保健院" in parts[2]:
          variant.add(parts[0]+parts[1]+"妇幼")
          variant.add(parts[0]+"省"+parts[1]+"妇幼")
        break
      elif i == 1:    # (.+)省(.+)(医院)
        variant.add(parts[0]+parts[1]+parts[2])
        break
      elif i == 2:    # (.+)市(.+)区(.*)(中医院|妇幼保健院|中心)
        variant.add(parts[0]+parts[1]+parts[2]+parts[3])
        variant.add(parts[0]+"市"+parts[1]+parts[2]+parts[3])
        variant.add(parts[0]+parts[1]+"区"+parts[2]+parts[3])
        variant.add(parts[1]+parts[2]+parts[3])
        variant.add(parts[1]+"区"+parts[2]+parts[3])
        if "妇幼保健院" in parts[3]:
          variant.add(parts[0]+parts[1]+parts[2]+"妇幼")
          variant.add(parts[0]+"市"+parts[1]+parts[2]+"妇幼")
          variant.add(parts[0]+parts[1]+"区"+parts[2]+"妇幼")
          variant.add(parts[0]+"市"+parts[1]+"区"+parts[2]+"妇幼")
          variant.add(parts[1]+parts[2]+"妇幼")
          variant.add(parts[1]+"区"+parts[2]+"妇幼")   
        break
      elif i == 3:    # (.+)市(.+)区(.+)(医院)
        variant.add(parts[0]+parts[1]+parts[2]+parts[3])
        variant.add(parts[0]+"市"+parts[1]+parts[2]+parts[3])
        variant.add(parts[0]+parts[1]+"区"+parts[2]+parts[3])
        variant.add(parts[1]+parts[2]+parts[3])
        variant.add(parts[1]+"区"+parts[2]+parts[3])
        break
      elif i == 4:    # (.+)(市|县)(.*)(中医院|妇幼保健院|中心)
        variant.add(parts[0]+parts[2]+parts[3])
        if "妇幼保健院" in parts[3]:
          variant.add(parts[0]+parts[2]+"妇幼")
          variant.add(parts[0]+parts[1]+parts[2]+"妇幼")
        break
      elif i == 5:    # (.+)(市|县)(.*)(医院)
        variant.add(parts[0]+parts[2]+parts[3])
        break
      elif i == 6:    # (.+)(大学|学院)附属第([一二三四五六七八九])医院
        variant.add(parts[0]+"附"+parts[2])           #中山附一
        variant.add(parts[0]+parts[1]+"附"+parts[2])  #中山大学附一
        variant.add(parts[0]+parts[2]+"院")           #中山一院
        variant.add(parts[0]+parts[1]+parts[2]+"院")  #中山大学一院
        variant.add(parts[0]+"附属"+parts[2]+"院")    #中山附属一院
        variant.add(parts[0]+parts[1]+"附属"+parts[2]+"院")    #中山大学附属一院
        variant.add(parts[0]+"附属第"+parts[2]+"医院")    #中山附属第一医院
        break
      elif i == 7:    # (.+)(大学|学院)第([一二三四五六七八九])附属医院
        variant.add(parts[0]+"附"+parts[2])           #安徽医科附一
        variant.add(parts[0]+parts[1]+"附"+parts[2])  #安徽医科大学附一
        variant.add(parts[0]+parts[2]+"院")           #安徽医科一院
        variant.add(parts[0]+parts[1]+parts[2]+"院")  #安徽医科大学一院
        variant.add(parts[0]+"附属"+parts[2]+"院")    #安徽医科附属一院
        variant.add(parts[0]+parts[1]+"附属"+parts[2]+"院")    #中安徽医科大学附属一院
        variant.add(parts[0]+"附属第"+parts[2]+"医院")    #安徽医科附属第一医院
        break
      elif i == 8:    # (.+)(大学|学院)第([一二三四五六七八九])医院
        variant.add(parts[0]+parts[2]+"院")
        variant.add(parts[0]+parts[1]+parts[2]+"院")
        variant.add(parts[0]+parts[2]+"医院")
        variant.add(parts[0]+parts[1]+parts[2]+"医院")
        break
      elif i == 9:    # (.+)(大学|学院)附属(.+)医院     #复旦大学附属中山医院
        variant.add(parts[0]+parts[2]+"医院")             # 复旦中山医院
        variant.add(parts[0]+parts[1]+parts[2]+"医院")    # 复旦大学中山医院
        variant.add(parts[0]+parts[2]+"附院")             # 复旦中山附院
        variant.add(parts[0]+parts[1]+parts[2]+"附院")    # 复旦大学中山附院
        variant.add(parts[2]+"医院")                      # 中山医院
        break
      elif i == 10:     # (.+)(大学|学院)附属医院       # 青岛大学附属医院
        variant.add(parts[0]+parts[1]+"附院")         # 青岛大学附院
        variant.add(parts[0]+"附院")                  # 青岛附院
        break
      elif i == 11:   # (.+)(大学|学院)(.+)医院       # 南方医科大学南方医院
        variant.add(parts[0]+parts[2]+"医院")       # 南方医科南方医院
        variant.add(parts[2]+"医院")                # 南方医院
        break
    variant = list(variant)
    if len(variant) > 0:
      print(variant)
      write_agency_2_json(self.agency_name, '///'.join(variant))


  

def time_filter(start, end, target_timestp):
  start_timestp = datetime.strptime(start + " 00:00:00", "%Y-%m-%d %H:%M:%S").timestamp()
  end_timestp = datetime.strptime(end + " 23:59:59", "%Y-%m-%d %H:%M:%S").timestamp()
  target_time = datetime.fromtimestamp(target_timestp).strftime('%Y-%m-%d %H:%M:%S') 
  return start_timestp <= target_timestp <= end_timestp, target_time

def model_output_filter(input_path, output_path):
  preds = read_json(input_path)
  preds_new = []
  trans_rels_list = []
  for i, pred in enumerate(preds):
    text = pred["text"]
    rels = pred["relations"]
    rels_new = []
    trans_rels = []
    prompt_len = len(text.split(":")[0]) + 1
    text_new = text[prompt_len:]
    for rel in rels:
      if rel[1][1] >= prompt_len and rel[2][1] >=prompt_len:
        rels_new.append([rel[0], [rel[1][0], rel[1][1]-prompt_len, rel[1][2]-prompt_len, rel[1][3]], [rel[2][0], rel[2][1]-prompt_len, rel[2][2]-prompt_len, rel[2][3]]])
        trans_rels.append(((rel[1][3],rel[1][0]),(rel[2][3],rel[2][0]+"-"+rel[0].split("-")[-1])))
    preds_new.append({"doc_key":pred["doc_key"], "sent_id":pred["sent_id"], "text":text_new, "relations":rels_new})
    trans_rels_list.append(trans_rels)
  save_json(preds_new, output_path)
  return preds_new, trans_rels_list



