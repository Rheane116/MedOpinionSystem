import json
from src.utils.file_utils import *
from src.utils.filter import *
from config.path import *
from src.data_access_layer.repositories import CrawlDataRepository, TaskDataRepository, TaskRepository, TaskManualRepository
from src.service_layer.system_service import TaskDataService


def crawl_process(info, app):
  reviews = []
  sample_num = 0
  num_tot = 0
  agencyFilter = AgencyFilter(info["agency_name"], info["other_names"])
  medicalTextCleaner = MedicalTextCleaner()
  with app.app_context():
    platforms = info["platform"].split("///")
    for platform in platforms:
      num_tot_platform = 0
      num_ok_platform = 0
      rows = CrawlDataRepository.get_comments(platform, info["task_id"]) + CrawlDataRepository.get_contents(platform, info["task_id"])
      for row in rows:
        num_tot += 1
        num_tot_platform += 1
        id, text, time = row.id, row.content, row.create_time
        if platform == "小红书":
          time /= 1000
        text = medicalTextCleaner.pipeline(text)
        if not text:
          continue
        time_ok, target_time = time_filter(info["start_time"], info["end_time"], float(time))
        if_full, if_other, sims, variants = agencyFilter.agency_filter(text)
        if time_ok == False or if_full + if_other <= 0:
          continue
        num_ok_platform += 1
        data_key = platform +'_'+str(info['task_id']) + '_' + str(id)
        TaskDataRepository.create_taskdata(data_id=data_key, task_id=info["task_id"], platform=platform, time=target_time, text=text)
        prompt_list = []
        if if_full:
          prompt_list.append(info["agency_name"])
        if if_other:
          for i, variant in enumerate(variants):
            if sims[i] < 0.8:
              prompt_list.append(variant)
            else:
              prompt_list.append(info["agency_name"])
        prompt_list = list(set(prompt_list))
        for i, prompt in enumerate(prompt_list):
          reviews.append({'doc_key': f"{data_key}_{i}", 
                        'sent_id' : sample_num, 'text': f"{prompt}:{text}",
                        'original_position':[0,0],
                        'entities':[], 'relations':[]})
          sample_num += 1
      print(f'平台{platform}: {num_ok_platform}条数据被写入task_data中,过滤前的数据为{num_tot_platform}条')
  save_json(reviews, model_input_folder + str(info['task_id']) + '_.json')
  print(f"{sample_num}条数据已被写入{model_input_folder}{info['task_id']}_.json中")
  return sample_num

def manual_input_process(info):
  reviews = []
  sample_num = 0
  num_tot = 0
  agencyFilter = AgencyFilter(info["agency_name"], info["other_names"])
  medicalTextCleaner = MedicalTextCleaner()
  texts = read_txt(f"{manual_input_folder}{info['task_id']}_.txt", info['sep'])
  for i, text in enumerate(texts):
    num_tot += 1
    text = medicalTextCleaner.pipeline(text)
    if not text:
      continue
    if_full, if_other, sims, variants = agencyFilter.agency_filter(text)
    if if_full + if_other <= 0:
      continue
    prompt_list = []
    if if_full:
      prompt_list.append(info["agency_name"])
    if if_other:
      for j, variant in enumerate(variants):
        if sims[j] < 0.8:
          prompt_list.append(variant)
        else:
          prompt_list.append(info["agency_name"])
    prompt_list = list(set(prompt_list))
    for j, prompt in enumerate(prompt_list):
      reviews.append({
        "doc_key": str(info["task_id"]) + "_" + str(i) + "_" + str(j),
        "sent_id": i,
        "text": f"{prompt}:{text}",
        "original_position":[0, 0],
        "entities":[], "relations":[]
      })
      sample_num += 1
  save_json(reviews, manual_model_input_folder + str(info["task_id"]) + "_.json")
  print(f"{sample_num}条数据已被写入{manual_model_input_folder}{info['task_id']}_.json中")
  return sample_num


def output_process(info, app, manual=False):
  task_id = info["task_id"]
  if manual == False:
    preds, trans_rels_list = model_output_filter(f"{model_output_folder}/{str(task_id)}_.rel", f"{filtered_model_output_folder}/{str(task_id)}_.rel")
  else:
    preds, trans_rels_list = model_output_filter(f"{manual_model_output_folder}/{str(task_id)}_.rel", f"{manual_filtered_model_output_folder}/{str(task_id)}_.rel")

  result = {}
  for h_label in head_labels:
    result[h_label] = {}
    for t_label in tail_labels:
      result[h_label][t_label] = [0, 0]
  for pred in preds:
    head_tailtype_set = set()
    for rel in pred["relations"]:
      rel_type = rel[0]
      head = rel[1]
      tail = rel[2]
      if (rel_type, tuple(head), tail[0]) not in head_tailtype_set:
        if head[0] in head_labels and tail[0] in tail_labels:
          if rel_type.split("-")[-1] == "正":
            result[head[0]][tail[0]][0] += 1
          else:
            result[head[0]][tail[0]][1] += 1
        head_tailtype_set.add((rel_type, tuple(head), tail[0]))
  result_ = sumup(result)
  with app.app_context():
    if manual == False:
      TaskRepository.update_task_result(task_id, json.dumps(result_, ensure_ascii=False))
      for i, pred in enumerate(preds):
        #TaskDataRepository.update_data_output(pred["doc_key"], str(trans_rels_list[i]))
        TaskDataService.update_data_output( "_".join(pred["doc_key"].split("_")[:-1]), trans_rels_list[i])
      print(f"任务{task_id}:任务以及任务数据的结果已被写入相应数据表和excel表格")
    else:
      TaskManualRepository.update_taskManual_result(task_id, json.dumps(result_, ensure_ascii=False))
      taskdata_output_2_excel_manual(task_id, trans_rels_list)
      print(f"任务{task_id}:任务的结果已被写入相应数据表和excel表格")


def sumup(result):
    sumup = {}
    # 主题：[正向数目，负向数目，总数占比] 
    head_labels_ = head_labels + ["所有主体"]
    tail_labels_= tail_labels + ["所有主题"]
    for h_label in head_labels_:
        sumup[h_label] = {}
        for t_label in tail_labels_:
            sumup[h_label][t_label] = [0, 0, -1]

    per_head_count = {}
    for h_label in head_labels:
      per_head_count[h_label] = 0

    for h_label, h_data in result.items():
        for t_label, t_data in h_data.items():
            pos_num = t_data[0]
            neg_num = t_data[1]
            sumup[h_label][t_label][0] = pos_num
            sumup[h_label][t_label][1] = neg_num
            sumup["所有主体"][t_label][0] += pos_num
            sumup["所有主体"][t_label][1] += neg_num
            sumup[h_label]["所有主题"][0] += pos_num
            sumup[h_label]["所有主题"][1] += neg_num
            per_head_count[h_label] += pos_num + neg_num
            sumup["所有主体"]["所有主题"][0] += pos_num
            sumup["所有主体"]["所有主题"][1] += neg_num

    sum_tot = sumup["所有主体"]["所有主题"][0] +sumup["所有主体"]["所有主题"][1]
    if sum_tot == 0:
      return {}
    for h_label in head_labels:
      sumup[h_label]["所有主题"][-1] = round((sumup[h_label]["所有主题"][0] + sumup[h_label]["所有主题"][1]) / sum_tot, 3)
      if per_head_count[h_label] == 0:
        continue
      for t_label in tail_labels:
        sumup[h_label][t_label][-1] = round((sumup[h_label][t_label][0] + sumup[h_label][t_label][1]) / per_head_count[h_label], 3)

    for t_label in tail_labels:
      sumup["所有主体"][t_label][-1] = round((sumup["所有主体"][t_label][0] + sumup["所有主体"][t_label][1]) / sum_tot, 3)

    sumup["所有主体"]["所有主题"][-1] = round(sumup["所有主体"]["所有主题"][0] / sum_tot, 3)
    return sumup  





  


