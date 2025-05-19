import json
from src.web_layer.utils.query import *
from src.utils.file_utils import *
from src.utils.filter import *
from config.path import *
info = {
  'task_id': 52, 
  'agency_name': '复旦大学附属华山医院', 
  'other_names': '华山', 
  'platform': '知乎', 
  'status': 4, 
  'user_id': 0, 
  'start_time': '2022-03-03', 
  'end_time': '2025-04-27', 
  'create_time': '2025-4-30-10-10-6', 
  'max_note': 10, 'max_comment': 50, 'if_level': 'False', 'gpu': '0', 'batch_size': 8, 'result': None
}

def crawl_process():
  print(info)
  reviews = []
  sample_num = 0
  num_tot = 0
  agencyFilter = AgencyFilter(info["agency_name"], info["other_names"])
  medicalTextCleaner = MedicalTextCleaner()

  platforms = info["platform"].split("///")
  for platform in platforms:
    num_tot_platform = 0
    num_ok_platform = 0
    #rows = CrawlDataRepository.get_comments(platform, info["task_id"])
    rows = query("select id, content, publish_time from zhihu_comment where task_id=%s", (info['task_id'], ), if_sys=False, type="select")
    for row in rows:
      num_tot += 1
      num_tot_platform += 1
      id, text, time = row
      text = medicalTextCleaner.pipeline(text)
      if not text:
        continue
      time_ok, target_time = time_filter(info["start_time"], info["end_time"], float(time))
      if_full, if_other, sims, variants = agencyFilter.agency_filter(text)
      if time_ok and if_full + if_other > 0:
        print(f"\n{text}")
        print(f"if_full:{if_full}  if_other:{if_other} sims:{[sims]} variants:{variants}")
        num_ok_platform += 1
        data_key = platform +'_'+str(info['task_id']) + '_' + str(id)
        #TaskDataRepository.create_taskdata(data_id=data_key, task_id=info["task_id"], platform=platform_map_inv[platform], time=target_time, text=text)
        query("insert into task_data(data_id,task_id,platform,time,text) values (%s,%s,%s,%s,%s)", (data_key, info["task_id"], platform, target_time, text))
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
        print(f"prompt_list:{prompt_list}")
        for prompt in prompt_list:
          reviews.append({'doc_key': data_key, 
                        'sent_id' : sample_num, 'text': f"{prompt}:{text}",
                        'original_position':[0,0],
                        'entities':[], 'relations':[]})
          sample_num += 1
    print(f'平台{platform}: {num_ok_platform}条数据被写入task_data中,过滤前的数据为{num_tot_platform}条')
    save_json(reviews, model_input_folder + str(info['task_id']) + '_.json')
    print(f"{sample_num}条数据已被写入{model_input_folder}{info['task_id']}_.json中")
    return sample_num
  
def output_process():
  task_id = info["task_id"]
  preds, trans_rels_list = model_output_filter(f"{model_output_folder}/{str(task_id)}_.rel", f"{filtered_model_output_folder}/{str(task_id)}_.rel")

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
  print(f"{result}\n\n{result_}")
  '''with app.app_context():
    TaskRepository.update_task_result(task_id, json.dumps(result_, ensure_ascii=False))
    for i, pred in enumerate(preds):
      TaskDataRepository.update_data_output(pred["doc_key"], str(trans_rels_list[i]))
    print(f"任务{task_id}:任务以及任务数据的结果已被写入相应数据表")'''
  
  '''query("update task set result=%s where task_id=%s", (json.dumps(result_, ensure_ascii=False), task_id,))
  for i, pred in enumerate(preds):
     query("update task_data set output=%s where data_id=%s", (str(trans_rels_list[i]), pred["doc_key"]))
  print(f"任务{task_id}:任务以及任务数据的结果已被写入相应数据表")'''
  taskdata_output_2_excel(task_id)
  print(f"任务{task_id}:任务数据已被写入excel表")

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

def taskdata_output_2_excel(task_id):
  #with app.app_context():
  #  datalist = TaskDataRepository.get_data_by_task(task_id)
  datalist = query("select * from task_data where task_id=%s", (task_id, ), type="select")
  rows = []
  for data in datalist:
    rows.append((data[0], data[1], data[2], data[3], data[4]))
  cols = ['评论编号', '平台', '发布时间', '评论内容', '关系输出']
  create_excel(cols, rows, title=f'任务{task_id}数据', path=f'{task_output_excel_folder}{task_id}_.xlsx')

output_process()