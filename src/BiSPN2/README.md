# 部署说明

模型为[BiSPN](https://aclanthology.org/2023.findings-emnlp.136/)，训练采用多场景数据联合学习范式，推理时为保证准确率进行多模型集成（多线程加速）。

## 环境配置

请保证机器至少有一张Nvidia显卡，且显卡驱动支持的CUDA版本大于等于11.7。即命令行执行nvidia-smi后能看到打印出来的表第一行显示CUDA Version: 11.7或更高。

若未安装anaconda请先自行安装。然后使用conda命令安装python3.8环境（命名为bispn）：

```bash
conda create -n bispn python=3.8
conda activate bispn
pip install -r requirements.txt
```

## 模型下载

使用百度网盘下载预训练语言模型CPT-Large（[链接](https://pan.baidu.com/s/1-ksgbdm55kICEHyPVr1cIQ?pwd=qztn)），解压后将其中的所有文件（pytorch_model.bin等）放到cpt-large目录内。

使用百度网盘下载训练好的模型权重（[链接](https://pan.baidu.com/s/1ebTkVWzLfMRJBQJQp2Up0w?pwd=x8km)），解压后将其中的所有文件放到checkpoints目录内。

## （推理）输入数据格式

对于现病史、家族史、个人史、病理报告、婚育史、既往史6种病历文本，需要将待处理数据转成以下JSON格式

```json
[
  {
    "doc_key": "既往史.0",
    "sent_id": 0,
    "text": "【既往史】 疾病史：有“小三阳”病史10年...否认手术史、外伤史。",
    "original_position": [
      0,
      0
    ],
    "entities": [],
    "relations": [],
    "doc_type": "既往史"
  },
  {
    "doc_key": "病理报告.99",
    "sent_id": 0,
    "text": "【病理报告】 (肝右前叶) 腺癌...PIK3CA基因第20外显子未检测到突变。",
    "original_position": [
      0,
      0
    ],
    "entities": [],
    "relations": [],
    "doc_type": "病理报告"
  }
]
```

其中，"doc_type"为文本种类（现病史|家族史|个人史|病理报告|婚育史|既往史）；"doc_key"格式为文本种类 + "." + 文本ID（ID可以为任意数字字母组合但要保证独特性）；"sent_id"置为0即可；"original_position"置为[0,0]即可；"entities"和"relations"均置为[]即可；"text"格式为【文本种类】+ 英文空格 + 待处理文本。

test.json是一个示例。

对于CT、SPECT、X射线、超声、核磁5种影像报告文本，输入数据格式同上（test2.json是一个示例）。注意不要将病历文本数据和影像报告文本数据混合放在同一个JSON文件内。

## （推理）运行

运行以下命令以处理病历文本数据

```bash
python infer.py \
    --dataset_name baidu_cancer \
    --train_file data/baidu_cancer/train_dev_aug.json \
    --test_file test.json \
    --max_len 512 \
    --bert_directory cpt-large \
    --model_ids "['03-02-19-57-40','03-02-17-26-05','03-02-17-21-51','03-02-20-12-31','03-03-13-25-37']" \
    --class_fn data/baidu_cancer/baidu_cancer_classes.json \
    --ent_ths 2 \
    --rel_ths 1 \
    --pred_ent_fn pred_ent___.json \
    --pred_rel_fn pred_rel___.json \
    --max_span_length 20 \
    --ent_class_ths 0.3 \
    --rel_class_ths 0.45 \
    --boundary_ths 0.15 \
    --eval_batch_size 32 \
    --visible_gpus "0,1"
```

其中，可指定"test_file"为其他待处理文件路径；可根据显卡显存设置"eval_batch_size"；可根据显卡数量设置"visible_gpus"。

实体输出会被保存到"pred_ent_fn"对应的文件路径；关系输出会被保存到"pred_rel_fn"对应的文件路径。

处理影像报告数据的命令和上面的类似，只有几个参数不同（详见infer_med_image.sh）。

## 训练数据格式
参考data/baidu_cancer或data/baidu_image两个目录内的数据格式。

train_ang.json等带"_ang"的文件预先进行了数据增强，就是在已有数据太少的情况下根据已有数据扩充部分数据，具体怎么扩充可自由发挥。data/baidu_cancer数据集的预处理脚本见"百度数据预处理/parse.py"。
data/baidu_image数据集的预处理脚本见"百度数据预处理/parse_image.py"。设置其中的"do_augment"为False或True以关闭或启动数据增强。

## （训练）运行
参考train_med_record.sh或train_med_image.sh两个训练脚本训练单个模型。单个模型训练完成后会自动保存在验证集上效果最好的模型到checkpoints目录下（文件命名为Start RE Training对应的时间 + '.model'后缀）。设置不同random_seed以训练不同模型。
