#!/bin/bash

echo "Starting bispn......"
echo "请在log/bispn.log中查看模型推理进度~"

cd ./src/BiSPN2

conda run --no-capture-output -n bispn \
        python ./+++infer.py \
            --dataset_name MediaReview \
            --task_id $1 \
            --max_len 512 \
            --bert_directory cpt_large \
            --ent_ths 2 \
            --rel_ths 1 \
            --pred_ent_fn pred_ent___.json \
            --pred_rel_fn pred_rel___.json \
            --max_span_length 20 \
            --ent_class_ths 0.3 \
            --rel_class_ths 0.45 \
            --boundary_ths 0.15 \
            --eval_batch_size 8 \
            --train_file "./+++data/_addtrain/train_tmp.json" \
            --class_fn "+++data/_addtrain/review_labels.json"\
            --model_ids "2280_v1" \
            --visible_gpus $2 \
            --batch_size $3 > ../../logs/bispn.log 2>&1

# 获取上一个命令的退出状态
EXIT_STATUS=$?
 
# 判断退出状态
if [ $EXIT_STATUS -eq 0 ]; then
    echo "程序正确退出，日志已保存到 logs/bispn.log"
else
    echo "程序异常退出，退出状态码：$EXIT_STATUS, 日志已保存到 logs/bispn.log"
    exit $EXIT_STATUS
fi

cd ..

echo "Bispn ends!"

