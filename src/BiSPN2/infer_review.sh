python ./+++infer.py \
    --dataset_name MediaReview \
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
    --test_file "./+++data/_addtrain/train_tmp.json" \
    --class_fn "+++data/_addtrain/review_labels.json"\
    --model_ids "2280_v1" \
    --visible_gpus "3" \
    --batch_size 8 

