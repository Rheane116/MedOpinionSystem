import argparse, os, json
from threading import Thread
from copy import deepcopy
import json
import sys
import time

time_start = time.time()

def str2bool(v):
    return v.lower() in ('true')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg

def get_args():
    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed

parser = argparse.ArgumentParser()
data_arg = add_argument_group('Data')

data_arg.add_argument('--dataset_name', type=str, default="MediaReview")
data_arg.add_argument('--version', type=str, default="_")
data_arg.add_argument('--train_file', type=str, default="+++data/_4_6_17_new/train.json")
data_arg.add_argument('--valid_file', type=str, default="+++data/_4_6_17_new/val.json")
data_arg.add_argument('--test_file', type=str, default="+++data/_4_6_17_new/test.json")
data_arg.add_argument('--task_id', type=str, default=None)
data_arg.add_argument('--task_m_id', type=str, default=None)

#data_arg.add_argument('--if_sys', type=str2bool, default=False)
data_arg.add_argument('--max_len', type=int, default=512)
data_arg.add_argument('--generated_data_directory', type=str, default="./+++data/generated_data/")
data_arg.add_argument('--checkpoint_directory', type=str, default="./+++checkpoints")
data_arg.add_argument('--prediction_directory', type=str, default="./+++predictions_infer/")
data_arg.add_argument('--bert_directory', type=str, default="./cpt_large")
data_arg.add_argument("--partial", type=str2bool, default=False)
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--model_name', type=str, default="Set-Prediction-Networks")
learn_arg.add_argument('--repeat_gt_entities', type=int, default=25)
learn_arg.add_argument('--repeat_gt_triples', type=int, default=15)
learn_arg.add_argument('--num_generated_triples', type=int, default=40)
learn_arg.add_argument('--entity_queries_num', type=int, default=80)
learn_arg.add_argument('--num_decoder_layers', type=int, default=3)
learn_arg.add_argument('--matcher', type=str, default="avg", choices=['avg', 'min'])
learn_arg.add_argument('--hybrid', type=str2bool, default=True)
learn_arg.add_argument('--consistency_loss_weight', type=float, default=1)
learn_arg.add_argument('--start_consistency_epoch', type=int, default=0)
learn_arg.add_argument('--na_rel_coef', type=float, default=1)
learn_arg.add_argument('--rel_loss_weight', type=float, default=1)
learn_arg.add_argument('--head_ent_loss_weight', type=float, default=0.5)
learn_arg.add_argument('--tail_ent_loss_weight', type=float, default=0.5)
learn_arg.add_argument('--na_ent_coef', type=float, default=1)
learn_arg.add_argument('--ent_type_loss_weight', type=float, default=1)
learn_arg.add_argument('--ent_span_loss_weight', type=float, default=1)
learn_arg.add_argument('--ent_part_loss_weight', type=float, default=1)
learn_arg.add_argument('--head_part_loss_weight', type=float, default=1)
learn_arg.add_argument('--tail_part_loss_weight', type=float, default=1)
learn_arg.add_argument('--head_tail_type_loss_weight', type=float, default=1)
learn_arg.add_argument('--head_tail_entID_loss_weight', type=float, default=1)
learn_arg.add_argument('--relID_loss_weight', type=float, default=1)
learn_arg.add_argument('--ent_have_rel_loss_weight', type=float, default=1)
learn_arg.add_argument('--start_ent_have_rel_epoch', type=int, default=0)
learn_arg.add_argument('--stop_ent_have_rel_epoch', type=int, default=100)
learn_arg.add_argument('--fix_bert_embeddings', type=str2bool, default=True)
learn_arg.add_argument('--batch_size', type=int, default=8)
learn_arg.add_argument('--eval_batch_size', type=int, default=8)
learn_arg.add_argument('--max_epoch', type=int, default=50)
learn_arg.add_argument('--start_eval', type=int, default=40)
learn_arg.add_argument('--gradient_accumulation_steps', type=int, default=1)
learn_arg.add_argument('--decoder_lr', type=float, default=2e-5)
learn_arg.add_argument('--encoder_lr', type=float, default=1e-5)
learn_arg.add_argument('--weight_decay', type=float, default=1e-5)
learn_arg.add_argument('--dropout', type=float, default=0.2)
learn_arg.add_argument('--max_grad_norm', type=float, default=0)
learn_arg.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'])
learn_arg.add_argument('--save_model', type=str2bool, default=True)
learn_arg.add_argument('--lr_warmup', type=float, default=0.1,
                        help="Proportion of total train iterations to warmup in linear increase/decrease schedule")
# PIQN
learn_arg.add_argument('--prop_drop', type=float, default=0.1, help="Probability of dropout used in piqn")
learn_arg.add_argument('--freeze_transformer', type=str2bool, default=False, help="Freeze BERT weights")
learn_arg.add_argument('--pos_size', type=int, default=25)
learn_arg.add_argument('--char_lstm_layers', type=int, default=1)
learn_arg.add_argument('--lstm_layers', type=int, default=2)
learn_arg.add_argument('--char_size', type=int, default=25)
learn_arg.add_argument('--char_lstm_drop', type=float, default=0.2)
learn_arg.add_argument('--use_glove', type=str2bool, default=False)
learn_arg.add_argument('--use_pos', type=str2bool, default=False)
learn_arg.add_argument('--use_char_lstm', type=str2bool, default=False)
learn_arg.add_argument('--use_lstm', type=str2bool, default=False)
learn_arg.add_argument('--pool_type', type=str, default = "max")
learn_arg.add_argument('--wordvec_path', type=str, default = "../glove/glove.6B.300d.txt")
learn_arg.add_argument('--share_query_pos', type=str2bool, default=True)
learn_arg.add_argument('--use_token_level_encoder', type=str2bool, default=True)
learn_arg.add_argument('--num_token_ent_rel_layer', type=int, default=4)
learn_arg.add_argument('--num_token_ent_layer', type=int, default=1)
learn_arg.add_argument('--num_token_rel_layer', type=int, default=1)
learn_arg.add_argument('--num_token_head_tail_layer', type=int, default=1)
learn_arg.add_argument('--use_entity_attention', type=str2bool, default=True)
learn_arg.add_argument('--use_aux_loss', type=str2bool, default=True)
learn_arg.add_argument('--split_epoch', type=int, default=0, help="")
# PIQN EntityAwareConfig
learn_arg.add_argument('--mask_ent2tok', type=str2bool, default=True)
learn_arg.add_argument('--mask_tok2ent', type=str2bool, default=False)
learn_arg.add_argument('--mask_ent2ent', type=str2bool, default=False)
learn_arg.add_argument('--mask_entself', type=str2bool, default=True)
learn_arg.add_argument('--word_mask_ent2tok', type=str2bool, default=True)
learn_arg.add_argument('--word_mask_tok2ent', type=str2bool, default=False)
learn_arg.add_argument('--word_mask_ent2ent', type=str2bool, default=False)
learn_arg.add_argument('--word_mask_entself', type=str2bool, default=True)
learn_arg.add_argument('--entity_aware_attention', type=str2bool, default=False)
learn_arg.add_argument('--entity_aware_selfout', type=str2bool, default=False)
learn_arg.add_argument('--entity_aware_intermediate', type=str2bool, default=False)
learn_arg.add_argument('--entity_aware_output', type=str2bool, default=False)
learn_arg.add_argument('--use_entity_pos', type=str2bool, default=True)
learn_arg.add_argument('--use_entity_common_embedding', type=str2bool, default=True)

evaluation_arg = add_argument_group('Evaluation')
evaluation_arg.add_argument('--n_best_size', type=int, default=50)
evaluation_arg.add_argument('--max_span_length', type=int, default=20)
evaluation_arg.add_argument('--ent_class_ths', type=float, default=0.3)
evaluation_arg.add_argument('--rel_class_ths', type=float, default=0.45)
evaluation_arg.add_argument('--boundary_ths', type=float, default=0.15)
evaluation_arg.add_argument('--combined_f1', type=str2bool, default=False)
misc_arg = add_argument_group('MISC')
misc_arg.add_argument('--refresh', type=str2bool, default=True)
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument('--print_pred', type=str2bool, default=True)
misc_arg.add_argument('--visible_gpus', type=str, default="3")
misc_arg.add_argument('--random_seed', type=int, default=1)

misc_arg.add_argument('--model_ids', type=str, default='04-08-11-16-16')
misc_arg.add_argument('--model_id', type=str, default=None)
misc_arg.add_argument('--class_fn', type=str, default="+++data/_4_6_17_new/review_labels.json")
misc_arg.add_argument('--ent_ths', type=int, default=2)
misc_arg.add_argument('--rel_ths', type=int, default=1)
misc_arg.add_argument('--pred_ent_fn', type=str, default=None)
misc_arg.add_argument('--pred_rel_fn', type=str, default=None)

args, unparsed = get_args()

if args.task_id:
    print("----task-----")
    args.test_file = f"../../shared/task/model_input/{args.task_id}_.json"
    args.prediction_directory = "../../shared/task/model_output/"
    #args.test_file = f"./+++data/{args.task_id}/test.json"
elif args.task_m_id:
    print("----task_m-----")
    args.test_file = f"../../shared/manual/model_input/{args.task_m_id}_.json"
    args.prediction_directory = "../../shared/manual/model_output/"


os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

#for arg in vars(args):
#    print(arg, ":",  getattr(args, arg))

import torch

torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(24/32, device=0)

import random
import numpy as np
#from utils.data_infer import build_data
from utils.data_infer_new import build_data
#from utils.data_new import build_data
from trainer.trainer import Trainer
from models.setpred4RE import SetPred4RE
from models.setpred4RE_cpt import SetPred4RE_cpt
from utils.ensemble import ensemble, pred_2_json
from utils.ensemble_image import ensemble_image

# 创建 Thread 的子类
class MyThread(Thread):
    def __init__(self, func, args):
        '''
        :param func: 可调用的对象
        :param args: 可调用对象的参数
        '''
        Thread.__init__(self)
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        self.result = self.func(*self.args)

    def getResult(self):
        return self.result


def func(engine: Trainer, device_id: int, model_id: str):
    engine.to_cuda_device(device_id)
    engine.load_state_dict(args.checkpoint_directory + "/%s.model" % model_id)
    result = engine.eval_model(data.test_loader, data.entity_type_alphabet, data.relational_alphabet,
                                    log_fn=os.path.join(args.prediction_directory, model_id), print_pred=False)
    return result

def chunk_list(lst, size):
    return [lst[i:i+size] for i in range(0, len(lst), size)]

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(args.random_seed)

data = build_data(args)

if 'cpt' in args.bert_directory:
    model = SetPred4RE_cpt(args, data.relational_alphabet.size(), data.entity_type_alphabet.size(), RE=True)
else:
    model = SetPred4RE(args, data.relational_alphabet.size(), data.entity_type_alphabet.size(), RE=True)



engine = Trainer(model, data, args, args.max_epoch, args.start_eval)

engine.to_cuda_device(0)
print(f"\nmodel_ids:{args.model_ids}\n")
print(f"\ntrain_file:{args.train_file}\n")
print(f"\nclass_fn:{args.class_fn}\n")
print(f"\ntest_file:{args.test_file}\n")
engine.load_state_dict(args.checkpoint_directory + "/%s.model" % args.model_ids)

time_before_eval = time.time()

if args.task_id:
    engine.eval_model(data.test_loader, data.entity_type_alphabet, data.relational_alphabet,
                    log_fn=os.path.join(args.prediction_directory, args.task_id + '_'), print_pred=True)
elif args.task_m_id:
    engine.eval_model(data.test_loader, data.entity_type_alphabet, data.relational_alphabet,
                    log_fn=os.path.join(args.prediction_directory, args.task_m_id + '_'), print_pred=True)
elif args.test_file:
    from datetime import datetime
    engine.eval_model(data.test_loader, data.entity_type_alphabet, data.relational_alphabet,
                    log_fn=os.path.join(args.prediction_directory, datetime.now().strftime('%m-%d-%H-%M-%S')), print_pred=True)
      
time_end_eval = time.time()
print(f"Time taken before evalution: {time_before_eval - time_start} seconds")
print(f"Time taken for evaluation: {time_end_eval - time_before_eval} seconds")
'''engine = Trainer(model, data, args, args.max_epoch, args.start_eval, delay_to_gpu=True)

num_gpu = torch.cuda.device_count()
model_ids = eval(args.model_ids)
device_ids = list(range(num_gpu)) * (len(model_ids) // num_gpu + 1)'''

'''threads = []
for device_id, model_id in zip(device_ids, model_ids):
    t = MyThread(func, (deepcopy(engine), device_id, model_id))
    threads.append(t)

for threads_ in chunk_list(threads, num_gpu):
    # 启动线程运行
    for t in threads_:
        t.start()
    # 等待所有线程执行完毕
    for t in threads_:
        t.join()'''

'''ent_preds = []
rel_preds = []
for t in threads:
    ent_preds.append(t.getResult()['ent_pred'])
    rel_preds.append(t.getResult()['rel_pred'])'''

'''doc_type_2_classes = json.load(open(args.class_fn))
if args.dataset_name == 'baidu_cancer':
    ensemble_func = ensemble 
elif args.dataset_name == 'baidu_image':
    ensemble_func = ensemble_image
else:
    raise ValueError(f'Unsupported dataset name {args.dataset_name}')

doc_key_2_text, doc_key_2_relations, doc_key_2_entities = ensemble_func(
    None, None, args.ent_ths, args.rel_ths, doc_type_2_classes, rel_preds, ent_preds
)
pred_2_json(doc_key_2_text, doc_key_2_relations, doc_key_2_entities, args.pred_ent_fn, args.pred_rel_fn)'''

#model.load_state_dict(args.model_ids)

