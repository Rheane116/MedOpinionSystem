import torch, random, gc, os, json
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import AdamW
from datetime import timedelta, datetime
from collections import defaultdict
from utils.average_meter import AverageMeter
from utils.functions import formulate_gold_, formulate_gold_ent, formulate_gold
from utils.metric import metric_, ent_metric, metric, ent_metric_overlap, metric_overlap
from scorer.biorelex import evaluate_biorelex
from scorer.ade import evaluate_ade

# 带有两个阶段预热和线性衰减的学习率调度器
def get_linear_schedule_with_warmup_two_stage( # 要调整学习率的优化器
                                              optimizer, 
                                              # 第一阶段的学习率预热步数
                                              num_warmup_steps_stage_one, 
                                              # 第一阶段的总训练步数
                                              num_training_steps_stage_one,
                                              # 第二阶段的学习率预热步数
                                              num_warmup_steps_stage_two, 
                                              # 第一阶段的总训练步数
                                              num_training_steps_stage_two,  
                                              # 最后一个epoch的索引，默认为-1，表示从头开始。
                                              last_epoch=-1):
    # 计算当前步数的学习率乘法因子
    def lr_lambda(current_step: int):
        current_step += 1
        # 第一阶段
        if current_step < num_training_steps_stage_one:
            # 第一阶段的预热期内
            if current_step < num_warmup_steps_stage_one:
                # 线性增加学习率
                return float(current_step) / float(max(1, num_warmup_steps_stage_one))
            # 如果已经过了第一阶段的预热期，进入衰减期
            return max(
                0.0, float(num_training_steps_stage_one - current_step) / float(max(1, num_training_steps_stage_one - num_warmup_steps_stage_one))
            )
        # 第二阶段
        else:
            # 将当前步数转换为第二阶段的步数
            current_step = current_step - num_training_steps_stage_one
            # 如果还在第二阶段的预热期内
            if current_step < num_warmup_steps_stage_two:
                # 线性增加学习率
                return float(current_step) / float(max(1, num_warmup_steps_stage_two))
            # 注意这里的最小学习率设置为0.01而不是0.0，这是一个超参数选择
            return max(
                0.01, float(num_training_steps_stage_two - current_step) / float(max(1, num_training_steps_stage_two - num_warmup_steps_stage_two))
            )
    # 返回LambdaLR对象，它使用自定义的lr_lambda函数来调整学习率
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class Trainer(nn.Module):
    def __init__(self, model, data, args, max_epoch, start_eval_epoch, delay_to_gpu=False):
        super().__init__()
        self.args = args
        self.model = model
        self.data = data

        self.max_epoch = max_epoch
        self.start_eval_epoch = start_eval_epoch
        self.save_model = args.save_model
        #os.makedirs(args.checkpoint_directory, exist_ok=True)
        #os.makedirs(args.prediction_directory, exist_ok=True)

        self.start_eval = args.start_eval

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        grouped_params = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert' in n],
                'weight_decay': args.weight_decay,
                'lr': args.encoder_lr,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and 'bert' in n],
                'weight_decay': 0.0,
                'lr': args.encoder_lr,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if
                           not any(nd in n for nd in no_decay) and 'bert' not in n],
                'weight_decay': args.weight_decay,
                'lr': args.decoder_lr,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if
                           any(nd in n for nd in no_decay) and 'bert' not in n],
                'weight_decay': 0.0,
                'lr': args.decoder_lr,
            }
        ]

        #print(grouped_params)
        
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(grouped_params)
        elif args.optimizer == 'AdamW':
            self.optimizer = AdamW(grouped_params)
        else:
            raise Exception("Invalid optimizer.")
        if not delay_to_gpu and args.use_gpu:
            self.cuda()

    def to_cuda_device(self, device_id):
        self.cuda(device_id)
        self.model.set_device_id(device_id)

    def train_model(self):
        #best_f1 = 0
        best_f1 = 0
        best_f1_overlap = 0
        train_loader = self.data.train_loader
        train_num = len(train_loader)
        batch_size = self.args.batch_size
        total_batch = train_num // batch_size + 1
        # 在第一阶段中应该训练的轮数（模型权重应该被更新的总次数）
        updates_total_stage_one = total_batch * self.args.split_epoch
        # 在第二阶段中应该训练的轮数（模型权重应该被更新的总次数）
        updates_total_stage_two = total_batch * (self.max_epoch - self.args.split_epoch)
        scheduler = get_linear_schedule_with_warmup_two_stage(self.optimizer,
                                                    num_warmup_steps_stage_one = self.args.lr_warmup * updates_total_stage_one,
                                                    num_training_steps_stage_one = updates_total_stage_one,
                                                    num_warmup_steps_stage_two = self.args.lr_warmup * updates_total_stage_two,
                                                    num_training_steps_stage_two = updates_total_stage_two)

        start_datetime_str = datetime.now().strftime('%m-%d-%H-%M-%S')
        if self.model.RE:
            print('\n----------- Start RE Training -----------', start_datetime_str)
        else:
            print('\n----------- Start NER Training -----------', start_datetime_str)

        for epoch in range(self.max_epoch):
            # Train
            # torch.nn.Module.train()将模块（及所有子模块）设置为训练模式
            self.model.train()
            self.model.zero_grad()

            if epoch == 0:
                # print('Freeze Decoder.')
                # for name, param in self.model.decoder.named_parameters():
                #     param.requires_grad = False
                
                # 选择性地冻结编码器中的某些参数
                # （将name中不含"entity"或"triple"的param的requires_grad属性置为0，在训练过程中不会被更新）
                print("Freeze bert weights")
                for name, param in self.model.encoder.model.named_parameters():
                    if "entity" not in name and "triple" not in name:
                        param.requires_grad = False
            # 在第二阶段解冻参数
            if epoch == self.args.split_epoch:
                print("Now, update bert weights.")
                for name, param in self.model.encoder.model.named_parameters():
                    param.requires_grad = True
                # Embedding层中的参数保持冻结
                if self.args.fix_bert_embeddings:
                    self.model.encoder.model.embeddings.word_embeddings.weight.requires_grad = False
                    self.model.encoder.model.embeddings.position_embeddings.weight.requires_grad = False
                    self.model.encoder.model.embeddings.token_type_embeddings.weight.requires_grad = False
                # 调用self.optimizer.__setstate__({'state': defaultdict(dict)})重置了优化器的状态。
                # 这通常是为了在更改了哪些参数需要梯度之后，清除优化器内部关于之前哪些参数被更新的任何记忆。
                    # 然而，需要注意的是，直接调用__setstate__方法通常不是重置优化器状态的推荐方式。
                    # 更常见和更安全的做法是使用优化器的state_dict()和load_state_dict()方法来保存和加载状态，或者简单地创建一个新的优化器实例。
                self.optimizer.__setstate__({'state': defaultdict(dict)})

            print("\n\n=== Epoch %d train ===" % epoch, flush=True)
            avg_loss = AverageMeter()
            # 随机打乱
            random.shuffle(train_loader)

            for batch_id in range(total_batch):
                # 当前batch的开始/结束样本索引
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num

                # 从train_loader中获取当前batch的数据
                train_instance = train_loader[start:end]
                # print([ele[0] for ele in train_instance])

                if not train_instance:
                    continue
                
                # 从训练样本中提取信息
                    # attention_mask/ token_masks : 一个seq中有token的那些位置（输入的时候用的是一个batch的最大seq_len）
                input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, targets, _, info= self.model.batchify(train_instance)
                loss = self.model(input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, targets, info, epoch=epoch)[0]
                avg_loss.update(loss.item(), 1)
                # Optimize
                loss.backward()

                if self.args.max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                if (batch_id + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
            # for param_group in self.optimizer.param_groups:
            #     print(param_group['lr'])

            print("     Instance: %d; loss: %.4f" % (end, avg_loss.avg), flush=True)
            if epoch >= self.start_eval_epoch:
                # Validation
                print("\n=== Epoch %d Validation ===" % epoch)
                result, result_overlap = self.eval_model(self.data.valid_loader, self.data.entity_type_alphabet, self.data.relational_alphabet)
                if self.args.combined_f1:
                    f1 = (result['f1'] + result['entity_f1']) / 2
                    f1_overlap = (result_overlap['f1'] + result_overlap['entity_f1']) / 2
                else:
                    f1 = result['f1'] if self.model.RE else result['entity_f1']
                    f1_overlap = result_overlap['f1'] if self.model.RE else result_overlap['entity_f1']
                    #f1 = (result['f1'] + result['entity_f1']) / 2
                if f1 >= best_f1:
                    print("Achieving Best Result on Validation Set.", flush=True)
                    if self.save_model:
                        #torch.save(self.model.state_dict(), self.args.checkpoint_directory + "/%s.model" % start_datetime_str)
                        try:
                            torch.save(self.model.state_dict(), self.args.checkpoint_directory + "/%s.model" % start_datetime_str)
                            print("Model saved successfully.")
                        except Exception as e:
                            print("Failed to save model:", e)
                    best_f1 = f1
                    best_result_epoch = epoch
                if f1_overlap >= best_f1_overlap:
                    print("Achieving Best Result on Validation Set(!With Overlap).", flush=True)
                    if self.save_model:
                        #torch.save(self.model.state_dict(), self.args.checkpoint_directory + "/%s.model" % start_datetime_str)
                        try:
                            torch.save(self.model.state_dict(), self.args.checkpoint_directory + "/%s_overlap.model" % start_datetime_str)
                            print("Model saved successfully.")
                        except Exception as e:
                            print("Failed to save model:", e)
                    best_f1_overlap = f1_overlap
                    best_result_epoch_overlap = epoch
                    # # Test
                    # print("=== Epoch %d Test ===" % epoch, flush=True)
                    # result = self.eval_model(self.data.test_loader, self.data.entity_type_alphabet, self.data.relational_alphabet)


        end_datetime_str = datetime.now().strftime('%m-%d-%H-%M-%S')
        if self.model.RE:
            print('\n----------- Finish RE Training -----------', end_datetime_str)
        else:
            print('\n----------- Finish NER Training -----------', end_datetime_str)
        # if self.save_model:
        #     torch.save(self.model.state_dict(), self.args.checkpoint_directory + "/%s.model" % end_datetime_str)
        print("Best result on validation set is achieved at epoch %d." % best_result_epoch, flush=True)
        print("Best result on validation set(!With Overlap) is achieved at epoch %d." % best_result_epoch_overlap, flush=True)
        print("=== Final Test === ", flush=True)
        print('---no overlap')
        self.load_state_dict(self.args.checkpoint_directory + "/%s.model" % start_datetime_str)
        result = self.eval_model(self.data.test_loader, self.data.entity_type_alphabet, self.data.relational_alphabet,
                                log_fn=os.path.join(self.args.prediction_directory, start_datetime_str), print_pred=self.args.print_pred)
        #self.load_state_dict(self.args.checkpoint_directory + "/%s.model" % end_datetime_str)
        #result = self.eval_model(self.data.test_loader, self.data.entity_type_alphabet, self.data.relational_alphabet)
        print('---with overlap')
        self.load_state_dict(self.args.checkpoint_directory + "/%s_overlap.model" % start_datetime_str)
        result = self.eval_model(self.data.test_loader, self.data.entity_type_alphabet, self.data.relational_alphabet,
                                log_fn=os.path.join(self.args.prediction_directory,  start_datetime_str + '_overlap'), print_pred=self.args.print_pred)

    def eval_model(self, eval_loader, ent_type_alphabet, relation_alphabet, log_fn=None, print_pred=False):

        if 'child' in self.args.dataset_name:
            return self.default_eval_model(eval_loader, ent_type_alphabet, relation_alphabet, log_fn, print_pred, eval_ent=True)
        elif 'baidu' in self.args.dataset_name:
            return self.default_eval_model(eval_loader, ent_type_alphabet, relation_alphabet, log_fn, print_pred, eval_ent=True, remove_overlap=True)
        elif 'cancer' in self.args.dataset_name:
            return self.default_eval_model(eval_loader, ent_type_alphabet, relation_alphabet, log_fn, print_pred, eval_ent=False)
        elif self.args.dataset_name == 'Text2DT':
            return self.default_eval_model(eval_loader, ent_type_alphabet, relation_alphabet, log_fn, print_pred)
        elif self.args.dataset_name == 'biorelex':
            return self.biorelex_eval_model(eval_loader, ent_type_alphabet, relation_alphabet, log_fn, print_pred)
        elif self.args.dataset_name == 'ade':
            return self.ade_eval_model(eval_loader, ent_type_alphabet, relation_alphabet, log_fn, print_pred)
        else:
            #return self.default_eval_model(eval_loader, ent_type_alphabet, relation_alphabet, log_fn, print_pred, eval_ent=True)
            return self.default_eval_model(eval_loader, ent_type_alphabet, relation_alphabet, log_fn, print_pred, eval_ent=True, remove_overlap=True)

    def default_eval_model(self, eval_loader, ent_type_alphabet, relation_alphabet, log_fn=None, print_pred=False, eval_ent=False, filtrate=True, remove_overlap=False):
        self.model.eval()
        # print(self.model.decoder.query_embed.weight)
        prediction, gold_ = {}, {}
        prediction_ent, gold_ent = {}, {}
        list_text = []
        list_category = []
        import time
        time1 = time.time()
        with torch.no_grad():
            batch_size = self.args.eval_batch_size
            eval_num = len(eval_loader)
            total_batch = eval_num // batch_size + 1
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > eval_num:
                    end = eval_num
                eval_instance = eval_loader[start:end]
                if not eval_instance:
                    continue  # input_ids attention_mask token_masks: torch.size([bsz, max_seq_len])
                input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, target, _, info= self.model.batchify(eval_instance, is_test=True)
                # print(target) #每个sample: 
                gen_triples, gen_entities = self.model.default_predict(
                    input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, info, ent_type_alphabet, relation_alphabet, filtrate,
                )
                # print(batch_id, flush=True)
                if self.model.RE:
                    gold_.update(formulate_gold(target, info, relation_alphabet, ent_type_alphabet))
                    prediction.update(gen_triples)
                if eval_ent:
                    gold_ent.update(formulate_gold_ent(target, info, ent_type_alphabet))
                    prediction_ent.update(gen_entities)
                list_text += info['text']
                #list_category += info['category']
        print(time.time()-time1)
        scores = dict()
        scores_overlap = dict() 
        if eval_ent:
            ent_scores = ent_metric(prediction_ent, gold_ent, list_text, log_fn, print_pred)
            ent_scores_overlap = ent_metric_overlap(prediction_ent, gold_ent, list_text, log_fn, print_pred)
            scores.update(ent_scores)
            scores_overlap.update(ent_scores_overlap)
        if self.model.RE:
            #rel_scores = metric(prediction, gold_, list_text, relation_alphabet, ent_type_alphabet, log_fn, print_pred, remove_overlap=remove_overlap)
            #rel_scores_overlap = metric_overlap(prediction, gold_, list_text, relation_alphabet, ent_type_alphabet, log_fn, print_pred, remove_overlap=remove_overlap)
            rel_scores = metric(prediction, gold_, prediction_ent, list_text, relation_alphabet, ent_type_alphabet, log_fn, print_pred, remove_overlap=remove_overlap)
            rel_scores_overlap = metric_overlap(prediction, gold_, prediction_ent, list_text, relation_alphabet, ent_type_alphabet, log_fn, print_pred, remove_overlap=remove_overlap)
            scores.update(rel_scores)
            scores_overlap.update(rel_scores_overlap)
        return scores, scores_overlap

    def biorelex_eval_model(self, eval_loader, ent_type_alphabet, relation_alphabet, log_fn=None, print_pred=False):
        self.model.eval()

        truth_sentences, pred_sentences, keys = {}, {}, set()
        with torch.no_grad():
            batch_size = self.args.eval_batch_size
            eval_num = len(eval_loader)
            total_batch = eval_num // batch_size + 1
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > eval_num:
                    end = eval_num
                eval_instance = eval_loader[start:end]
                if not eval_instance:
                    continue
                input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, _, _, info= self.model.batchify(eval_instance, is_test=True)

                ground_truth = dict(zip(info['sent_idx'], info['inst']))
                truth_sentences.update(ground_truth)

                prediction = self.model.biorelex_predict(input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, 
                                                        info, ent_type_alphabet, relation_alphabet)
                pred_sentences.update(prediction)

                keys = keys | set(info['sent_idx'])

        if print_pred:
            predictions = list(pred_sentences.values())
            with open(log_fn, 'w') as f:
                json.dump(predictions, f, indent=True)

        return evaluate_biorelex(truth_sentences, pred_sentences, keys)
    
    def ade_eval_model(self, eval_loader, ent_type_alphabet, relation_alphabet, log_fn=None, print_pred=False):
        self.model.eval()

        truth_sentences, pred_sentences, keys = {}, {}, set()
        with torch.no_grad():
            batch_size = self.args.eval_batch_size
            eval_num = len(eval_loader)
            total_batch = eval_num // batch_size + 1
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > eval_num:
                    end = eval_num
                eval_instance = eval_loader[start:end]
                if not eval_instance:
                    continue
                input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, _, _, info= self.model.batchify(eval_instance, is_test=True)

                ground_truth = dict(zip(info['sent_idx'], info['inst']))
                truth_sentences.update(ground_truth)

                prediction = self.model.ade_predict(input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, 
                                                        info, ent_type_alphabet, relation_alphabet)
                pred_sentences.update(prediction)

                keys = keys | set(info['sent_idx'])

        if print_pred:
            predictions = list(pred_sentences.values())
            with open(log_fn, 'w') as f:
                json.dump(predictions, f, indent=True)

        return evaluate_ade(truth_sentences, pred_sentences, keys)

    def load_state_dict(self, path):
        #self.model.load_state_dict(torch.load(path), strict=False)
        missing_keys, unexpected_keys = self.model.load_state_dict(torch.load(path), strict=False)
        print("Missing keys:", missing_keys)  # 重点关注未被加载的权重
        print("Unexpected keys:", unexpected_keys)

    @staticmethod
    def lr_decay(optimizer, epoch, decay_rate):
        # lr = init_lr * ((1 - decay_rate) ** epoch)
        if epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * (1 - decay_rate)
                # print(param_group['lr'])
        return optimizer
