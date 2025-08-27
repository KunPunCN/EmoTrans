import json
import os
import random
import numpy as np
import time
import torch
import random
import logging
import datetime
import torch.nn as nn
from torch.nn import MSELoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig
from transformers import get_linear_schedule_with_warmup, AdamW
from argparse import ArgumentParser
from tqdm import tqdm
from utils import *
from dataset import *
from model_crf import *
from torch.utils.data import DataLoader, RandomSampler
from tensorboardX import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

def compute_macro_PRF(predicted_idx, gold_idx, i=-1, empty_label=None):
    '''
    This evaluation function follows work from Sorokin and Gurevych(https://www.aclweb.org/anthology/D17-1188.pdf)
    code borrowed from the following link:
    https://github.com/UKPLab/emnlp2017-relation-extraction/blob/master/relation_extraction/evaluation/metrics.py
    '''
    if i == -1:
        i = len(predicted_idx)

    # print(type(predicted_idx))
    # print(predicted_idx)
    # print(type(gold_idx))
    # print(gold_idx)
    complete_rel_set = set(gold_idx) - {empty_label}
    avg_prec = 0.0
    avg_rec = 0.0
    # print(complete_rel_set)
    for r in complete_rel_set:
        # print(i)
        # print(predicted_idx[:i])
        r_indices = (predicted_idx[:i] == r)
        # print(r_indices)
        # print(r_indices.nonzero())
        tp = len((predicted_idx[:i][r_indices] == gold_idx[:i][r_indices]).nonzero()[0])
        tp_fp = len(r_indices.nonzero()[0])
        tp_fn = len((gold_idx == r).nonzero()[0])
        prec = (tp / tp_fp) if tp_fp > 0 else 0
        rec = tp / tp_fn
        avg_prec += prec
        avg_rec += rec
    f1 = 0
    avg_prec = avg_prec / len(set(predicted_idx[:i]))
    avg_rec = avg_rec / len(complete_rel_set)
    if (avg_rec + avg_prec) > 0:
        f1 = 2.0 * avg_prec * avg_rec / (avg_prec + avg_rec)

    return avg_prec, avg_rec, f1

def custom_collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    des_input_ids = [item['des_input_ids'] for item in batch]
    des_attention_mask = [item['des_attention_mask'] for item in batch]
    rid = [item['rid'] for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    des_input_ids = pad_sequence(des_input_ids, batch_first=True, padding_value=0)
    des_attention_mask = pad_sequence(des_attention_mask, batch_first=True, padding_value=0)
    rid = pad_sequence(rid, batch_first=True, padding_value=-1)
    features = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "des_input_ids": des_input_ids,
        "des_attention_mask": des_attention_mask,
        "rid": rid,
    }
    return features


def train(train_dataset, model, args, device):
    writer = SummaryWriter()
    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=custom_collate_fn)
    t_total = len(train_dataloader) * args.epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.1},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warm_up, num_training_steps=t_total)

    global_step = 0
    best_step = 0
    best_f1 = 0
    min_train_loss = float('inf')

    model.zero_grad()

    des_features = train_dataset.get_evaluate_des_features().to(device)
    for epoch in range(1, int(args.epochs) + 1):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch, args.epochs))
        print('Training...')
        total_train_loss = 0
        t0 = time.time()


        for batch in tqdm(train_dataloader, desc="Iteration"):
            model.train()
            inputs = {'sen_input_ids':      batch["input_ids"].to(device),
                    'sen_att_masks':   batch["attention_mask"].to(device),
                    'des_input_ids':      batch["des_input_ids"].to(device),
                    'des_att_masks':   batch["des_attention_mask"].to(device),
                    'label': batch["rid"].to(device),
                    'des_features': des_features,
                    }

            outputs = model(**inputs)
            loss = outputs

            total_train_loss += loss.item()
            loss.backward() 
            optimizer.step() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            scheduler.step() 
            model.zero_grad()
            
            avg_train_loss = loss.item() / args.train_batch_size
            global_step += 1
            # print("Epoch: {}  global_step: {} Average training loss: {:.7f}".format(epoch, global_step, avg_train_loss))
            writer.add_scalar('avg_train_loss', avg_train_loss, global_step=global_step)
            
            if avg_train_loss < min_train_loss:
                min_train_loss = avg_train_loss
                best_step = global_step
        # avg_train_loss = total_train_loss / len(train_dataloader)  
        training_time = format_time(time.time() - t0)
        # print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("Training epcoh took: {:}".format(training_time))


        # dev
        train_dataset.mode = "dev"
        dev_dataset = train_dataset
        # p_macro_sim, r_macro_sim, f_macro_sim, p_macro_classify, r_macro_classify, f_macro_classify = evaluate(dev_dataset, model, args, args.device)
        p_macro_sim, r_macro_sim, f_macro_sim = evaluate(dev_dataset, model, args, args.device, args.dataset, args.dataset2)
        dev_info_sim = f'[dev][sim] (macro) final precision: {p_macro_sim:.2f}, recall: {r_macro_sim:.2f}, f1 score: {f_macro_sim:.2f}'
        print(dev_info_sim)
        if f_macro_sim > best_f1:
            best_f1 = f_macro_sim
            print('Saveing Model...!!!!!!!!!!!!!!!!!!!!!!!-------------------')
            torch.save(model, args.checkpoint_dir)


        # test
        train_dataset.mode = "test"
        test_dataset = train_dataset
        # p_macro_sim, r_macro_sim, f_macro_sim, p_macro_classify, r_macro_classify, f_macro_classify = evaluate(test_dataset, model, args, args.device)
        p_macro_sim, r_macro_sim, f_macro_sim = evaluate(test_dataset, model, args, args.device, args.dataset, args.dataset3)
        test_info_sim = f'[test][sim] (macro) final precision: {p_macro_sim:.2f}, recall: {r_macro_sim:.2f}, f1 score: {f_macro_sim:.2f}'
        print(test_info_sim)
        # test_info_classify = f'[test][classify] (macro) final precision: {p_macro_classify:.4f}, recall: {r_macro_classify:.4f}, f1 score: {f_macro_classify:.4f}'
        # print(test_info_classify)
        train_dataset.mode = "train"

    return model, best_step, min_train_loss, t_total

    
def evaluate(dataset, model, args, device, source, task):
    t0 = time.time()
    model.eval()
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.evaluate_batch_size, collate_fn=custom_collate_fn)

    with torch.no_grad():
        predict_labels_sim = []
        # predict_labels_classify = []
        true_labels = []
        # epoch_iterator = tqdm(dataloader, desc="Iteration")

        des_features = dataset.get_evaluate_des_features().to(device)
        model.gen_des_vectors(des_features)
        source_des_features = dataset.get_source_des_features().to(device)
        model.gen_source_des_vectors(source_des_features)
        model.set_tran()#修改tran CRF

        for step, batch in enumerate(dataloader):
            inputs = {'sen_input_ids':      batch["input_ids"].to(device),
                    'sen_att_masks':   batch["attention_mask"].to(device),
                    'label': batch["rid"].to(device),
                    'des_features': des_features,
                    'eval': True,
                    }
            outputs = model(**inputs)
            # max_sim_idx, max_classify_idx = outputs
            max_sim_idx = outputs
            predict_labels_sim.extend(max_sim_idx.tolist())
            # predict_labels_classify.extend(max_classify_idx.tolist())

            true_label = batch["rid"]
            true_label = true_label[true_label >= 0].tolist()
            true_labels.extend(true_label)

        # IEMOCAP: happiness-2, sadness-3, anger-4, frustration-1, excited-5, and neutral-0

        # MELD: neutral-0, joy-2, surprise-6, sadness-3, anger-5, disgust-1, and fear-4
        # DailyDialog: neutral-0, joy-4, surprise-6, sadness-2, anger-1, disgust-5, and fear-3

        # EmoryNLP: sad-3, mad-5, scared-4, powerful-6, peaceful-1, joyful-2, and neutral-0
        if source == 'meld' or source == 'dialogues':#neutral, joy, surprise, sadness, anger, disgust, and fear
            if task == 'iemocap':#[2]happiness  [1, 5]frustration,excited  test
                avg_p, avg_r, avg_fscore, _ = precision_recall_fscore_support(true_labels, predict_labels_sim, average='weighted', labels=[1, 2, 5])
            elif task == 'emory':#[4, 5]scared,mad [1, 6]peaceful,powerful  dev
                avg_p, avg_r, avg_fscore, _ = precision_recall_fscore_support(true_labels, predict_labels_sim, average='weighted', labels=[1, 4, 5, 6])
            elif task == 'dialogues' or task == 'meld':
                print('can\'t')
        elif source == 'iemocap':#happiness, sadness, anger, frustration, excited, and neutral
            if task == 'meld':#[2]joy,  [1, 4, 6]disgust,fear,surprise
                avg_p, avg_r, avg_fscore, _ = precision_recall_fscore_support(true_labels, predict_labels_sim, average='weighted', labels=[1, 2, 4, 6])
            elif task == 'dialogues':#[4]joy,  [5, 3, 6]disgust,fear,surprise
                avg_p, avg_r, avg_fscore, _ = precision_recall_fscore_support(true_labels, predict_labels_sim, average='weighted', labels=[3, 4, 5, 6])
            elif task == 'emory':#[2,5]joy,mad  [1, 4, 6]peaceful,scared,powerful
                avg_p, avg_r, avg_fscore, _ = precision_recall_fscore_support(true_labels, predict_labels_sim, average='weighted', labels=[1, 2, 4, 5, 6])
        elif source == 'emory':#sad, mad, scared, powerful, peaceful, joyful, and neutral
            if task == 'iemocap':#[2,4]happiness,anger  [1, 5]frustration,excited
                avg_p, avg_r, avg_fscore, _ = precision_recall_fscore_support(true_labels, predict_labels_sim, average='weighted', labels=[1, 2, 4, 5])
            elif task == 'dialogues':#[3,1]fear,anger  [5, 6]disgust,surprise
                avg_p, avg_r, avg_fscore, _ = precision_recall_fscore_support(true_labels, predict_labels_sim, average='weighted', labels=[1, 3, 5, 6])
            elif task == 'meld':#[4,5]fear,anger  [1, 6]disgust,surprise
                avg_p, avg_r, avg_fscore, _ = precision_recall_fscore_support(true_labels, predict_labels_sim, average='weighted', labels=[1, 4, 5, 6])
        else:
            print('error')

        avg_p = round(avg_p * 100, 2)
        avg_r = round(avg_r * 100, 2)
        avg_fscore = round(avg_fscore * 100, 2)
        # p_macro_sim, r_macro_sim, f_macro_sim = compute_macro_PRF(np.array(predict_labels_sim), np.array(true_labels))
        # return p_macro_sim, r_macro_sim, f_macro_sim
        return avg_p, avg_r, avg_fscore

if __name__=='__main__':
    parser = ArgumentParser()

    # hyperparameters
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--gamma", type=float, default=0.06, help="Loss function: margin factor gamma")
    # parser.add_argument("--alpha", type=float, default=0.33,
                        # help="Similarity: balance entity and context weights in single sample")
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--evaluate_batch_size", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=5, help='training epochs')
    parser.add_argument("--max_seq_len", type=int, default=128, help='max sequence length')
    parser.add_argument("--lr", type=float, default=2e-6, help='learning rate')
    parser.add_argument("--k", type=int, default=3, help='Number of classification')

    parser.add_argument("--warm_up", type=float, default=100, help='warm_up steps')
    parser.add_argument("--unseen", type=int, default=15, help='Number of unseen class')
    parser.add_argument("--expand_data", action='store_true', help='expand the input data')
    # parser.add_argument("--entity_way", type=str, choices=['tmp', 'keyword'], default='tmp',
    #                     help='Representation of the described entity')
    
    # file_path
    parser.add_argument("--dataset_path", type=str, default='ERC_dataset', help='where data stored')
    parser.add_argument("--dataset", type=str, default='dialogues', choices=['dialogues', 'emory', 'iemocap', 'meld'],
                        help='original dataset')
    parser.add_argument("--dataset2", type=str, default='emory', choices=['dialogues', 'emory', 'iemocap', 'meld'],
                        help='original dataset')
    parser.add_argument("--relation_description_processed", type=str,
                        default='relation_description_processed.json',
                        help='relation descriptions marked entity')

    # model and cuda config
    parser.add_argument("--gpu_available", type=str, default='0', help='the device on which this model will run')
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='bert-base-uncased', help='huggingface pretrained model')
    parser.add_argument("--add_auto_match", type=str, default='True', help='')
    args = parser.parse_args()

    # 'meld' #neutral-0, joy-2, surprise-6, sadness-3, anger-5, disgust-1, and fear-4
    # 'dialogues' neutral-0, joy-4, surprise-6, sadness-2, anger-1, disgust-5, and fear-3
    args.dataset3 = 'dialogues'#'dialogues'#'meld' #neutral-0, joy-2, surprise-6, sadness-3, anger-5, disgust-1, and fear-4
    args.dataset = 'emory'#sad-3, mad-5, scared-4, powerful-6, peaceful-1, joyful-2, and neutral-0
    args.dataset2 = 'iemocap'#happiness-2, sadness-3, anger-4, frustration-1, excited-5, and neutral-0
    # args.data_file = os.path.join(args.dataset_path, f'{args.dataset}_train.csv')
    # args.data_file2 = os.path.join(args.dataset_path, f'{args.dataset2}_train.csv')
    # args.relation_description_file = os.path.join(args.dataset_path, args.dataset, 'relation_description',
    #                                               f'{args.dataset}_relation_description.json')
    # args.relation_description_file_processed = os.path.join(args.dataset_path, args.dataset, 'relation_description',
    #                                             args.relation_description_processed)

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    args.checkpoint_dir = f'checkpoints/{args.dataset}_{args.dataset3}_unseen_{str(args.unseen)}.pth'
    add_auto_match = True if args.add_auto_match == 'True' else False

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_time = time.time()
    # if not os.path.exists(args.checkpoint_dir):
    #     os.makedirs(args.checkpoint_dir)
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning(f'device: {args.gpu_available}, epochs: {args.epochs}, lr: {args.lr}, seed: {args.seed}, batch size: {args.train_batch_size}')

    # set seed
    set_seed(args.seed)
    
    args.device = torch.device("cuda:" + args.gpu_available if torch.cuda.is_available() else "cpu")

    if args.dataset == 'iemocap':
        num_tags = 6
    else:
        num_tags = 7
    model = EMMA(args.pretrained_model_name_or_path, add_auto_match, args.max_seq_len, args.k, num_tags)
    model.to(args.device)

    # train
    train_dataset = Dataset("train", args.dataset_path, args.dataset,
                            args.dataset2, args.dataset3, args.unseen,
                            args.pretrained_model_name_or_path, args.max_seq_len, model, args, expand_or_not=args.expand_data)
    model, best_step, min_train_loss, total_steps= train(train_dataset, model, args, args.device)

    model = torch.load(args.checkpoint_dir)#"checkpoints/dialogues_split_7_unseen_15.pth")
    # dev
    train_dataset.mode = "dev"
    dev_dataset = train_dataset
    # p_macro_sim, r_macro_sim, f_macro_sim, p_macro_classify, r_macro_classify, f_macro_classify = evaluate(dev_dataset, model, args, args.device)
    p_macro_sim, r_macro_sim, f_macro_sim = evaluate(dev_dataset, model, args, args.device, args.dataset, args.dataset2)
    dev_info_sim = f'[dev][sim] (macro) final precision: {p_macro_sim:.2f}, recall: {r_macro_sim:.2f}, f1 score: {f_macro_sim:.2f}'
    print(dev_info_sim)
    # dev_info_classify = f'[dev][classify] (macro) final precision: {p_macro_classify:.4f}, recall: {r_macro_classify:.4f}, f1 score: {f_macro_classify:.4f}'
    # print(dev_info_classify)

    # test
    dev_dataset.mode = "test"
    test_dataset = dev_dataset
    # p_macro_sim, r_macro_sim, f_macro_sim, p_macro_classify, r_macro_classify, f_macro_classify = evaluate(test_dataset, model, args, args.device)
    p_macro_sim, r_macro_sim, f_macro_sim = evaluate(test_dataset, model, args, args.device, args.dataset, args.dataset3)
    test_info_sim = f'[test][sim] (macro) final precision: {p_macro_sim:.2f}, recall: {r_macro_sim:.2f}, f1 score: {f_macro_sim:.2f}'
    print(test_info_sim)
    # test_info_classify = f'[test][classify] (macro) final precision: {p_macro_classify:.4f}, recall: {r_macro_classify:.4f}, f1 score: {f_macro_classify:.4f}'
    # print(test_info_classify)

    # running time
    end_time = time.time()
    run_time = end_time - start_time

    # with open("result.txt", "a") as file:
    #     file.write("Datetime: " + current_datetime + "\n")
    #     file.write("Run time: {:.2f} seconds\n".format(run_time))
    #     file.write(f"Total steps: {total_steps}\n")
    #     file.write(f"Best step: {best_step}\n")
    #     file.write(f"Min train loss: {min_train_loss}\n")
    #     file.write("Parameters info:\n")
    #     for arg in vars(args):
    #         file.write(f"\t {arg}: {getattr(args, arg)}\n")
    #     file.write("\n")
    #     file.write("Evaluation results:\n")
    #     file.write(dev_info_sim + "\n")
    #     # file.write(dev_info_classify + "\n")
    #     file.write(test_info_sim + "\n")
    #     # file.write(test_info_classify + "\n")
    #     file.write("\n")
        