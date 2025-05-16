import torch
import math
import random
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertForPreTraining, AlbertForPreTraining, DebertaV2PreTrainedModel, DebertaV2ForMaskedLM

from avd import CRF


def cal(l, sigma):
    matrix = torch.zeros(l, l)
    # 填充对角线两侧的元素
    for i in range(l):
        for j in range(l):
            # 对角线上方的元素
            if j > i:
                matrix[i, j] = j - i + 1
            # 对角线下方的元素
            elif j < i:
                matrix[i, j] = i - j + 1

    mu = torch.tensor([0.0])  # 均值
    sigma = torch.tensor([sigma])  # 标准差
    # 计算正态分布的PDF值
    pdf_values = torch.exp(-torch.pow(matrix - mu, 2) / (2 * torch.pow(sigma, 2))) / (
        torch.sqrt(2 * torch.pi * torch.pow(sigma, 2)))
    return pdf_values

class EMMA(nn.Module):  
    def __init__(self, pretrain_model_name_or_path, add_auto_match=True, max_seq_len=128, k=3, num_tags=6):
        super(EMMA, self).__init__()
        self.config = AutoConfig.from_pretrained(pretrain_model_name_or_path)
        self.bert = AutoModel.from_pretrained(pretrain_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name_or_path)
        self.max_seq_len = max_seq_len
        self.k = k
        self.cos = nn.CosineSimilarity(dim=-1)
        self.add_auto_match = add_auto_match

        if add_auto_match:
            self.des_weights1 = nn.Parameter(torch.ones(self.config.hidden_size, 1))
            self.des_bias1 = nn.Parameter(torch.zeros(max_seq_len - 1, 1))
            self.des_weights2 = nn.Parameter(torch.ones(self.config.hidden_size, 1))
            self.des_bias2 = nn.Parameter(torch.zeros(max_seq_len - 1, 1))

        self.des_vectors = None
        self.source_vectors = None

        self.crf = CRF(num_tags, batch_first=True)
    
    # def forward(self, sen_input_ids, sen_att_masks, des_input_ids, des_att_masks, sen_e1_pos, sen_e2_pos, sen_e1_pos_end, sen_e2_pos_end):
    def forward(self, sen_input_ids, sen_att_masks, des_input_ids=None, des_att_masks=None, marked_e1=None, marked_e2=None, label=None, des_features=None, eval=False):
        '''
        sen_input_ids: [bs, max_seq_length]
        sen_att_masks: [bs, max_seq_length]
        des_input_ids: [bs, max_seq_length]
        des_att_masks: [bs, max_seq_length]
        marked_e1: [bs, max_seq_length] 
        marked_e2: [bs, max_seq_length]
        mark_head: [bs, max_seq_length]
        mark_tail: [bs, max_seq_length]
        '''
        batch_size, l = sen_input_ids.size(0), sen_input_ids.size(1)#[b, l, 128]
        device = sen_input_ids.device
        
        # sentence = [cls] + ... + [E1] + E1 + [E1/] + ... + [E2] + E2 + [E2/] + .....
        # label_description = [cls] + .....

        sen_outputs = self.bert(
            input_ids=sen_input_ids.view(-1, self.max_seq_len),
            attention_mask=sen_att_masks.view(-1, self.max_seq_len),
        )
        sen_output = sen_outputs.last_hidden_state#[b*l, 128, 768]->[b*l,2*768]
        # sen_vecx = self.get_sen_vec(sen_output, marked_e1, marked_e2)#[b*l, 128, 768]->[b*l,2*768]
        sen_vec_seq = sen_output[:, 0, :]
        dim = sen_vec_seq.shape[-1]
        sen_vec_seq = sen_vec_seq.reshape(batch_size, -1, dim)#[b, l, dim]

        # train
        # sen_vec = sen_vec_seq

        mask = label + 1  #
        mask = mask.unsqueeze(-1) * mask.unsqueeze(1)
        mask = torch.where(mask > 0, 0, -1e20)
        sigma = 1.0
        p_values = cal(l, sigma).to(device)  # [l,l]

        ##################type3
        sen_vec = sen_vec_seq.unsqueeze(1).expand([batch_size, l, l, dim])
        sen_table = self.cos(sen_vec.permute(0,2,1,3), sen_vec)
        sen_table = sen_table/0.02 + mask
        sen_table = sen_table * p_values + mask
        p = torch.softmax(sen_table, dim=-1)
        sen_vec = torch.sum(sen_vec * (p.unsqueeze(-1)), dim=-2)  # [b,l,dim]


        if eval != True:
            self.gen_des_vectors(des_features)  # 先计算相似度然后flatten
            cos_sim = self.cos(sen_vec.unsqueeze(2),
                               self.des_vectors.unsqueeze(0))  # [b, l, 1, 768]   [1, m, 768] -> [b, l, m]
            pred = torch.softmax(cos_sim / 0.02, dim=-1)  # [b, k, 7]
            mask = label >= 0  # [b, k]
            loss = -self.crf(pred, label, mask=mask, reduction='mean')
            return loss
        else:
            cos_sim = self.cos(sen_vec.unsqueeze(2),
                               self.source_vectors.unsqueeze(0))  # [b, l, 1, 768]   [1, m, 768] -> [b, l, m]
            pred = torch.softmax(cos_sim / 0.02, dim=-1)  # [b, k, 7]
            mask = label >= 0  # [b, k]

            bestpath, crf_logits = self.crf.decode(pred, mask=mask)#
            crf_logits = torch.softmax(crf_logits, dim=-1).to(device)#[bl, 7]
            # print(crf_logits)

            # sim = self.cos(self.source_vectors.unsqueeze(1), self.des_vectors.unsqueeze(0))  # [m,1,768]   [1, n, 768] -> [m, n]
            # sim = torch.softmax(sim/0.02, dim=1)
            # cos_sim = torch.einsum('ij,jk->ik', crf_logits, sim)
            new_sen_vec = torch.einsum('ij,jk->ik', crf_logits, self.source_vectors)  # [bs, m] [m, 768] -> [bs, 768]
            sen_vec = sen_vec[mask].reshape(-1, dim)# [b*l, dim], [b*l]
            new_sen_vec = 0.5*sen_vec + 0.5*new_sen_vec
            cos_sim = self.cos(new_sen_vec.unsqueeze(1), self.des_vectors.unsqueeze(0))#[bs, 1, 768], [1, m, 768]->[bs, m]
            max_sim, max_sim_idx = torch.max(cos_sim, dim=1)

            return max_sim_idx



    def gen_des_vectors(self, des_features):
        # print(des_features.shape)
        max_len = des_features.shape[1]
        des_input_ids = des_features[:, 0: int(max_len / 2)]
        self.des_input_ids_for_predict = des_input_ids
        des_attention_masks = des_features[:, int(max_len / 2): max_len]

        des_outputs = self.bert(
                input_ids=des_input_ids,
                attention_mask=des_attention_masks,
            )
        des_output = des_outputs.last_hidden_state
        self.des_vectors = self.get_des_vec(des_output)

    def get_sen_vec(self, sen_output, marked_e1, marked_e2):
        if self.add_auto_match:
            # e1_h = extract_entity(sen_output, marked_e1) # [E1] [bs, hs]
            # e2_h = extract_entity(sen_output, marked_e2) # [E2]
            e1_h = torch.max(sen_output[:, 1:, :], dim=-2)
            sen_cls = sen_output[:, 0, :] # [bs, hs]
            sen_vec = torch.cat([sen_cls, e1_h], dim=1) # [b, l, 2*1024]
        else:
            sen_vec = sen_output[:, 0, :] # [cls]
        return sen_vec

    def get_des_vec(self, des_output):
        if self.add_auto_match:
            # [bs, ml, hs] x [hs, 1]
            des_cls = des_output[:, 0, :]
            des_output = des_output[:, 1:, :]
            bert_layer1 = torch.squeeze(torch.add(torch.matmul(des_output, self.des_weights1), self.des_bias1), dim=-1) #[bs, ml-1]
            bert_layer_softmax1 = torch.softmax(bert_layer1, dim=-1) # [bs, ml-1]
            e1_h = torch.sum(torch.unsqueeze(bert_layer_softmax1, dim=2) * des_output, dim=1) # [bs, ml-1, 1] * [bs, ml-1, hs]
            des_vec = torch.cat([des_cls, e1_h], dim=1) # [bs, 2, 1024]
        else:
            des_vec = des_output[:, 0, :]
        return des_vec

    def gen_source_des_vectors(self, des_features):
        max_len = des_features.shape[1]
        des_input_ids = des_features[:, 0: int(max_len / 2)]
        self.des_input_ids_for_predict = des_input_ids
        des_attention_masks = des_features[:, int(max_len / 2): max_len]

        des_outputs = self.bert(
                input_ids=des_input_ids,
                attention_mask=des_attention_masks,
            )
        des_output = des_outputs.last_hidden_state
        self.source_vectors = self.get_des_vec(des_output)

    def set_tran(self):
        trans = self.cos(self.des_vectors.unsqueeze(1), self.source_vectors.unsqueeze(0))  # [n, 1, 768]   [1, m, 768] -> [n, m]
        trans = torch.softmax(trans / 0.02, dim=-1)  # [n, m]
        self.crf.new_transitions(trans)