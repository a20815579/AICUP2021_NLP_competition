# -*- coding: utf-8 -*-

import json
import unicodedata
import re
import tqdm
import math
import re
import os
import gc
import random
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from google.colab import drive

import transformers
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers import BertTokenizer
from transformers import BertForMultipleChoice
from transformers import LongformerTokenizer
from transformers import LongformerForMultipleChoice
from transformers.modeling_outputs import MultipleChoiceModelOutput

from datasets import load_metric

from IPython.display import clear_output

drive.mount('/content/drive')

# helper
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

PRETRAINED_MODEL_NAME = "bert-base-chinese"
#PRETRAINED_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
#PRETRAINED_MODEL_NAME = 'voidful/albert_chinese_base'

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

clear_output()
print("PyTorch 版本：", torch.__version__)

with open("./drive/MyDrive/Aicup/Train_qa_ans.json", "r", encoding = "utf-8") as f_QA:
    data = json.load(f_QA)

train_df = pd.json_normalize(data)

with open("./drive/MyDrive/Aicup/Develop_QA.json", "r", encoding = "utf-8") as f_QA:
    data = json.load(f_QA)

test_df = pd.json_normalize(data)

df = pd.concat([train_df,test_df], axis=0)
train_df_length = len(train_df)


print(f"max article length: {max([len(i) for i in test_df['text']])}")
print(f"max question length: {max([len(i) for i in test_df['question.stem']])}")
print()
print(train_df.answer.value_counts())

# Fix answer error (remove space and error label)
def fix_answer(x):
    if type(x.answer) is float:
        return x
    n = x.answer.replace(" ", "")
    n = unicodedata.normalize("NFKC", n)
    if n not in ['A', 'B', 'C']:
        for c in x["question.choices"]:
            if c["text"] == n:
                n = c["label"]
                break
        else:
            global w
            w = x.answer
            print(x.answer)
            print(x.answer in ['A', 'B', 'C'])
    x.answer = n
    return x
# lower and unify
def fix_unicode(x):
    x.text = unicodedata.normalize("NFKC", x.text.lower())
    x.question = unicodedata.normalize("NFKC", x.question.lower())
    x.choiceA = unicodedata.normalize("NFKC", x.choiceA.lower())
    x.choiceB = unicodedata.normalize("NFKC", x.choiceB.lower())
    x.choiceC = unicodedata.normalize("NFKC", x.choiceC.lower())
    return x

df = df.apply(fix_answer, axis=1)
rows = {"choiceA":[], "choiceB":[], "choiceC":[]}
for index, row in df.iterrows():
    rows['choiceA'].append(row['question.choices'][0]['text'])
    rows['choiceB'].append(row['question.choices'][1]['text'])
    rows['choiceC'].append(row['question.choices'][2]['text'])
df["choiceA"] = rows["choiceA"]
df["choiceB"] = rows["choiceB"]
df["choiceC"] = rows["choiceC"]
df = df.drop("question.choices", axis=1)
df = df.rename(columns={'question.stem': 'question'})
df = df.apply(fix_unicode, axis=1)

"""## 資料處理"""

def get_talker_name():
    talker = set()

    for idx, row in df.iterrows():
        talker = talker.union(set([row.text[m.start()-2:m.start()] for m in re.finditer(':', row.text)]))

    return talker

print(get_talker_name())

def replace_name(row):
    row.text = row.text.replace('醫師a', '醫師')
    row.text = row.text.replace('醫師b', '醫師')
    row.text = row.text.replace('家屬a', '家屬')
    row.text = row.text.replace('家屬b', '家屬')
    row.text = row.text.replace('家屬1', '家屬')
    row.text = row.text.replace('家屬2', '家屬')
    row.text = row.text.replace('家屬、b', '家屬')
    row.text = row.text.replace('民眾a', '民眾')
    row.text = row.text.replace('民眾：我是82。生', '民眾')
    row.text = row.text.replace('護理師a', '護理師')
    row.text = row.text.replace('護理師b', '護理師')
    row.text = row.text.replace('護理師c', '護理師')
    row.text = row.text.replace('個管師', '醫師')
    row.text = row.text.replace('不確定人物', '醫師')
    row.text = re.sub("[ab]先生","先生", row.text)
    row.text = re.sub('[阿啊痾啦呀啦嘛哇吧ㄟ欸诶誒耶餒咧嘞勒了嗎呢唔]', '', row.text)
    row.text = re.sub('[哼亨蛤蝦呃呵哈嘿喔哦唷呦喲齁啦摁恩嗯哎唉]', '', row.text)
    row.text = row.text.replace('挂', '掛')
    row.text = row.text.replace('如果假設', '如果')
    row.text = row.text.replace('艾滋病', '愛滋病')
    row.text = row.text.replace('\u200b', '')
    row.text = re.sub('好+', '好', row.text)
    row.text = re.sub('我+', '我', row.text)
    row.text = re.sub('對+', '對', row.text)    
    row.text = re.sub('會+', '會', row.text)
    row.text = re.sub('謝+', '謝', row.text)
    row.text = re.sub('又+', '又', row.text)
    row.text = re.sub('都+', '都', row.text)
    row.text = re.sub('不+', '不', row.text)
    row.text = re.sub('是+', '是', row.text)
    #row.text = re.sub("醫師[AB]?：","", row.text) 
    #row.text = re.sub("家屬[B12]?：","", row.text) 
    #row.text = re.sub("民眾A?：","", row.text)
    #row.text = re.sub("護理師[AB]?：","", row.text)
    #row.text = row.text.replace('個管師：', '')
    
    row.text = row.text.replace('…', '')
    row.text = row.text.replace('⋯', '') 
    row.text = row.text.replace('·', '')
    row.text = re.sub('[阿啊痾啦呀啦嘛吧]', '', row.text)
    row.text = re.sub('[ㄟ欸诶誒餒]', '', row.text)
    row.text = re.sub('[哼亨]', '', row.text)
    row.text = re.sub("[蛤蝦]？?", "", row.text)
    row.text = re.sub('[喔哦唷呦]', '', row.text)
    row.text = re.sub('[呃呵哈嘿]', '', row.text)
    row.text = re.sub('[(OK)(ok)好]+', '好', row.text)
    row.text = re.sub('對+', '對', row.text)
    row.text = re.sub('[恩嗯]+', '嗯', row.text)
    row.text = re.sub('會+', '會', row.text)
    row.text = re.sub('謝謝+', '謝', row.text)
    row.text = row.text.replace('～', '')
    #row.text = re.sub('[(然後)(所以)(而且)]', '', row.text)
    #row.text = re.sub('[(如果)(還是)(因為)]', '', row.text) # check
    row.text = re.sub('[嗎的嗯]', '', row.text) # check
    row.text = row.text.replace('如果假設', '如果')
    #row.text = row.text.replace('', '')
    row.text = re.sub("就是?", "", row.text)
    row.text = re.sub("這樣子?[。，]", "", row.text)
    row.text = re.sub("[好嗯][。，？][好嗯][。，？]", "好，", row.text)
    row.text = row.text.replace('，。', '。')
    row.text = row.text.replace('。，', '。')
    row.text = row.text.replace('，，', '，')
    row.text = row.text.replace('。。', '。')
    row.text = re.sub('？[。，]', '？', row.text)
    row.text = re.sub('[。，]？', '？', row.text)
    row.text = re.sub("[。，？]", '', row.text) # check
    row.text = re.sub("帶套", '戴套', row.text)
    #row.text = re.sub("一", '1', row.text)
    #row.text = re.sub("二", '2', row.text)
    #row.text = re.sub("三", '3', row.text)
    #row.text = re.sub("四", '4', row.text)
    #row.text = re.sub("五", '5', row.text)
    #row.text = re.sub("六", '6', row.text)
    #row.text = re.sub("七", '7', row.text)
    #row.text = re.sub("八", '8', row.text)
    #row.text = re.sub("九", '9', row.text)
    #row.text = re.sub("十點", '10點', row.text)
    #row.text = re.sub("十", '', row.text)

    row.text = re.sub("砲", '炮', row.text)
    row.question = re.sub("砲", '炮', row.question)
    row.choiceA = re.sub("砲", '炮', row.choiceA)
    row.choiceB = re.sub("砲", '炮', row.choiceB)
    row.choiceC = re.sub("砲", '炮', row.choiceC)
    row.question = re.sub("有誤", '錯誤', row.question)
    row.question = re.sub("何這錯誤", '錯誤', row.question)
    row.question = re.sub("何者為真", '何者正確', row.question)
    row.question = re.sub("何項", '何者', row.question)
    row.question = re.sub("請問", '', row.question)
    row.question = re.sub("q1.", '', row.question)
    return row

df = df.apply(replace_name, axis=1)

print(get_talker_name())

# run these to avoid OOM
def clear_mm():
    gc.collect()
    torch.cuda.empty_cache()

import jieba
import synonyms
import random
from random import shuffle

random.seed(2019)

import jieba
import jieba.analyse

jieba.set_dictionary('dict.txt.big')
jieba.analyse.set_stop_words('stopwords.txt')

jieba.add_word('個管師')
jieba.add_word('這個月')
jieba.add_word('性行為')
jieba.add_word('無套')
jieba.add_word('保險套')
jieba.add_word('固炮')
jieba.add_word('約炮')
jieba.add_word('性生活')
jieba.add_word('小時')
jieba.add_word('內射')
jieba.add_word('69')
jieba.add_word('10') # ###
jieba.add_word('甲狀腺')
jieba.add_word('發燒')
jieba.add_word('噴嚏')
jieba.add_word('回診')
jieba.add_word('口交')
jieba.add_word('梅毒')
jieba.add_word('抗體')
jieba.add_word('洗腎')
jieba.add_word('hpv')
jieba.add_word('prep')
jieba.add_word('hiv')
jieba.add_word('免疫')
jieba.add_word('菜花')
jieba.add_word('藥')
jieba.add_word('伴侶')
jieba.add_word('陽性')
#jieba.add_word('固定性伴侶')
jieba.add_word('任務型')
jieba.add_word('感染')
jieba.add_word('健保卡')
jieba.add_word('檢驗')
jieba.add_word('抗體')
jieba.add_word('患者')
jieba.add_word('肛交')
jieba.add_word('愛滋')
jieba.add_word('開放式關係')
jieba.add_word('帶原')
jieba.add_word('小三')
jieba.add_word('口爆')
jieba.add_word('白血球')
jieba.add_word('紅血球')
jieba.add_word('血小板')
jieba.add_word('紅血球')
jieba.add_word('關於')
jieba.add_word('建議')
jieba.add_word('藥')
jieba.add_word('傳染病')
jieba.add_word('對於')
jieba.add_word('原廠藥')
jieba.add_word('學名藥')
jieba.add_word('民眾')
jieba.add_word('b型肝炎')
jieba.add_word('匿篩')
jieba.add_word('篩檢')
jieba.add_word('性關係')
jieba.add_word('循環')
jieba.add_word('追蹤')
jieba.add_word('x光')
jieba.add_word('鼻竇炎')
jieba.add_word('cd4')
jieba.add_word('c肝')
jieba.add_word('胃痙攣')
jieba.add_word('特效藥')
jieba.add_word('傳統')
jieba.add_word('腎指數')
jieba.add_word('何種')
jieba.add_word('牙齒')
jieba.add_word('刮鬍刀')
jieba.add_word('b肝')
jieba.add_word('a肝')
jieba.add_word('c肝')
jieba.add_word('c型肝炎')
jieba.add_word('5點')
jieba.add_word('安全性行為')
jieba.add_word('事後')
jieba.add_word('大於')
jieba.add_word('膽固醇')
jieba.add_word('9點')
jieba.add_word('胃食道')
jieba.add_word('澱粉')
jieba.add_word('小便')
jieba.add_word('復健')
jieba.add_word('前任')
jieba.add_word('止痛藥')
jieba.add_word('唸書')
jieba.add_word('打砲')
jieba.add_word('滷肉飯')
jieba.add_word('複合性')
jieba.add_word('結核菌')
jieba.add_word('b群')
jieba.add_word('不戴')
jieba.add_word('噁心感')
jieba.add_word('6點')
jieba.add_word('有關')
jieba.add_word('肝')
jieba.add_word('腎')
jieba.add_word('普心寧')
jieba.add_word('十三價')
jieba.add_word('頭')
jieba.add_word('會痛')
jieba.del_word('的藥')
jieba.del_word('性行')
jieba.del_word('訂會')
jieba.del_word('藥就')
jieba.add_word('u=u') # wtf

def split_sent(sentence: str):
    first_role_idx = re.search(':', sentence).end(0)
    #out = [sentence[:first_role_idx]]
    out = ['']
    tmp = sentence[first_role_idx:]
    while tmp:
        res = re.search(
            r'(護理師[\w*]\s*:|醫師\s*:|民眾\s*:|家屬[\w*]\s*:|個管師\s*:)', tmp)
        if res is None:
            break
        #字本身不能超過150＝＝
        idx = res.start(0)
        idx_end = res.end(0)
        f = out[-1] + tmp[:idx]
        if len(f) > 150:
            r = split_sentence(f)
            out[-1] = r[0]
            out += r[1:]
        else:
            out[-1] = f
        out.append('')
        #print(tmp[idx:idx_end])
        #out.append(tmp[idx:idx_end])
        tmp = tmp[idx_end:]

    out[-1] = out[-1] + tmp

    return out

global_tmp = []
def split_df(x):
    sentences = split_sent(x.text)
    tmp = []
    try:
        a = jieba.analyse.extract_tags(x.choiceA, topK=10)
        b = jieba.analyse.extract_tags(x.choiceB, topK=10)
        c = jieba.analyse.extract_tags(x.choiceC, topK=10)
        if len(x.choiceA) <= 2:
            tmp.append(x.choiceA)
        elif len(a) == 0:
            tmp.append(x.choiceA) # check
        else:
            tmp.append(a[0])
        if len(x.choiceB) <= 2:
            tmp.append(x.choiceB) # check
        elif len(b) == 0:
            tmp.append(x.choiceB)
        else:
            tmp.append(b[0])
        if len(x.choiceC) <= 2:
            tmp.append(x.choiceC)
        elif len(c) == 0:
            tmp.append(x.choiceC) # check
        else:
            tmp.append(c[0])
    except:
        print(x['id'])
    global_tmp.append(tmp)
    
    #if len(tmp) == 0:
    #    assert f'{x.idx} no tokens'
    text = []
    addition = False
    for sentence in sentences:
        #sentence = re.sub(r'[:]', '說', sentence)
        sentence = re.sub(r'[^\w\s:]', '', sentence)
        if len(sentence) == 0:
            continue
        if addition:
            text.append(sentence)
            addition = False
            continue
        for match in tmp:
            if match in sentence:
                text.append(sentence)
                addition = True
                break
    if len(text) == 0:
        with open('stopwords.txt') as f:
            stopwords = f.read().splitlines()
        text += [i for i in jieba.lcut(x.choiceA) if i not in stopwords]
        text += [i for i in jieba.lcut(x.choiceA) if i not in stopwords]
        text += [i for i in jieba.lcut(x.choiceA) if i not in stopwords]
    x['new_text'] = ''.join(text)
    return x

df_tmp = df.apply(split_df, axis=1)
df['text'] = df_tmp['new_text']

t_df = df[:len(train_df)]
d_df = df[len(train_df):]

positive = []
negative = []
n_list = ['錯誤', '非', '沒有', '不是', '不正確', '不包含', '不用', '不需要']

for j,i in t_df.iterrows():
    for n in n_list:
        if n in i.question:
            negative.append(j)
            break
    else:
        positive.append(j)
tp_df = t_df.iloc[positive]
tn_df = t_df.iloc[negative]

positive = []
negative = []

for j,i in d_df.iterrows():
    for n in n_list:
        if n in i.question:
            negative.append(j)
            break
    else:
        positive.append(j)

dp_df = d_df.iloc[positive]
dn_df = d_df.iloc[negative]

class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def convert_examples_to_features(df, tokenizer, max_seq_length=512, is_training=False):
    features = []
    label_map = {'A': 0, 'B': 1, 'C': 2}
    for idx, (_, example) in enumerate(df.iterrows()):
        context_tokens = tokenizer.tokenize(example.text)
        start_ending_tokens = tokenizer.tokenize(example.question)
        
        choices_features = []
        choices = [df.iloc[idx]["choiceA"], df.iloc[idx]["choiceB"], df.iloc[idx]["choiceC"]]
        for ending_index, ending in enumerate(choices):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]
            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            while True:
                total_length = len(context_tokens_choice) + len(ending_tokens)
                if total_length <= max_seq_length - 3:
                    break
                if len(context_tokens_choice) > len(ending_tokens):
                    context_tokens_choice.pop()
                else:
                    ending_tokens.pop()

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = None
        if is_training:
            label = df.iloc[idx]["answer"]
            label = label.replace(" ", "")
            label = unicodedata.normalize("NFKC", label)
            label = label_map[label]

        features.append(
            InputFeatures(
                example_id = example.id,
                choices_features = choices_features,
                label = label
            )
        )
    return features

tp_features = convert_examples_to_features(tp_df, tokenizer, 512, True)
tn_features = convert_examples_to_features(tn_df, tokenizer, 512, True)
dp_features = convert_examples_to_features(dp_df, tokenizer, 512, False)
dn_features = convert_examples_to_features(dn_df, tokenizer, 512, False)

class QADataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, input_mask, segment_ids, labels = None):
        # Dataset class 的 parameters 放入我們 tokenization 後的資料以及資料的標籤
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.labels = labels
        self.len = len(input_ids)

    def __getitem__(self, idx):
        # 請注意 tokenization 後的資料是一個 dict
        # 在此步驟將資料以及標籤都轉換為 PyTorch 的 tensors
        item = {"input_ids":self.input_ids[idx], "token_type_ids":self.segment_ids[idx], \
                "attention_mask":self.input_mask[idx]}
        if self.labels != None:
          item['labels'] = self.labels[idx]
        return item
        
    def __len__(self):
        # 回傳資料集的總數
        return self.len

all_input_ids = torch.tensor(select_field(tp_features, 'input_ids'), dtype=torch.long)
all_input_mask = torch.tensor(select_field(tp_features, 'input_mask'), dtype=torch.long)
all_segment_ids = torch.tensor(select_field(tp_features, 'segment_ids'), dtype=torch.long)
all_label = torch.tensor([f.label for f in tp_features], dtype=torch.long)

dataset = QADataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
train_p_dataset, val_p_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

all_input_ids = torch.tensor(select_field(tn_features, 'input_ids'), dtype=torch.long)
all_input_mask = torch.tensor(select_field(tn_features, 'input_mask'), dtype=torch.long)
all_segment_ids = torch.tensor(select_field(tn_features, 'segment_ids'), dtype=torch.long)
all_label = torch.tensor([f.label for f in tn_features], dtype=torch.long)

dataset = QADataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
train_n_dataset, val_n_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_p = BertForMultipleChoice.from_pretrained(pretrained_model_name_or_path=PRETRAINED_MODEL_NAME)
model_p = model_p.to(device)
model_n = BertForMultipleChoice.from_pretrained(pretrained_model_name_or_path=PRETRAINED_MODEL_NAME)
model_n = model_n.to(device)

metric = load_metric("accuracy")
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args_p = transformers.TrainingArguments(
    output_dir='./drive/MyDrive/bert_new_keywords_p/result',             # 輸出的資料夾
    num_train_epochs=5,               # 總共訓練的 epoch 數目
    per_device_train_batch_size=4,          # 訓練模型時每個裝置的 batch size
    per_device_eval_batch_size=4,          # 驗證模型時每個裝置的 batch size
    #warmup_steps=500,                # learning rate scheduler 的參數
    #weight_decay=0.01,                # 最佳化演算法 (optimizer) 中的權重衰退率
    logging_dir='./drive/MyDrive/bert_new_keywords_p/logs/',              # 存放 log 的資料夾
    logging_steps=15,
    seed=random_seed,
    learning_rate=1e-5,
    gradient_accumulation_steps = 3,
    eval_accumulation_steps = 3,
    evaluation_strategy = "steps",
)

training_args_n = transformers.TrainingArguments(
    output_dir='./drive/MyDrive/bert_new_keywords_n/result',             # 輸出的資料夾
    num_train_epochs=5,               # 總共訓練的 epoch 數目
    per_device_train_batch_size=4,          # 訓練模型時每個裝置的 batch size
    per_device_eval_batch_size=4,          # 驗證模型時每個裝置的 batch size
    #warmup_steps=500,                # learning rate scheduler 的參數
    #weight_decay=0.01,                # 最佳化演算法 (optimizer) 中的權重衰退率
    logging_dir='./drive/MyDrive/bert_new_keywords_n/logs/',              # 存放 log 的資料夾
    logging_steps=15,
    seed=random_seed,
    learning_rate=1e-5,
    gradient_accumulation_steps = 3,
    eval_accumulation_steps = 3,
    evaluation_strategy = "steps",
)

trainer_p = transformers.Trainer(
    model=model_p,                         # 🤗 的模型
    args=training_args_p,                  # Trainer 所需要的引數
    train_dataset=train_p_dataset,      
       # 訓練集 (注意是 PyTorch Dataset)
    eval_dataset=val_p_dataset,            # 驗證集 (注意是 PyTorch Dataset)，可使 Trainer 在進行訓練時也進行驗證
    compute_metrics=compute_metrics      # 自定的評估的指標
)

# 指定使用 1 個 GPU 進行訓練
# trainer.args._n_gpu=1

# 開始進行模型訓練
clear_mm()
trainer_p.train()

trainer_p.save_model('./drive/MyDrive/bert_new_keywords_p/model/')

trainer_n = transformers.Trainer(
    model=model_n,                         # 🤗 的模型
    args=training_args_n,                  # Trainer 所需要的引數
    train_dataset=train_n_dataset,      
       # 訓練集 (注意是 PyTorch Dataset)
    eval_dataset=val_n_dataset,            # 驗證集 (注意是 PyTorch Dataset)，可使 Trainer 在進行訓練時也進行驗證
    compute_metrics=compute_metrics      # 自定的評估的指標
)

# 指定使用 1 個 GPU 進行訓練
# trainer.args._n_gpu=1

# 開始進行模型訓練
clear_mm()
trainer_n.train()

trainer_n.save_model('./drive/MyDrive/bert_new_keywords_n/model/')

pred = trainer_n.predict(val_n_dataset)