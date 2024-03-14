from tqdm.notebook import tqdm
from sklearn.utils import shuffle
from transformers import AutoTokenizer

from src.neural_punctuator.utils.data import get_config_from_yaml
from src.neural_punctuator.models.BertPunctuator import BertPunctuator

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.neural_punctuator.data.dataloader import BertDataset, collate, get_data_loaders, get_datasets
from src.neural_punctuator.models.BertPunctuator import BertPunctuator
from torch.optim import AdamW
from torch import nn

from src.neural_punctuator.utils.io import save, load
from src.neural_punctuator.utils.metrics import get_total_grad_norm, get_eval_metrics
import numpy as np
import pickle
from bot_helpper.load_metrics import *

from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")


model_type = 'DeepPavlov/rubert-base-cased'  # RuBERT model for Russian

def preprocessing_message(msg):
    msg = msg.split()
    msg = " ".join(msg)
    return [msg]

def create_target(text, tokenizer, target_token2id, id2target):
    encoded_words, targets = [], []
    puncs = ['.', ',', '?']
    words = text.split(' ')
    if words[0] in puncs:
        words = words[1:]

    for word in words:
        target = 0
        for target_token, target_id in target_token2id.items():
            if word.endswith(target_token):
                word = word.rstrip(target_token)
                target = id2target[target_id]

        encoded_word = tokenizer.encode(word, add_special_tokens=False)

        for w in encoded_word:
            encoded_words.append(w)
        for _ in range(len(encoded_word) - 1):
            targets.append(-1)
        targets.append(target)

        #         print([tokenizer._convert_id_to_token(ew) for ew in encoded_word], target)
        if not len(encoded_word) > 0:
            print(text)
            print(words)
            print('error at index ', words.index(word))

        assert (len(encoded_word) > 0)

    encoded_words = [tokenizer.cls_token_id or tokenizer.bos_token_id] + \
                    encoded_words + \
                    [tokenizer.sep_token_id or tokenizer.eos_token_id]
    targets = [-1] + targets + [-1]

    return encoded_words, targets

def load_tokenization_model():
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    target_ids = tokenizer.encode(".?,")[1:-1]
    target_token2id = {t: tokenizer.encode(t)[-2] for t in ".?,"}
    target_ids = list(target_token2id.values())
    id2target = {
        0: 0,
        -1: -1,
    }
    for i, ti in enumerate(target_ids):
        id2target[ti] = i + 1
    target2id = {value: key for key, value in id2target.items()}
    return tokenizer, target_token2id, id2target

def load_punctuator_model(metrics):
    model_name = 'rubert-base-cased'
    model_type = 'by_f_score'
    # metrics = metrics_loader()
    if model_type == 'by_f_score':
        epoch, _ = best_epoch_by_f_score(metrics[model_name])
    elif model_type == 'by_loss':
        epoch, _ = best_epoch_by_loss(metrics[model_name])
    else:
        raise ValueError("Model type not valid, options: by_f_score/by_loss")

    config = get_config_from_yaml(f'./src/neural_punctuator/configs/config-{model_name}-unfreeze.yaml')
    config.trainer.load_model = f"{model_name}-epoch-{epoch + 1}.pth"

    device = torch.device('cpu')  # cuda:0
    # torch.cuda.set_device(device)

    model = BertPunctuator(config)
    model.to(device)

    load(model, None, config)

    config.model.predict_step = 1
    return config, model, device
def tokenization(msg, tokenizer, target_token2id, id2target, config, model, device):
    global model_type
    msg = preprocessing_message(msg)

    encoded_text, target = [], []

    x = list(zip(*(create_target(ts, tokenizer, target_token2id, id2target) for ts in tqdm(msg))))  # change data input here
    encoded_text.append(x[0])
    target.append(x[1])
    with open(data_path + f'{model_type}/message_data.pkl', 'wb') as f:
        pickle.dump((encoded_text, target), f)

    e = []
    i = 0

    raw_words = msg[0].split(' ')

    for te, ta in zip(encoded_text[0][0], target[0][0]):
        if ta == -1:
            e.append(te)
        else:
            e.append(te)
            # print(f"{tokenizer.decode(e):15}\t{tokenizer.decode(target2id[ta]):10}\t{raw_words[i]}")
            e = []
            i += 1



    class BertDataset(Dataset):
        def __init__(self, prefix, config, is_train=False):

            self.config = config
            self.is_train = is_train
            with open(config.data.data_path + prefix + "_data.pkl", 'rb') as f:
                texts, targets = pickle.load(f)
                # texts = texts[:10]
                # targets = targets[:10]
                # print(texts)
                # print(targets)
                self.encoded_texts = [word for t in texts for word in t]
                self.targets = [t for ts in targets for t in ts]

        def __getitem__(self, idx):
            if self.is_train:
                shift = np.random.randint(self.config.trainer.seq_shift) - self.config.trainer.seq_shift // 2
            else:
                shift = 0

            start_idx = idx * config.model.predict_step + shift
            start_idx = max(0, start_idx)
            end_idx = start_idx + self.config.model.seq_len
            return torch.LongTensor(self.encoded_texts[start_idx: end_idx]), \
                torch.LongTensor(self.targets[start_idx: end_idx])

        def __len__(self):
            res = (
                              len(self.encoded_texts) - self.config.model.seq_len) // config.model.predict_step + 1  # self.config.model.seq_len = 4
            if len(self.encoded_texts) < self.config.model.seq_len:
                return 1
            else:
                return res

    valid_dataset = BertDataset("message", config)

    config.predict.batch_size = 1
    valid_loader = DataLoader(valid_dataset, batch_size=config.predict.batch_size, collate_fn=collate)

    model.eval()
    all_valid_preds = []

    for data in tqdm(valid_loader):
        text, targets = data
        with torch.no_grad():
            preds, _ = model(text[0].to(device))

        all_valid_preds.append(preds.detach().cpu().numpy())

    all_valid_target = valid_dataset.targets
    all_valid_preds = np.concatenate(all_valid_preds)
    pred_num = 1
    all_targets = []
    all_preds = []

    for i in tqdm(range(0, all_valid_preds.shape[0] // pred_num)):
        targets = all_valid_target[i: (i + 1)]

        preds = all_valid_preds[i * pred_num]

        all_targets.append(targets)
        all_preds.append(preds)

    targets = np.concatenate(all_targets)
    preds = np.concatenate(all_preds)
    pred = [preds.argmax(-1)]

    targets.shape, preds.shape

    words = []
    indices = []
    for te, ta in zip(encoded_text[0][0], pred[0]):
        words.append(tokenizer._convert_id_to_token(te))
        indices.append(ta)

    # Initialize lists to store modified words
    punctuated_words = []
    combined_words = []

    current_index = -1

    for word, index in zip(words, indices):
        current_index += 1
        if index == 3:
            # Append ',' to the current word
            words[current_index] = word + ","
        if index == 1:
            # Append '.' to the current word
            words[current_index] = word + "."
        if index == 2:
            words[current_index] = word + "?"

    final_mess = " ".join(words[1:-1])
    final_mess = final_mess.replace(" ##", "")

    # Split the text into sentences
    sentences = final_mess.split('. ')

    # Capitalize the first letter of each sentence
    capitalized_sentences = [sentence.strip().capitalize() for sentence in sentences]

    # Join the sentences back together
    final_mess = '. '.join(capitalized_sentences)

    return final_mess
