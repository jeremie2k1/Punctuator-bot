import os
from glob import glob
import torch
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings("ignore")

# path = "../src/neural_punctuator/models/"

data_path = "./data_cleaning/"
model_path = "./src/neural_punctuator/models/"
model_names = ["rubert-base-cased"]
plt_names = ["DeepPavlov/rubert-base-cased"]


files = {}
for model_name in model_names:
    f = sorted(glob(model_path + f"{model_name}-epoch*.*"), key=os.path.getmtime)
    files[model_name] = f
def load_scores(model_path):
    checkpoint = torch.load(model_path)
    return checkpoint['metrics']

def metrics_loader():

    metrics = {}
    for model_name in model_names:
        m = []
        for file in tqdm(files[model_name]):
            m.append(load_scores(file))
        metrics[model_name] = m

    def get_strict_f_score(report):
        return sum(float(report['cls_report'][output]['f1-score']) for output in ('period', 'question', 'comma')) / 3

    for _, m in metrics.items():
        for epoch in m:
            epoch['strict_f_score'] = get_strict_f_score(epoch)

    print(metrics)
    print(metrics['rubert-base-cased'][0])

    return metrics

def best_epoch_by_f_score(metrics):
    best_score = metrics[0]['strict_f_score']
    best_epoch = 0
    for i, m in enumerate(metrics):
        if m['strict_f_score'] > best_score:
            best_score = m['strict_f_score']
            best_epoch = i
    return best_epoch, best_score

def best_epoch_by_loss(metrics):
    best_loss = metrics[0]['loss']
    best_epoch = 0
    for i, m in enumerate(metrics):
        if m['loss'] < best_loss:
            best_loss = m['loss']
            best_epoch = i
    return best_epoch, best_loss