#!/usr/bin/env python
# coding: utf-8

# In[1]:

import dill
import time
import random
import numpy as np
from sklearn.metrics import roc_curve, auc

import nltk

nltk.download("punkt")
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn

from torchtext.data import Field  # 필드 정의하기
from torchtext.data import TabularDataset  # 데이터셋 만들기
from torchtext.data import BucketIterator  # 모든 텍스트 작업을 일괄로 처리하고 단어를 인덱스 숫자로 변환 하는것을 도움
from torchtext.data import Iterator  # 토치텍스트의 데이터로더 만들기


# In[2]:


RANDOM_SEED = 2020
torch.manual_seed(RANDOM_SEED) # manual_seed: pytorch에서 random seed를 고정하기 위한 함수
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# DATA_PATH = "data/processed/"
DATA_PATH = "../data/processed/"

# ## 데이터 불러오기
# - `torchtext.Field`를 이용해 각각의 필드를 정의해줍니다.

# In[3]:


TEXT = Field(
    sequential=True,
    use_vocab=True,
    tokenize=word_tokenize,
    lower=True,
    batch_first=True,
)
LABEL = Field(
    sequential=False,
    use_vocab=False,
    batch_first=True,
)


# 수능 데이터를 불러오는 코드입니다. `DATA_PATH`가 데이터가 있는 폴더로 정확히 입력되어 있는지 확인해주세요.

# In[4]:


sat_train_data, sat_valid_data, sat_test_data = TabularDataset.splits(
    path=DATA_PATH,
    train="sat_train.tsv",
    validation="sat_valid.tsv",
    test="sat_test.tsv",
    format="tsv",
    fields=[("text", TEXT), ("label", LABEL)],
    skip_header=1,
)

sat_train_iterator, sat_valid_iterator, sat_test_iterator = BucketIterator.splits(
    (sat_train_data, sat_valid_data, sat_test_data),
    batch_size=8,
    device=None,
    sort=False,
)

TEXT.build_vocab(sat_train_data, min_freq=2)


# ## LSTM Classifier

# In[5]:


class LSTMClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers, pad_idx):
        super().__init__()
        self.embed_layer = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )
        self.lstm_layer = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=0.5
        )
        self.last_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        embed_x = self.embed_layer(x)
        output, (_, _) = self.lstm_layer(embed_x)
        last_output = output[:, -1, :]
        last_output = self.last_layer(last_output)
        return last_output


# Train, Evaluate, Test 를 정의합니다.

# In[6]:


def train(model: nn.Module, iterator: Iterator, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str):
    model.train()
    epoch_loss = 0

    for _, batch in enumerate(iterator):
        optimizer.zero_grad()

        text = batch.text
        if text.shape[0] > 1:
            label = batch.label.type(torch.FloatTensor)
            text = text.to(device)
            label = label.to(device)
            output = model(text).flatten()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module, iterator: Iterator, criterion: nn.Module, device: str):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for _, batch in enumerate(iterator):
            text = batch.text
            label = batch.label.type(torch.FloatTensor)
            text = text.to(device)
            label = label.to(device)
            output = model(text).flatten()
            loss = criterion(output, label)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def test(model: nn.Module, iterator: Iterator, device: str):
    model.eval()
    with torch.no_grad():
        y_real = []
        y_pred = []
        for batch in iterator:
            text = batch.text
            label = batch.label.type(torch.FloatTensor)
            text = text.to(device)
            output = model(text).flatten().cpu()
            y_real += [label]
            y_pred += [output]
        y_real = torch.cat(y_real)
        y_pred = torch.cat(y_pred)

    fpr, tpr, _ = roc_curve(y_real, y_pred)
    auroc = auc(fpr, tpr)

    return auroc


def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# ## 모델 학습하기

# In[7]:


PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
N_EPOCHS = 20

lstm_classifier = LSTMClassifier(
    num_embeddings=len(TEXT.vocab),
    embedding_dim=100,
    hidden_size=200,
    num_layers=4,
    pad_idx=PAD_IDX,
)
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
_ = lstm_classifier.to(device)

optimizer = torch.optim.Adam(lstm_classifier.parameters())
bce_loss_fn = nn.BCELoss()

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(lstm_classifier, sat_train_iterator, optimizer, bce_loss_fn, device)
    valid_loss = evaluate(lstm_classifier, sat_valid_iterator, bce_loss_fn, device)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.5f}")
    print(f"\t Val. Loss: {valid_loss:.5f}")


# In[8]:


_ = lstm_classifier.cpu()
test_auroc = test(lstm_classifier, sat_test_iterator, "cpu")

print(f"SAT Dataset Test AUROC: {test_auroc:.5f}")


# In[9]:


with open("../save_model/sat_baseline_model.dill", "wb") as f:
    model = {
        "TEXT": TEXT,
        "LABEL": LABEL,
        "classifier": lstm_classifier
    }
    dill.dump(model, f)

