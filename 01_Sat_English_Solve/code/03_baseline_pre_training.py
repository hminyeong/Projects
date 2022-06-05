#!/usr/bin/env python
# coding: utf-8

import dill
from copy import deepcopy
import time
import random
import numpy as np
from sklearn.metrics import roc_curve, auc

import nltk

nltk.download("punkt")
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn

from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator
from torchtext.data import Iterator

RANDOM_SEED = 2020
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# DATA_PATH = "data/processed/"
DATA_PATH = "../data/processed/"
print('in')

# ## 데이터 불러오기
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


cola_train_data, cola_valid_data, cola_test_data = TabularDataset.splits(
    path=DATA_PATH,
    train="cola_train.tsv",
    validation="cola_valid.tsv",
    test="cola_test.tsv",
    format="tsv",
    fields=[("text", TEXT), ("label", LABEL)],
    skip_header=1,
)

TEXT.build_vocab(cola_train_data, min_freq=2)


cola_train_iterator, cola_valid_iterator, cola_test_iterator = BucketIterator.splits(
    (cola_train_data, cola_valid_data, cola_test_data),
    batch_size=32,
    device=None,
    sort=False,
)


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


# ## LSTM Classifier
class LSTMClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers, pad_idx):
        super().__init__()
        self.embed_layer = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=pad_idx)
        self.lstm_layer = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, dropout=0.5
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

    with torch.no_grad():
        y_real = []
        y_pred = []
        model.eval()
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


# ## CoLA 데이터를 이용해 사전학습
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

    train_loss = train(lstm_classifier, cola_train_iterator, optimizer, bce_loss_fn, device)
    valid_loss = evaluate(lstm_classifier, cola_valid_iterator, bce_loss_fn, device)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.5f}")
    print(f"\t Val. Loss: {valid_loss:.5f}")


before_tuning_lstm_classifier = deepcopy(lstm_classifier)


# ## 수능 데이터를 이용해 추가 학습 (Fine-Tune)
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
N_EPOCHS = 20


for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(lstm_classifier, sat_train_iterator, optimizer, bce_loss_fn, device)
    valid_loss = evaluate(lstm_classifier, sat_valid_iterator, bce_loss_fn, device)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.5f}")
    print(f"\t Val. Loss: {valid_loss:.5f}")



# ## 모델 성능 확인하기
_ = before_tuning_lstm_classifier.cpu()
lstm_sat_test_auroc = test(before_tuning_lstm_classifier, sat_test_iterator, "cpu")

_ = lstm_classifier.cpu()
lstm_tuned_test_auroc = test(lstm_classifier, sat_test_iterator, "cpu")

print(f"Before fine-tuning SAT Dataset Test AUROC: {lstm_sat_test_auroc:.5f}")
print(f"After fine-tuning SAT Dataset Test AUROC: {lstm_tuned_test_auroc:.5f}")


with open("../save_model/sat_before_tuning_model.dill", "wb") as f:
    model = {
        "TEXT": TEXT,
        "LABEL": LABEL,
        "classifier": before_tuning_lstm_classifier
    }
    dill.dump(model, f)

_ = lstm_classifier.cpu()
with open("../save_model/sat_after_tuning_model.dill", "wb") as f:
    model = {
        "TEXT": TEXT,
        "LABEL": LABEL,
        "classifier": lstm_classifier
    }
    dill.dump(model, f)

