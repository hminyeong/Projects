#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dill
import random
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
import nltk

nltk.download("punkt")
from nltk.tokenize import word_tokenize

from torchtext.data import TabularDataset
from torchtext.data import BucketIterator

# In[2]:


RANDOM_SEED = 2020
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# DATA_PATH = "data/processed/"
DATA_PATH = "../save_model/"


# In[3]:


class LSTMClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers,
                 pad_idx):
        super().__init__()
        self.embed_layer = nn.Embedding(num_embeddings=num_embeddings,
                                        embedding_dim=embedding_dim,
                                        padding_idx=pad_idx)
        self.lstm_layer = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_size,
            num_layers=num_layers, bidirectional=True, dropout=0.5
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


class LSTMPoolingClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers,
                 pad_idx):
        super(LSTMPoolingClassifier, self).__init__()
        self.embed_layer = nn.Embedding(num_embeddings=num_embeddings,
                                        embedding_dim=embedding_dim,
                                        padding_idx=pad_idx)
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.ih2h = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.pool2o = nn.Linear(2 * hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.embed_layer(x)
        o, _ = self.ih2h(x)
        pool = nn.functional.max_pool1d(o.transpose(1, 2), x.shape[1])
        pool = pool.transpose(1, 2).squeeze()
        pool = self.dropout(pool)
        output = self.sigmoid(self.pool2o(pool))
        return output.squeeze()


# In[6]:


def test(model_path):
    with open(model_path, "rb") as f:
        model = dill.load(f)

    sat_test_data = TabularDataset(
        path=f"../data/processed/sat_test.tsv",
        format="tsv",
        fields=[("text", model["TEXT"]), ("label", model["LABEL"])],
        skip_header=1
    )

    sat_test_iterator = BucketIterator(
        sat_test_data,
        batch_size=8,
        device=None,
        sort=False,
        shuffle=False
    )
    classifier = model["classifier"]
    with torch.no_grad():
        y_real = []
        y_pred = []
        classifier.eval()
        for batch in sat_test_iterator:
            text = batch.text
            label = batch.label.type(torch.FloatTensor)

            output = classifier(text).flatten().cpu()

            y_real += [label]
            y_pred += [output]

        y_real = torch.cat(y_real)
        y_pred = torch.cat(y_pred)

    fpr, tpr, _ = roc_curve(y_real, y_pred)
    auroc = auc(fpr, tpr)

    return auroc.round(5)


# In[7]:


# model_list = [
#     "/opt/ml/input/my/deep-learning-with-projects/08_수능_영어_풀기/minyeong/model/sat_baseline_model.dill",
#     "/opt/ml/input/my/deep-learning-with-projects/08_수능_영어_풀기/minyeong/model/sat_before_tuning_model.dill",
#     "/opt/ml/input/my/deep-learning-with-projects/08_수능_영어_풀기/minyeong/model/sat_after_tuning_model.dill",
#     "/opt/ml/input/my/deep-learning-with-projects/08_수능_영어_풀기/minyeong/model/sat_advanced_before_tuning_model.dill",
#     "/opt/ml/input/my/deep-learning-with-projects/08_수능_영어_풀기/minyeong/model/sat_advanced_after_tuning_model.dill",
# ]
model_list = [
    "../save_model/sat_baseline_model.dill",
]

test_auroc = []
for file_name in model_list:
    model_name = file_name.replace(".dill", "")
    auroc = test(file_name)
    test_auroc += [(model_name, auroc)]

# In[8]:


test_auroc = sorted(test_auroc, key=lambda x: x[1], reverse=True)
for rank, (model_name, auroc) in enumerate(test_auroc):
    print(f"Rank {rank + 1} - {model_name:30} - Test AUROC: {auroc:.5f}")


# In[9]:


def predict_problem(model_path, problem):
    with open(model_path, "rb") as f:
        model = dill.load(f)
    TEXT = model["TEXT"]
    classifier = model["classifier"]

    problem = list(map(lambda x: x.replace("[", "").replace("]", ""), problem))
    tokenized_sentences = [word_tokenize(sentence) for sentence in problem]
    sentences = []
    for tokenized_sentence in tokenized_sentences:
        sentences.append([TEXT.vocab.stoi[word] for word in tokenized_sentence])

    with torch.no_grad():
        classifier.eval()
        predict = []
        for sentence in sentences:
            sentence = torch.LongTensor([sentence])
            predict += [classifier(sentence).item()]
    return predict


def predict_problem_with_models(model_list, problem):
    scores = {}
    for file_name in model_list:
        model_name = file_name.replace(".dill", "")
        score = predict_problem(file_name, problem)
        scores[model_name] = score

    score_df = pd.DataFrame(scores).T
    score_df.columns = [f"answer_{i}_score" for i in range(1, 6)]

    selected_answer = pd.Series(np.argmin(score_df.values, 1) + 1,
                                index=score_df.index, name="selected_answer")

    return pd.concat([selected_answer, score_df], 1)


# In[10]:


problem_1 = [
    "Competitive activities can be more than just performance showcases which the best is recognized and the rest are overlooked.",
    "The provision of timely, constructive feedback to participants on performance is an asset that some competitions and contests offer.",
    "The provision of that type of feedback can be interpreted as shifting the emphasis to demonstrating superior performance but not necessarily excellence.",
    "The emphasis on superiority is what we typically see as fostering a detrimental effect of competition.",
    "Information about performance can be very helpful, not only to the participant who does not win or place but also to those who do.",
]
problem_1_label = [0, 1, 1, 1, 1]

# In[11]:


a = predict_problem_with_models(model_list, problem_1).loc[
    map(lambda x: x[0], test_auroc)]
print(a)

# In[12]:


# problem_2 = [
#     "People from more individualistic cultural contexts tend to be motivated to maintain self-focused agency or control 1 as these serve as the basis of one’s self-worth.",
#     "With this form of agency comes the belief that individual successes 2 depending primarily on one’s own abilities and actions, and thus, whether by influencing the environment or trying to accept one’s circumstances, the use of control ultimately centers on the individual.",
#     "The independent self may be more 3 driven to cope by appealing to a sense of agency or control.",
#     "Research has shown 4 that East Asians prefer to receive, but not seek, more social support rather than seek personal control in certain cases.",
#     "Therefore, people 5 who hold a more interdependent self-construal may prefer to cope in a way that promotes harmony in relationships.",
# ]
# problem_2_label = [1, 0, 1, 1, 1]
#
# # In[13]:
#
#
# b = predict_problem_with_models(model_list, problem_2).loc[
#     map(lambda x: x[0], test_auroc)]
# print(b)
