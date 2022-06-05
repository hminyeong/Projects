#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import time

import dill
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import nltk

nltk.download("punkt")
from nltk.tokenize import word_tokenize

RANDOM_SEED = 2020
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

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


model_list = [
    "sat_baseline_model.dill",
    # "sat_before_tuning_model.dill",
    # "sat_after_tuning_model.dill",
    # "sat_advanced_before_tuning_model.dill",
    # "sat_advanced_after_tuning_model.dill",
]

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

    # print("예측 정답 번호: ", selected_answer[-1])
    result = selected_answer[-1]
    # return pd.concat([selected_answer, score_df], 1)
    return result


problem_1 = [
    "Competitive activities can be more than just performance showcases which the best is recognized and the rest are overlooked.",
    "The provision of timely, constructive feedback to participants on performance is an asset that some competitions and contests offer.",
    "The provision of that type of feedback can be interpreted as shifting the emphasis to demonstrating superior performance but not necessarily excellence.",
    "The emphasis on superiority is what we typically see as fostering a detrimental effect of competition.",
    "Information about performance can be very helpful, not only to the participant who does not win or place but also to those who do.",
]


problem_1_label = [0, 1, 1, 1, 1]


# use full page width
st.set_page_config(page_title="Example", layout="wide")

# Streamlit Timeline Component Example
import streamlit as st
from streamlit_timeline import timeline


# load data
with open('example.json', "r", encoding="utf-8") as f:
    data = f.read()

import time as ts
from datetime import time

# render timeline
def converter(value):
    m, s, mm = value.split(":")
    t_s = int(m)*60 + int(s) + int(mm)/1000
    return t_s

def main():
    answer = ""
    col1, col2 = st.columns([3, 1])

    with col1:
        timeline(data, height=800)
    with col2:
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")

        # st.write("민영")
        # col2.subheader("민영")
        val = st.time_input("시간을 맞춰놓고 문제를 풀어보세요. (2:30초 권장)", value=time(0, 0, 0))
        if str(val) == "00:00:00":
            st.write("Please sent timer")
        else:
            sec = converter(str(val))
            bar = st.progress(0)
            per = sec / 100
            progress_status = st.empty()
            if st.button("start!"):
                for i in range(100):
                    bar.progress((i * 1))
                    progress_status.write((str(i + 1) + " %"))
                    ts.sleep(per)

            answer = ""
            if st.button("Predict"):
                answer = predict_problem_with_models(model_list, problem_1)
                st.write(f'인공지능이 예측한 정답은 : {answer}번이에요.')


if __name__ == '__main__':
    main()