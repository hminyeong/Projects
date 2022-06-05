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

# "sat_baseline_model.dill",
# "sat_before_tuning_model.dill",
# "sat_after_tuning_model.dill",
model_list = [
    "sat_after_tuning_model.dill",
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

# 2 1 5 5 3 4
# 2 1 1 1 3 1
problem_2106 = [
    "People from more individualistic cultural contexts tend to be motivated to maintain self-focused agency or control 1 as these serve as the basis of one’s self-worth.",
    "With this form of agency comes the belief that individual successes 2 depending primarily on one’s own abilities and actions, and thus, whether by influencing the environment or trying to accept one’s circumstances, the use of control ultimately centers on the individual.",
    "The independent self may be more 3 driven to cope by appealing to a sense of agency or control.",
    "Research has shown 4 that East Asians prefer to receive, but not seek, more social support rather than seek personal control in certain cases.",
    "Therefore, people 5 who hold a more interdependent self-construal may prefer to cope in a way that promotes harmony in relationships.",
]
problem_2106_label = [1, 0, 1, 1, 1]


problem_2109 = [
    "Competitive activities can be more than just performance showcases which the best is recognized and the rest are overlooked.",
    "The provision of timely, constructive feedback to participants on performance is an asset that some competitions and contests offer.",
    "The provision of that type of feedback can be interpreted as shifting the emphasis to demonstrating superior performance but not necessarily excellence.",
    "The emphasis on superiority is what we typically see as fostering a detrimental effect of competition.",
    "Information about performance can be very helpful, not only to the participant who does not win or place but also to those who do.",
]
problem_2109_label = [0, 1, 1, 1, 1]


problem_21_test = [
    "Scientists who experiment on themselves can, functionally if not legally, avoid the restrictions associated with experimenting on other people.",
    "nobody, presumably, is more aware of an experiment’s potential hazards than the scientist who devised it.",
    "Nonetheless, experimenting on oneself remains deeply problematic.",
    "One obvious drawback is the danger involved; knowing that it exists does nothing to reduce it.",
    "Experimental results derived from a single subject are, therefore, of limited value; there is no way to know what the subject’s responses are typical or atypical of the response of humans as a group.",
]
problem_21_test_label = [1, 1, 1, 1, 0]


problem_2206 = [
    "Early astronomy provided information about when to plant crops and gave humans their first formal method of recording the passage of time.",
    "Stonehenge, the 4,000-year-old ring of stones in southern Britain, is perhaps the best-known monument to the discovery of regularity and predictability in the world we inhabit.",
    "The great markers of Stonehenge point to the spots on the horizon where the sun rises at the solstices and equinoxes the dates we still use to mark the beginnings of the seasons.",
    "The stones may even have been used to predict eclipses.",
    "The existence of Stonehenge, built by people without writing, bears silent testimony both to the regularity of nature and to the ability of the human mind to see behind immediate appearances and discovers deeper meanings in events.",
]
problem_2206_label = [1, 1, 1, 1, 0]


problem_2209 = [
    "As far as communication between humans is concerned, such commonality of interests is rarely achieved.",
    "A prey can convince a predator not to chase it.",
    "But for such communication to occur, there must be strong guarantees which those who receive the signal will be better off believing it.",
    "The messages have to be kept, on the whole, honest.",
    "In the case of humans, honesty is maintained by a set of cognitive mechanisms that evaluate communicated information.",

]
problem_2209_label = [1, 1, 0, 1, 1]


problem_22_test = [
    "A cell is 'born' as a twin when its mother cell divides, producing two daughter cells.",
    "Each daughter cell is smaller than the mother cell, and except for unusual cases, each grows until it becomes as large as the mother cell was.",
    "After the cell has grown to the proper size, its metabolism shifts as it either prepares to divide or matures and differentiates into a specialized cell.",
    "What cell metabolism and structure should be complex would not be surprising, but actually, they are rather simple and logical.",
    "Even the most complex cell has only a small number of parts, each responsible for a distinct, well defined aspect of cell life.",

]
problem_22_test_label = [1, 1, 1, 0, 1]



# use full page width
# st.set_page_config(page_title="Example", layout="wide")

# Streamlit Timeline Component Example
import streamlit as st


# load data
with open('example.json', "r", encoding="utf-8") as f:
    data = f.read()

import time as ts
from datetime import time
from PIL import Image

# render timeline
def converter(value):
    m, s, mm = value.split(":")
    t_s = int(m)*60 + int(s) + int(mm)/1000
    return t_s


def main():
    st.markdown(
        "<h3 style='text-align: center; color: Black;'>영어 어휘 문제 with 인공지능</h3>",
        unsafe_allow_html=True)

    answer = ""
    col1, col2 = st.columns([7, 4])

    with col2:
        st.write(" ")

        password = st.text_input("암호를 입력하세요.", type="password")

        if password == 'alsdud':
            user_name = '허민영'
            st.write(" ")
            st.write(" ")

            select = st.selectbox("문항을 선택하세요.", (
                "-",
                "2021년 6월 모의고사", "2021년 9월 모의고사", "2021년 수능",
                "2022년 6월 모의고사", "2022년 9월 모의고사", "2022년 수능"))

            print(select)

            st.write(" ")
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

            user_answer = st.text_input(f"{user_name} 학생이 생각하는 정답은?")


    with col1:
        if password == 'alsdud':
            if select == "2021년 6월 모의고사": ### problem_2106
                real_answer = "2"
                st.markdown(
                    "<h5 style='text-align: center; color: Black;'><br>2021년 6월 모의고사</h5>",
                    unsafe_allow_html=True)
                image = Image.open('../img/2021-6.jpg')
                st.image(image)
                if st.button("Predict"):
                    ai_answer = predict_problem_with_models(model_list, problem_2106)
                    st.write(f'   학생이 예측한 정답: {user_answer}번') #2
                    st.write(f'인공지능이 예측한 정답: {ai_answer}번') #2
                    if real_answer == user_answer:
                        st.markdown(
                            "<p style='text-align: left; color: Green;'>   학생: 정답을 맞췄어요!</p>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<p style='text-align: left; color: red;'>   학생: 틀렸어요ㅠㅠ</p>",
                            unsafe_allow_html=True)
                    if real_answer == str(ai_answer):
                        st.markdown(
                            "<p style='text-align: left; color: Green;'>인공지능: 정답을 맞췄어요!</p>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<p style='text-align: left; color: red;'>인공지능: 틀렸어요ㅠㅠ</p>",
                            unsafe_allow_html=True)

            elif select == "2021년 9월 모의고사": ### problem_2109
                real_answer = "1"
                st.markdown(
                    "<h5 style='text-align: center; color: Black;'><br>2021년 9월 모의고사</h5>",
                    unsafe_allow_html=True)
                image = Image.open('../img/2021-9.jpg')
                st.image(image)
                if st.button("Predict"):
                    ai_answer = predict_problem_with_models(model_list, problem_2109)
                    st.write(f'{user_name} 학생이 예측한 정답: {user_answer}번') #
                    st.write(f'인공지능이 예측한 정답: {ai_answer}번') #1
                    if real_answer == user_answer:
                        st.markdown(
                            "<p style='text-align: left; color: Green;'>학생: 정답을 맞췄어요!</p>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<p style='text-align: left; color: red;'>학생: 틀렸어요ㅠㅠ</p>",
                            unsafe_allow_html=True)
                    if real_answer == str(ai_answer):
                        st.markdown(
                            "<p style='text-align: left; color: Green;'>인공지능: 정답을 맞췄어요!</p>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<p style='text-align: left; color: red;'>인공지능: 틀렸어요ㅠㅠ</p>",
                            unsafe_allow_html=True)

            elif select == "2021년 수능":
                real_answer = "5"
                st.markdown(
                    "<h5 style='text-align: center; color: Black;'><br>2021년 수능</h5>",
                    unsafe_allow_html=True)
                image = Image.open('../img/2021-test.jpg')
                st.image(image)
                if st.button("Predict"):
                    ai_answer = predict_problem_with_models(model_list, problem_21_test)
                    st.write(f'{user_name} 학생이 예측한 정답: {user_answer}번') #
                    st.write(f'인공지능이 예측한 정답: {ai_answer}번') #5
                    if real_answer == user_answer:
                        st.markdown(
                            "<p style='text-align: left; color: Green;'>학생: 정답을 맞췄어요!</p>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<p style='text-align: left; color: red;'>학생: 틀렸어요ㅠㅠ</p>",
                            unsafe_allow_html=True)
                    if real_answer == str(ai_answer):
                        st.markdown(
                            "<p style='text-align: left; color: Green;'>인공지능: 정답을 맞췄어요!</p>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<p style='text-align: left; color: red;'>인공지능: 틀렸어요ㅠㅠ</p>",
                            unsafe_allow_html=True)

            elif select == "2022년 6월 모의고사":
                real_answer = "5"
                st.markdown(
                    "<h5 style='text-align: center; color: Black;'><br>2022년 6월 모의고사</h5>",
                    unsafe_allow_html=True)
                image = Image.open('../img/2022-6.jpg')
                st.image(image)
                if st.button("Predict"):
                    ai_answer = predict_problem_with_models(model_list, problem_2206)
                    st.write(f'{user_name} 학생이 예측한 정답: {user_answer}번') #
                    st.write(f'인공지능이 예측한 정답: {ai_answer}번') #5
                    if real_answer == user_answer:
                        st.markdown(
                            "<p style='text-align: left; color: Green;'>학생: 정답을 맞췄어요!</p>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<p style='text-align: left; color: red;'>학생: 틀렸어요ㅠㅠ</p>",
                            unsafe_allow_html=True)
                    if real_answer == str(ai_answer):
                        st.markdown(
                            "<p style='text-align: left; color: Green;'>인공지능: 정답을 맞췄어요!</p>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<p style='text-align: left; color: red;'>인공지능: 틀렸어요ㅠㅠ</p>",
                            unsafe_allow_html=True)

            elif select == "2022년 9월 모의고사":
                real_answer = "3"
                st.markdown(
                    "<h5 style='text-align: center; color: Black;'><br>2022년 9월 모의고사</h5>",
                    unsafe_allow_html=True)
                image = Image.open('../img/2022-9.jpg')
                st.image(image)
                if st.button("Predict"):
                    ai_answer = predict_problem_with_models(model_list, problem_2209)
                    st.write(f'{user_name} 학생이 예측한 정답: {user_answer}번') #
                    st.write(f'인공지능이 예측한 정답: {ai_answer}번') #3

                    if real_answer == user_answer:
                        st.markdown(
                            "<p style='text-align: left; color: Green;'>학생: 정답을 맞췄어요!</p>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<p style='text-align: left; color: red;'>학생: 틀렸어요ㅠㅠ</p>",
                            unsafe_allow_html=True)
                    if real_answer == str(ai_answer):
                        st.markdown(
                            "<p style='text-align: left; color: Green;'>인공지능: 정답을 맞췄어요!</p>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<p style='text-align: left; color: red;'>인공지능: 틀렸어요ㅠㅠ</p>",
                            unsafe_allow_html=True)

            elif select == "2022년 수능":
                real_answer = "4"
                st.markdown(
                    "<h5 style='text-align: center; color: Black;'><br>2022년 수능</h5>",
                    unsafe_allow_html=True)
                image = Image.open('../img/2022-test.jpg')
                st.image(image)
                if st.button("Predict"):
                    ai_answer = predict_problem_with_models(model_list, problem_22_test)
                    st.write(f'{user_name} 학생이 예측한 정답: {user_answer}번') #
                    st.write(f'인공지능이 예측한 정답: {ai_answer}번') #4

                    if real_answer == user_answer:
                        st.markdown(
                            "<p style='text-align: left; color: Green;'>학생: 정답을 맞췄어요!</p>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<p style='text-align: left; color: red;'>학생: 틀렸어요ㅠㅠ</p>",
                            unsafe_allow_html=True)
                    if real_answer == str(ai_answer):
                        st.markdown(
                            "<p style='text-align: left; color: Green;'>인공지능: 정답을 맞췄어요!</p>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<p style='text-align: left; color: red;'>인공지능: 틀렸어요ㅠㅠ</p>",
                            unsafe_allow_html=True)


if __name__ == '__main__':
    main()