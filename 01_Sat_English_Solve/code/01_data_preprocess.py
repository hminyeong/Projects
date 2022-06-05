#!/usr/bin/env python
# coding: utf-8

# In[16]:


import warnings
warnings.simplefilter(action='ignore')

import pandas as pd


# ## SAT datset

# In[17]:

# sat_data = pd.read_csv("/opt/ml/input/my/deep-learning-with-projects/08_수능_영어_풀기/minyeong/data/raw/all_data.csv")

sat_data = pd.read_csv("../data/raw/sat_problems.csv")
sat_data = sat_data[sat_data["question"].map(lambda x: "어법" in x)]


# In[18]:


sat_train = sat_data[(sat_data["year"] < 2020)]
sat_valid = sat_data[(sat_data["year"] == 2020)]
sat_test = sat_data[sat_data["year"] > 2020]


# In[19]:


print("# of train data:", sat_train.shape[0])
print("# of valid data:", sat_valid.shape[0])
print("# of test data:", sat_test.shape[0])


# In[20]:


def clean_bracket(string):
    string = string.replace("[", "")
    string = string.replace("]", "")
    return string


# In[21]:


sat_train["context"] = sat_train["context"].map(clean_bracket)
sat_valid["context"] = sat_valid["context"].map(clean_bracket)
sat_test["context"] = sat_test["context"].map(clean_bracket)


# In[22]:


sat_train["label"] = sat_train["label"].map(int)
sat_valid["label"] = sat_valid["label"].map(int)
sat_test["label"] = sat_test["label"].map(int)


# In[23]:

# sat_train.to_csv("/opt/ml/input/my/deep-learning-with-projects/08_수능_영어_풀기/minyeong/data/processed/all_train.csv", sep="\t", index=False)
# sat_valid.to_csv("/opt/ml/input/my/deep-learning-with-projects/08_수능_영어_풀기/minyeong/data/processed/all_valid.csv", sep="\t", index=False)
# sat_test.to_csv("/opt/ml/input/my/deep-learning-with-projects/08_수능_영어_풀기/minyeong/data/processed/all_test.csv", sep="\t", index=False)

sat_train.to_csv("../data/processed/sat_train.tsv", sep="\t", index=False)
sat_valid.to_csv("../data/processed/sat_valid.tsv", sep="\t", index=False)
sat_test.to_csv("../data/processed/sat_test.tsv", sep="\t", index=False)


# ## CoLA Daaset

# In[24]:


columns = ["source", "label", "original_judgement", "context"]
in_domian_train = pd.read_csv("../data/raw/cola/in_domain_train.tsv", sep="\t", header=None, names=columns)
in_domian_dev = pd.read_csv("../data/raw/cola/in_domain_dev.tsv", sep="\t", header=None, names=columns)
out_of_domian_dev = pd.read_csv("../data/raw/cola/out_of_domain_dev.tsv", sep="\t", header=None, names=columns)


# In[25]:


print("# of cola train data:", in_domian_train.shape[0])
print("# of cola valid data:", in_domian_dev.shape[0])
print("# of cola test data:", out_of_domian_dev.shape[0])


# In[26]:


in_domian_train[["context", "label"]].to_csv("../data/processed/cola_train.tsv", sep="\t", index=False)
in_domian_dev[["context", "label"]].to_csv("../data/processed/cola_valid.tsv", sep="\t", index=False)
out_of_domian_dev[["context", "label"]].to_csv("../data/processed/cola_test.tsv", sep="\t", index=False)

