# %%
import warnings
warnings.simplefilter(action='ignore')

import torch
import pandas as pd



# %%
## SAT datset

if torch.cuda.is_available():
    # CPU Local
    print("CPU Local")
    train = pd.read_csv("C:/Users/alsdu/Desktop/수능_Project/data/raw/train.csv")
    valid = pd.read_csv("C:/Users/alsdu/Desktop/수능_Project/data/raw/valid.csv")
    test = pd.read_csv("C:/Users/alsdu/Desktop/수능_Project/data/raw/test.csv")
else:
    # GPU
    print("GPU")
    train = pd.read_csv("C:/Users/alsdu/Desktop/수능_Project/data/raw/train.csv")
    valid = pd.read_csv("C:/Users/alsdu/Desktop/수능_Project/data/raw/valid.csv")
    test = pd.read_csv("C:/Users/alsdu/Desktop/수능_Project/data/raw/test.csv")


train = train[train["question"].map(lambda x: "어법" in x)]
valid = valid[valid["question"].map(lambda x: "어법" in x)]
test = test[test["question"].map(lambda x: "어법" in x)]

print("# SAT dataset preprocess done!")
print()


## CoLA Daaset
# %%
columns = ["source", "label", "original_judgement", "context"]
in_domian_train = pd.read_csv("data/raw/cola/in_domain_train.tsv", sep="\t", header=None, names=columns)
in_domian_dev = pd.read_csv("data/raw/cola/in_domain_dev.tsv", sep="\t", header=None, names=columns)
out_of_domian_dev = pd.read_csv("data/raw/cola/out_of_domain_dev.tsv", sep="\t", header=None, names=columns)

# %%
print("# of cola train data:", in_domian_train.shape[0])
print("# of cola valid data:", in_domian_dev.shape[0])
print("# of cola test data:", out_of_domian_dev.shape[0])

# %%
in_domian_train[["context", "label"]].to_csv("data/processed/cola_train.tsv", sep="\t", index=False)
in_domian_dev[["context", "label"]].to_csv("data/processed/cola_valid.tsv", sep="\t", index=False)
out_of_domian_dev[["context", "label"]].to_csv("data/processed/cola_test.tsv", sep="\t", index=False)

