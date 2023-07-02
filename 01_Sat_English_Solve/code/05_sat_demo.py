# %%
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

# %%
RANDOM_SEED = 2020
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

DATA_PATH = "C:/Users/alsdu/Desktop/수능_Project/data/processed/"

# %%
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


class LSTMPoolingClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers, pad_idx):
        super(LSTMPoolingClassifier, self).__init__()
        self.embed_layer = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=pad_idx)
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

# %%
def test(model_path):
    with open(model_path, "rb") as f:
        model = dill.load(f)
        
    sat_test_data = TabularDataset(
        path=f"{DATA_PATH}test.tsv",
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

# %%
model_list = [
    "C:/Users/alsdu/Desktop/수능_Project/streamlit/model/baseline_model.dill",
    "C:/Users/alsdu/Desktop/수능_Project/streamlit/model/before_tuning_model.dill", 
    "C:/Users/alsdu/Desktop/수능_Project/streamlit/model/after_tuning_model.dill",
    "C:/Users/alsdu/Desktop/수능_Project/streamlit/model/advanced_before_tuning_model.dill",
    "C:/Users/alsdu/Desktop/수능_Project/streamlit/model/advanced_after_tuning_model.dill"
    ]


test_auroc_temp = []
for file_name in model_list:
    model_name = file_name.replace(".dill", "")
    auroc = test(file_name)
    test_auroc_temp += [(model_name, auroc)]

# %%
test_auroc = sorted(test_auroc_temp, key=lambda x: x[1], reverse=True)
for rank, (model_name, auroc) in enumerate(test_auroc):  # 0 ('/root/수능_영어_풀기/models/0601/data2_advanced_before_tuning_model', 0.96154)
    print(f"Rank {rank+1} - {model_name:40} - Test AUROC: {auroc:.5f}")


# %%
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
    score_df.columns = [f"answer_{i}_score" for i in range(1,6)]

    selected_answer = pd.Series(np.argmin(score_df.values, 1)+1, index=score_df.index, name="selected_answer")
    
    return pd.concat([selected_answer, score_df], 1)



# ---------------------------------------------------------------------------평가원(고3)--------------------------------------------------------------------------- #
# 2 1 5 3 3 2
# -----2021----- #
evaluator_problem_2021_06 = [
    "People from more individualistic cultural contexts tend to be motivated to maintain self-focused agency or control 1 as these serve as the basis of one’s self-worth.",
    "With this form of agency comes the belief that individual successes 2 depending primarily on one’s own abilities and actions, and thus, whether by influencing the environment or trying to accept one’s circumstances, the use of control ultimately centers on the individual.",
    "The independent self may be more 3 driven to cope by appealing to a sense of agency or control.",
    "Research has shown 4 that East Asians prefer to receive, but not seek, more social support rather than seek personal control in certain cases.",
    "Therefore, people 5 who hold a more interdependent self-construal may prefer to cope in a way that promotes harmony in relationships."
]
evaluator_problem_2021_06_label = [1, 0, 1, 1, 1]
evaluator_p1_list = predict_problem_with_models(model_list, evaluator_problem_2021_06)
evaluator_p1_sorted = evaluator_p1_list.loc[map(lambda x:x[0], test_auroc)]
print("evaluator_problem_2021_06_label: ", evaluator_p1_sorted)
evaluator_p1_answer = evaluator_p1_sorted.iloc[[0], [0]].values[0][0]

evaluator_problem_2021_09 = [
    "Competitive activities can be more than just performance showcases which the best is recognized and the rest are overlooked.",
    "The provision of timely, constructive feedback to participants on performance is an asset that some competitions and contests offer.",
    "The provision of that type of feedback can be interpreted as shifting the emphasis to demonstrating superior performance but not necessarily excellence.",
    "The emphasis on superiority is what we typically see as fostering a detrimental effect of competition.",
    "Information about performance can be very helpful, not only to the participant who does not win or place but also to those who do."
]
evaluator_problem_2021_09_label = [0, 1, 1, 1, 1]
evaluator_p2_list = predict_problem_with_models(model_list, evaluator_problem_2021_09)
evaluator_p2_sorted = evaluator_p2_list.loc[map(lambda x:x[0], test_auroc)]
print("evaluator_problem_2021_09: ", evaluator_p2_sorted)
evaluator_p2_answer = evaluator_p2_sorted.iloc[[0], [0]].values[0][0]


# -----2022----- #
evaluator_problem_2022_06 = [
    "Early astronomy provided information about when to plant crops and gave humans their first formal method of recording the passage of time.",
    "Stonehenge, the 4,000-year-old ring of stones in southern Britain, is perhaps the best-known monument to the discovery of regularity and predictability in the world we inhabit.",
    "The great markers of Stonehenge point to the spots on the horizon where the sun rises at the solstices and equinoxes ― the dates we still use to mark the beginnings of the seasons.",
    "The stones may even have been used to predict eclipses.",
    "The existence of Stonehenge, built by people without writing, bears silent testimony both to the regularity of nature and to the ability of the human mind to see behind immediate appearances and discovers deeper meanings in events."
]
evaluator_problem_2022_06_label = [1, 1, 1, 1, 0]
evaluator_p3_list = predict_problem_with_models(model_list, evaluator_problem_2022_06)
evaluator_p3_sorted = evaluator_p3_list.loc[map(lambda x:x[0], test_auroc)]
print("evaluator_problem_2022_06: ", evaluator_p3_sorted)
evaluator_p3_answer = evaluator_p3_sorted.iloc[[0], [0]].values[0][0]

evaluator_problem_2022_09 = [ 
    "As far as communication between humans is concerned, such commonality of interests is rarely achieved; even a pregnant mother has reasons to mistrust the chemical signals sent by her fetus.",
    "A prey can convince a predator not to chase it.",
    "But for such communication to occur, there must be strong guarantees which those who receive the signal will be better off believing it.",
    "The messages have to be kept, on the whole, honest.",
    "In the case of humans, honesty is maintained by a set of cognitive mechanisms that evaluate communicated information."
]
evaluator_problem_2022_09_label = [1, 1, 0, 1, 1]
evaluator_p4_list = predict_problem_with_models(model_list, evaluator_problem_2022_09)
evaluator_p4_sorted = evaluator_p4_list.loc[map(lambda x:x[0], test_auroc)]
print("evaluator_problem_2022_09: ", evaluator_p4_sorted)
evaluator_p4_answer = evaluator_p4_sorted.iloc[[0], [0]].values[0][0]


# -----2023----- #
evaluator_problem_2023_06 = [
    "They can be defined as ranging from the communities and interactions of organisms in your mouth or those in the canopy of a rain forest to all those in Earth’s oceans.",
    "The processes governing them differ in complexity and speed.",
    "There are systems that turn over in minutes, and there are others which rhythmic time extends to hundreds of years.",
    "Divide an ecosystem into parts by creating barriers, and the sum of the productivity of the parts will typically be found to be lower than the productivity of the whole, other things being equal.",
    "Safe passages, for example, enable migratory species to survive."
]
evaluator_problem_2023_06_label = [1, 1, 0, 1, 1]
evaluator_p5_list = predict_problem_with_models(model_list, evaluator_problem_2023_06)
evaluator_p5_sorted = evaluator_p5_list.loc[map(lambda x:x[0], test_auroc)]
print("evaluator_problem_2023_06: ", evaluator_p5_sorted)
evaluator_p5_answer = evaluator_p5_sorted.iloc[[0], [0]].values[0][0]

evaluator_problem_2023_09 = [ 
    "An ethical issue is an identifiable problem, situation, or opportunity that requires a person to choose from among several actions that may be evaluated as right or wrong, ethical or unethical.",
    "Learn how to choose from alternatives and make a decision requires not only good personal values, but also knowledge competence in the business area of concern.",
    "Employees also need to know when to rely on their organizations’ policies and codes of ethics or have discussions with co-workers or managers on appropriate conduct.",
    "Ethical decision making is not always easy because there are always gray areas that create dilemmas, no matter how decisions are made.",
    "Such questions require the decision maker to evaluate the ethics of his or her choice and decide whether to ask for guidance."
]
evaluator_problem_2023_09_label = [1, 0, 1, 1, 1]
evaluator_p6_list = predict_problem_with_models(model_list, evaluator_problem_2023_09)
evaluator_p6_sorted = evaluator_p6_list.loc[map(lambda x:x[0], test_auroc)]
print("evaluator_problem_2023_09: ", evaluator_p6_sorted)
evaluator_p6_answer = evaluator_p6_sorted.iloc[[0], [0]].values[0][0]

print("\n평가원 2021~2023 기출 실제 정답: ", "2 1 5 3 3 2")
print("평가원 2021~2023 기출 예측 정답: ", evaluator_p1_answer, evaluator_p2_answer, evaluator_p3_answer, evaluator_p4_answer, evaluator_p5_answer, evaluator_p6_answer)


# ---------------------------------------------------------------------------서울(고1~고3)--------------------------------------------------------------------------- #
# 3 5 5
# 고1
# -----2021----- #
seoul_problem_2021_03 = [ 
    "Although there is usually a correct way of holding and playing musical instruments, the most important instruction to begin with is that they are not toys and that they must be looked after.",
    "Allow children time to explore ways of handling and playing the instruments for themselves before showing them.",
    "Finding different ways to produce sounds are an important stage of musical exploration.",
    "Correct playing comes from the desire to find the most appropriate sound quality and find the most comfortable playing position so that one can play with control over time.",
    "As instruments and music become more complex, learning appropriate playing techniques becomes increasingly relevant."
]
seoul_problem_2021_03_label = [1, 1, 0, 1, 1]
seoul_p7_list = predict_problem_with_models(model_list, seoul_problem_2021_03)
seoul_p7_sorted = seoul_p7_list.loc[map(lambda x:x[0], test_auroc)]
print("seoul_problem_2021_03: ", seoul_p7_sorted)
seoul_p7_answer = seoul_p7_sorted.iloc[[0], [0]].values[0][0]

# -----2022----- #
seoul_problem_2022_03 = [ 
    "It’s why places like Little Italy, Chinatown, and Koreatown exist.",
    "I’m talking about people who share our values and look at the world the same way we do.",
    "This is a very common human tendency what is rooted in how our species developed.",
    "You would be conditioned to avoid something unfamiliar or foreign because there is a high likelihood that it would be interested in killing you.",
    "Similarities make us relate better to other people because we think they’ll understand us on a deeper level than other people."
]
seoul_problem_2022_03_label = [1, 1, 1, 1, 0]
seoul_p8_list = predict_problem_with_models(model_list, seoul_problem_2022_03)
seoul_p8_sorted = seoul_p8_list.loc[map(lambda x:x[0], test_auroc)]
print("seoul_problem_2022_03: ", seoul_p8_sorted)
seoul_p8_answer = seoul_p8_sorted.iloc[[0], [0]].values[0][0]

# -----2023----- #
seoul_problem_2023_03 = [ 
    "The most noticeable human characteristic projected onto animals is that they can talk in human language.",
    "Physically, animal cartoon characters and toys made after animals are also most often deformed in such a way as to resemble humans.",
    "This is achieved by showing them with humanlike facial features and deformed front legs to resemble human hands.",
    "However, they still use their front legs like human hands (for example, lions can pick up and lift small objects with one paw), and they still talk with an appropriate facial expression.",
    "A general strategy that is used to make the animal characters more emotionally appealing, both to children and adults, are to give them enlarged and deformed childlike features."
]
seoul_problem_2023_03_label = [1, 1, 1, 1, 0]
seoul_p9_list = predict_problem_with_models(model_list, seoul_problem_2023_03)
seoul_p9_sorted = seoul_p9_list.loc[map(lambda x:x[0], test_auroc)]
print("seoul_problem_2023_03: ", seoul_p9_sorted)
seoul_p9_answer = seoul_p9_sorted.iloc[[0], [0]].values[0][0]

# 5 4 5
# 고2
# -----2021----- #
seoul2_problem_2021_03 = [
    "While reflecting on the needs of organizations, leaders, and families today, we realize that one of the unique characteristics is inclusivity.",
    "Because inclusivity supports what everyone ultimately wants from their relationships: collaboration.",
    "Yet the majority of leaders, organizations, and families are still using the language of the old paradigm in which one person — typically the oldest, most educated, and/or wealthiest —makes all the decisions, and their decisions rule with little discussion or inclusion of others, resulting in exclusivity.",
    "There is no need for others to present their ideas because they are considered inadequate.",
    "Yet research shows that exclusivity in problem solving, even with a genius, is not as effective as inclusivity, which everyone’s ideas are heard and a solution is developed through collaboration."
]
seoul2_problem_2021_03_label = [1, 1, 1, 1, 0]
seoul2_p10_list = predict_problem_with_models(model_list, seoul2_problem_2021_03)
seoul2_p10_sorted = seoul2_p10_list.loc[map(lambda x:x[0], test_auroc)]
print("seoul2_problem_2021_03: ", seoul2_p10_sorted)
seoul_p10_answer = seoul2_p10_sorted.iloc[[0], [0]].values[0][0]
# -----2022----- #
seoul2_problem_2022_03 = [
    "We’re not only meaning seeking creatures but social ones as well, constantly making interpersonal comparisons to evaluate ourselves,",
    "When comparing ourselves to someone who’s doing better than we are, we often feel inadequate for not doing as well. ",
    "This sometimes leads to what psychologists call malignant envy, the desire for someone to meet with misfortune (“I wish she didn’t have what she has”). ",
    "Also, comparing ourselves with someone who’s doing worse than we are risk scorn, the feeling that others are something undeserving of our beneficence (“She’s beneath my notice”). ",
    "Then again, comparing ourselves to others can also lead to benign envy, the longing to reproduce someone else’s accomplishments without wishing them ill (“I wish I had what she has”), which has been shown in some circumstances to inspire and motivate us to increase our efforts in spite of a recent failure."
]
seoul2_problem_2022_03_label = [1, 1, 1, 0, 1]
seoul2_p11_list = predict_problem_with_models(model_list, seoul2_problem_2022_03)
seoul2_p11_sorted = seoul2_p11_list.loc[map(lambda x:x[0], test_auroc)]
print("seoul2_problem_2022_03: ", seoul2_p11_sorted)
seoul_p11_answer = seoul2_p11_sorted.iloc[[0], [0]].values[0][0]
# -----2023----- #
seoul2_problem_2023_03 = [
    "This liking stems from our ancient ancestors who needed to survive alongside saber‑toothed tigers and poisonous berries. ",
    "Our brains evolved to help us attend to threats, keep away from them, and remain alive afterward. ",
    "In fact, we learned that the more certain we were about something, the better chance we had of making the right choice. ",
    "If I know for certain it is, my brain will direct me to eat it because I know it’s safe. ",
    "Our brains then generating sensations, thoughts, and action plans to keep us safe from the uncertain element, and we live to see another day."
]
seoul2_problem_2023_03_label = [1, 1, 1, 1, 0]
seoul2_p12_list = predict_problem_with_models(model_list, seoul2_problem_2023_03)
seoul2_p12_sorted = seoul2_p12_list.loc[map(lambda x:x[0], test_auroc)]
print("seoul2_problem_2023_03: ", seoul2_p12_sorted)
seoul_p12_answer = seoul2_p12_sorted.iloc[[0], [0]].values[0][0]


# 4 2 2 4 5
# 고3
# -----2021----- #
seoul3_problem_2021_03 = [
    "At the simplest level are the occasional trips made by individual !Kung and Dani to visit their individual trading partners in other bands or villages. ",
    "Suggestive of our open air markets and flea markets were the occasional markets at which Sio villagers living on the coast of northeast New Guinea met New Guineans from inland villages. ",
    "Up to a few dozen people from each side sat down in rows facing each other. ",
    "An inlander pushed forward a net bag containing between 10 and 35 pounds of taro and sweet potatoes, and the Sio villager sitting opposite responded by offering a number of pots and coconuts judging equivalent in value to the bag of food. ",
    "Trobriand Island canoe traders conducted similar markets on the islands that they visited, exchanging utilitarian goods (food, pots, and bowls) by barter, at the same time as they and their individual trade partners gave each other reciprocated gifts of luxury items (shell necklaces and armbands)."
]
seoul3_problem_2021_03_label = [1, 1, 1, 0, 1]
seoul3_p13_list = predict_problem_with_models(model_list, seoul3_problem_2021_03)
seoul3_p13_sorted = seoul3_p13_list.loc[map(lambda x:x[0], test_auroc)]
print("seoul3_problem_2021_03: ", seoul3_p13_sorted)
seoul_p13_answer = seoul3_p13_sorted.iloc[[0], [0]].values[0][0]

seoul3_problem_2021_10 = [
    "This genre is dominated, although not exclusively, by football and has produced a number of examples where popular songs become synonymous with the club and are enthusiastically adopted by the fans. ",
    "More than this they are often spontaneous expressions of loyalty and identity and, according to Desmond Morris, have ‘reached the level of something approached a local art form’. ",
    "A strong element of the appeal of such sports songs is that they feature ‘memorable and easily sung choruses in which fans can participate’. ",
    "This is a vital part of the team’s performance as it makes the fans’ presence more tangible. ",
    "This form of popular culture can be said to display pleasure and emotional excess in contrast to the dominant culture which tends to maintain ‘respectable aesthetic distance and control’."
]
seoul3_problem_2021_10_label = [1, 0, 1, 1, 1]
seoul3_p14_list = predict_problem_with_models(model_list, seoul3_problem_2021_10)
seoul3_p14_sorted = seoul3_p14_list.loc[map(lambda x:x[0], test_auroc)]
print("seoul3_problem_2021_10: ", seoul3_p14_sorted)
seoul_p14_answer = seoul3_p14_sorted.iloc[[0], [0]].values[0][0]

# -----2022----- #
seoul3_problem_2022_03 = [
    "We don’t know what ancient Greek music sounded like, because there are no examples of it in written or notated form, nor has it survived in oral tradition. ",
    "So we are forced largely to guess at its basis from the accounts of writers such as Plato and Aristotle, who were generally more concerned with writing about music as a philosophical and ethical exercise as with providing a technical primer on its practice. ",
    "It seems Greek music was predominantly a vocal form, consisting of sung verse accompanied by instruments such as the lyre or the plucked kithara (the root of ‘guitar’). ",
    "In fact, Plato considered music in which the lyre and flute played alone and not as the accompaniment of dance or song to be ‘exceedingly coarse and tasteless’. ",
    "The melodies seem to have had a very limited pitch range, since the instruments generally span only an octave, from one E (as we’d now define it) to the next."
]
seoul3_problem_2022_03_label = [1, 0, 1, 1, 1]
seoul3_p15_list = predict_problem_with_models(model_list, seoul3_problem_2022_03)
seoul3_p15_sorted = seoul3_p15_list.loc[map(lambda x:x[0], test_auroc)]
print("seoul3_problem_2022_03: ", seoul3_p15_sorted)
seoul_p15_answer = seoul3_p15_sorted.iloc[[0], [0]].values[0][0]

seoul3_problem_2022_10 = [
    "The idea that leaders inherently possess certain physical, intellectual, or personality traits that distinguish them from nonleaders was the foundational belief of the trait‑based approach to leadership. ",
    "Early trait theorists believed that some individuals are born with the traits that allow them to become great leaders. ",
    "Thus, early research in this area often presented the widely stated argument that “leaders are born, not made.” ",
    "Also, some of the earliest leadership studies were grounded in what referred to as the “great man” theory because researchers at the time focused on identifying traits of highly visible leaders in history who were typically male and associated with the aristocracy or political or military leadership. ",
    "In more recent history, numerous authors have acknowledged that there are many enduring qualities, whether innate or learned, that contribute to leadership potential. "
]
seoul3_problem_2022_10_label = [1, 1, 1, 0, 1]
seoul3_p16_list = predict_problem_with_models(model_list, seoul3_problem_2022_10)
seoul3_p16_sorted = seoul3_p16_list.loc[map(lambda x:x[0], test_auroc)]
print("seoul3_problem_2022_10: ", seoul3_p16_sorted)
seoul_p16_answer = seoul3_p16_sorted.iloc[[0], [0]].values[0][0]

# -----2023----- #
seoul3_problem_2023_03 = [
    "From the 8th to the 12th century CE, while Europe suffered the perhaps overdramatically named Dark Ages, science on planet Earth could be found almost exclusively in the Islamic world.",
    "This science was not exactly like our science today, but it was surely antecedent to it and was nonetheless an activity aimed at knowing about the world. ",
    "Great schools in all the cities covering the Arabic Near East and Northern Africa (and even into Spain) trained generations of scholars. ",
    "Almost every word in the modern scientific lexicon that begins with the prefix “al” owes its origins to Islamic science — algorithm, alchemy, alcohol, alkali, algebra. ",
    "And then, just over 400 years after it started, it ground to an apparent halt, and it would be a few hundred years, give or take, before that we would today unmistakably recognize as science appeared in Europe —with Galileo, Kepler, and, a bit later, Newton."
]
seoul3_problem_2023_03_label = [1, 1, 1, 1, 0]
seoul3_p17_list = predict_problem_with_models(model_list, seoul3_problem_2023_03)
seoul3_p17_sorted = seoul3_p17_list.loc[map(lambda x:x[0], test_auroc)]
print("seoul3_problem_2023_03: ", seoul3_p17_sorted)
seoul_p17_answer = seoul3_p17_sorted.iloc[[0], [0]].values[0][0]


print("\n서울 2021~2023 기출 실제 정답: ", "3 5 5 5 4 5 4 2 2 4 5")
print("서울 2021~2023 기출 예측 정답: ", seoul_p7_answer, seoul_p8_answer, seoul_p9_answer,
                                      seoul_p10_answer, seoul_p11_answer, seoul_p12_answer,
                                      seoul_p13_answer, seoul_p14_answer, seoul_p15_answer, seoul_p16_answer, seoul_p17_answer)


# ---------------------------------------------------------------------------인천(고1~고3)--------------------------------------------------------------------------- #
# 2 4 3 4 5 4
# 고1
# -----2021----- #
incheon1_problem_2021_08 = [ 
    "The money from anything that’s produced is used to buy something else.",
    "There can never be a situation which a firm finds that it can’t sell its goods and so has to dismiss workers and close its factories.",
    "Say’s Law applies because people use all their earnings to buy things.",
    "Savings are a ‘leakage’ of spending from the economy.",
    "That would mean firms producing less and dismissing some of their workers."
]
incheon1_problem_2021_08_label = [1, 0, 1, 1, 1]
incheon1_p10_list = predict_problem_with_models(model_list, incheon1_problem_2021_08)
incheon1_p10_sorted = incheon1_p10_list.loc[map(lambda x:x[0], test_auroc)]
print("incheon1_problem_2021_08: ", incheon1_p10_sorted)
incheon1_p10_answer = incheon1_p10_sorted.iloc[[0], [0]].values[0][0]
# -----2022----- #
incheon1_problem_2022_08 = [ 
    "The human brain, it turns out, has shrunk in mass by about 10 percent since it peaked in size 15,000-30,000 years ago.",
    "One possible reason is that many thousands of years ago humans lived in a world of dangerous predators where they had to have their wits about them at all times to avoid being killed.",
    "Today, we have effectively domesticated ourselves and many of the tasks of survival ― from avoiding immediate death to building shelters to obtaining food ― has been outsourced to the wider society.",
    "We are smaller than our ancestors too, and it is a characteristic of domestic animals that they are generally smaller than their wild cousins.",
    "None of this may mean we are dumber ― brain size is not necessarily an indicator of human intelligence but it may mean that our brains today are wired up differently, and perhaps more efficiently, than those of our ancestors."
]
incheon1_problem_2022_08_label = [1, 1, 0, 1, 1]
incheon1_p11_list = predict_problem_with_models(model_list, incheon1_problem_2022_08)
incheon1_p11_sorted = incheon1_p11_list.loc[map(lambda x:x[0], test_auroc)]
print("incheon1_problem_2022_08: ", incheon1_p11_sorted)
incheon1_p11_answer = incheon1_p11_sorted.iloc[[0], [0]].values[0][0]

# 고2
# -----2021----- #
incheon2_problem_2021_09 = [
    "Organisms living in the deep sea have adapted to the high pressure by storing water in their bodies, some consisting almost entirely of water.",
    "They are cold blooded organisms that adjust their body temperature to their environment, allowing them to survive in the cold water while maintaining a low metabolism.",
    "Many species lower their metabolism so much that they are able to survive without food for long periods of time, as finding the sparse food that is available expends a lot of energy.",
    "Many predatory fish of the deep sea are equipped with enormous mouths and sharp teeth, enabling them to hold on to prey and overpower it.",
    "Some predators hunting in the residual light zone of the ocean has excellent visual capabilities, while others are able to create their own light to attract prey or a mating partner."
]
incheon2_problem_2021_09_label = [1, 1, 0, 1, 1]
incheon2_p12_list = predict_problem_with_models(model_list, incheon2_problem_2021_09)
incheon2_p12_sorted = incheon2_p12_list.loc[map(lambda x:x[0], test_auroc)]
print("incheon2_problem_2021_09: ", incheon2_p12_sorted)
incheon2_p12_answer = incheon2_p12_sorted.iloc[[0], [0]].values[0][0]
# -----2022----- #
incheon2_problem_2022_08 = [ 
    "By noticing the relation between their own actions and resultant external changes, infants develop self efficacy, a sense that they are agents of the perceived changes.",
    "Although infants can notice the effect of their behavior on the physical environment, it is in early social interactions that infants most readily perceive the consequence of their actions.",
    "People have perceptual characteristics that virtually assure that infants will orient toward them.",
    "In addition, people engage with infants by exaggerating their facial expressions and inflecting their voices in ways that infants find fascinated.",
    "Consequentially, early social interactions provide a context where infants can easily notice the effect of their behavior."
]
incheon2_problem_2022_08_label = [1, 1, 1, 0, 1]
incheon2_p13_list = predict_problem_with_models(model_list, incheon2_problem_2022_08)
incheon2_p13_sorted = incheon2_p13_list.loc[map(lambda x:x[0], test_auroc)]
print("incheon2_problem_2022_08: ", incheon2_p13_sorted)
incheon2_p13_answer = incheon2_p13_sorted.iloc[[0], [0]].values[0][0]

# 고3
# -----2021----- #
incheon3_problem_2021_07 = [
    "The idea that people selectively expose themselves to news content has been around for a long time, but it is even more important today with the fragmentation of audiences and the proliferation of choices.",
    "Selective exposure is a psychological concept that says people seek out information that conforms to their existing belief systems and avoid information that challenges those beliefs.",
    "In the past when there were few sources of news, people could either expose themselves to mainstream news where they would likely see beliefs expressed counter to their ownor they could avoid news altogether.",
    "Now with so many types of news constantly available to a full range of niche audiences, people can easily find a source of news that consistently confirms their own personal set of beliefs.",
    "This leads to the possibility of creating many different small groups of people with each strongly believes they are correct and everyone else is wrong about how the world works."
]
incheon3_problem_2021_07_label = [1, 1, 1, 1, 0]
incheon3_p14_list = predict_problem_with_models(model_list, incheon3_problem_2021_07)
incheon3_p14_sorted = incheon3_p14_list.loc[map(lambda x:x[0], test_auroc)]
print("incheon3_problem_2021_07: ", incheon3_p14_sorted)
incheon3_p14_answer = incheon3_p14_sorted.iloc[[0], [0]].values[0][0]
# -----2022----- #
incheon3_problem_2022_07 = [
    "It helps the researcher to represent their data in a chart that shows the relative size of a response on one scale for interrelated variables.",
    "The spider chart is drawn with the variables spanning the chart, creating a spider web.",
    "An example of this is seen in a research study looking at self reported confidence in year 7 students across a range of subjects have taught in their first term in secondary school.",
    "The researcher takes the responses from a sample group and calculates the mean to plot on the spider chart.",
    "The chart, like the pie chart, can then be broken down for different groups of students within the study to elicit further analysis of findings."
]
incheon3_problem_2022_07_label = [1, 1, 1, 0, 1]
incheon3_p15_list = predict_problem_with_models(model_list, incheon3_problem_2022_07)
incheon3_p15_sorted = incheon3_p15_list.loc[map(lambda x:x[0], test_auroc)]
print("incheon3_problem_2022_07: ", incheon3_p15_sorted)
incheon3_p15_answer = incheon3_p15_sorted.iloc[[0], [0]].values[0][0]


print("\n인천 2021~2022 기출 실제 정답: ", "2 4 3 4 5 4")
print("인천 2021~2022 기출 예측 정답: ", incheon1_p10_answer, incheon1_p11_answer, 
                                          incheon2_p12_answer, incheon2_p13_answer,
                                          incheon3_p14_answer, incheon3_p15_answer)


