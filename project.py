import pandas as pd
import numpy as np
from os.path import join
import math
from fuzzywuzzy import fuzz
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

# 1. read data

ltable = pd.read_csv(join('data', "ltable.csv"))
rtable = pd.read_csv(join('data', "rtable.csv"))
train = pd.read_csv(join('data', "train.csv"))


# 2. blocking
def pairs2LR(ltable, rtable, candset):
    ltable.index = ltable.id
    rtable.index = rtable.id
    pairs = np.array(candset)
    tpls_l = ltable.loc[pairs[:, 0], :]
    tpls_r = rtable.loc[pairs[:, 1], :]
    tpls_l.columns = [col + "_l" for col in tpls_l.columns]
    tpls_r.columns = [col + "_r" for col in tpls_r.columns]
    tpls_l.reset_index(inplace=True, drop=True)
    tpls_r.reset_index(inplace=True, drop=True)
    LR = pd.concat([tpls_l, tpls_r], axis=1)
    return LR


def block_by_brand(ltable, rtable):
    # ensure brand is str
    ltable['brand'] = ltable['brand'].astype(str)
    rtable['brand'] = rtable['brand'].astype(str)

    # get all brands
    brands_l = set(ltable["brand"].values)
    brands_r = set(rtable["brand"].values)
    brands = brands_l.union(brands_r)

    brands = list(brands)
    for i in range(len(brands)):
        tempb = ""
        for character in brands[i]:
            if character.isdigit() or character.isalpha():
                tempb += character.lower()
        brands[i] = tempb
    brands = set(brands)

    # map each brand to left ids and right ids
    brand2ids_l = {b: [] for b in brands}
    brand2ids_r = {b: [] for b in brands}

    for i, x in ltable.iterrows():
        if x["brand"] != "nan":
            bindex = ""
            for c in x["brand"]:
                if c.isdigit() or c.isalpha():
                    bindex += c.lower()
            brand2ids_l[bindex].append(x["id"])
        else:
            for bd in brands:
                if bd in x["title"] and x["id"] not in brand2ids_l[bd]:
                    brand2ids_l[bd].append(x["id"])
    for i, x in rtable.iterrows():
        if x["brand"] != "nan":
            bindex = ""
            for c in x["brand"]:
                if c.isdigit() or c.isalpha():
                    bindex += c.lower()
            brand2ids_r[bindex].append(x["id"])
        else:
            for bd in brands:
                if bd in x["title"] and x["id"] not in brand2ids_r[bd]:
                    brand2ids_r[bd].append(x["id"])

    # put id pairs that share the same brand in candidate set
    candset = []
    for brd in brands:
        l_ids = brand2ids_l[brd]
        r_ids = brand2ids_r[brd]
        for i in range(len(l_ids)):
            for j in range(len(r_ids)):
                candset.append([l_ids[i], r_ids[j]])
    return candset

# blocking to reduce the number of pairs to be compared
candset = block_by_brand(ltable, rtable)
print("number of pairs originally", ltable.shape[0] * rtable.shape[0])
print("number of pairs after blocking",len(candset))
candset_df = pairs2LR(ltable, rtable, candset)

# 3. Feature engineering
import Levenshtein as lev

def jaccard_similarity(row, attr):
    x = set(row[attr + "_l"].lower().split())
    y = set(row[attr + "_r"].lower().split())
    return len(x.intersection(y)) / max(len(x), len(y))

def levenshtein_distance(row, attr):
    x = row[attr + "_l"].lower()
    y = row[attr + "_r"].lower()
    return lev.distance(x, y)

def price_difference(row):
    price_l = float(row["price_l"])
    price_r = float(row["price_r"])
    if math.isnan(price_l) == False and math.isnan(price_r) == False:
        price_diff = abs(price_l - price_r)
        return price_diff
    else:
        return 0

def feature_engineering(LR):
    LR = LR.astype(str)
    attrs = ["title", "category", "modelno"]
    features = []
    for attr in attrs:
        j_sim = LR.apply(jaccard_similarity, attr=attr, axis=1)
        l_dist = LR.apply(levenshtein_distance, attr=attr, axis=1)
        features.append(j_sim)
        features.append(l_dist)
    price_diff = LR.apply(price_difference, axis=1)
    features.append(price_diff)
    features = np.array(features).T
    return features
candset_features = feature_engineering(candset_df)

# also perform feature engineering to the training set
training_pairs = list(map(tuple, train[["ltable_id", "rtable_id"]].values))
training_df = pairs2LR(ltable, rtable, training_pairs)
training_features = feature_engineering(training_df)
training_label = train.label.values

# 4. Model training and prediction
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight="balanced_subsample",n_jobs=-1)
rf.fit(training_features, training_label)
y_pred = rf.predict(candset_features)

# 5. output

matching_pairs = candset_df.loc[y_pred == 1, ["id_l", "id_r"]]
matching_pairs = list(map(tuple, matching_pairs.values))

matching_pairs_in_training = training_df.loc[training_label == 1, ["id_l", "id_r"]]
matching_pairs_in_training = set(list(map(tuple, matching_pairs_in_training.values)))

pred_pairs = [pair for pair in matching_pairs if
              pair not in matching_pairs_in_training]  # remove the matching pairs already in training
pred_pairs = np.array(pred_pairs)
pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"])
pred_df.to_csv("output1.csv", index=False)
