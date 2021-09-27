import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def recommend_product(item1, item2, item3):
    df = pd.read_csv("dataset_group.csv", header=None)
    df.columns = ["Date", "ID", "Items"]

    df.drop('Date', inplace=True, axis=1)

    df1 = df.groupby('ID')['Items'].apply(','.join).reset_index()

    transac = []
    for i in range(0, len(df1)):
        transac.append([str(df1.values[i, j]) for j in range(0, 2) if str(df1.values[i, j]) != '0'])

    itemArray = df['Items'].unique()

    df = df.groupby('ID')['Items'].apply(','.join).reset_index()

    for i in range(0, len(itemArray)):
        df.insert(len(df.columns), itemArray[i], "")
        for index, row in df.iterrows():
            df.at[index, itemArray[i]] = 1 if len(
                [1 for item in transac[index][1].split(",") if item in itemArray[i]]) > 0 else 0

    df.drop('Items', inplace=True, axis=1)

    frequent_itemsets = apriori(df.drop(['ID'], axis=1), min_support=0.17, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

    consequent_list = rules.consequents[rules.antecedents.apply(str).str.contains(item1)]
    cons_list = []
    for i in range(0, len(consequent_list)):
        c_list = set(list(consequent_list)[i])
        cons_list.append(c_list)

    return cons_list
