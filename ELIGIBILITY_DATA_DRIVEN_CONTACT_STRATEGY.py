import pandas as pd
import numpy as np

dataset = pd.read_csv("DLT_AI_and_DATA_CUSTOMER_BASE_EN_PROVA.csv", sep=";")
dataset = pd.DataFrame(dataset)

set(dataset["COMMODITY"])
set(dataset["SOLUTIONS"])
set(dataset["CONSENSUS_PRIVACY"])

#GENERAL CLEANING
'''
togliamo i "nan" e "NO" dal dal dataset["CONSENSUS_PRIVACY"]
togliere risk of being churned or outbound
CLC_STATUS = clc --> customer life cycle
'''
dataset["CONSENSUS_PRIVACY"].isnull().sum()
dataset2 = dataset[dataset["CONSENSUS_PRIVACY"].notnull()]
dataset2["CONSENSUS_PRIVACY"].isnull().sum()

consensus_privacy_yes = dataset2["CONSENSUS_PRIVACY"].isin(["YES"])
dataset3 = dataset2[consensus_privacy_yes]
set(dataset3["CONSENSUS_PRIVACY"])
len(dataset3)

dataset2["CONSENSUS_PRIVACY"].value_counts()
dataset3["SOLUTIONS"].value_counts()
set(dataset3["CLC_STATUS"])
dataset3["CLC_STATUS"].value_counts()

dataset4 = dataset3.drop(dataset3[dataset3["CLC_STATUS"] == "4-Risk churn"].index)
dataset4 = dataset4.drop(dataset4[dataset4["CLC_STATUS"] == "5-Leaving"].index)
dataset4["CLC_STATUS"].value_counts()

#DATASET DIVISION
'''
DATASET SOLUTION
drop the people with already a solution --> "SOLUTIONS" == 1
'''

solution_dataset = dataset3.drop(dataset3[dataset3["SOLUTIONS"] == 1].index
set(solution_dataset["SOLUTIONS"])
len(solution_dataset)

#solution_dataset.to_csv("SOLUTION_DATASET.csv")
cross_selling_dataset = dataset3.drop(dataset3[dataset3["COMMODITY"] == "DUAL"].index)
set(cross_selling_dataset["COMMODITY"])
len(cross_selling_dataset)
#cross_selling_dataset.to_csv("CROSS_SELLING_DATASET.csv")
