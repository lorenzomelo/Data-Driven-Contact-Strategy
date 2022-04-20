import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("DLT_AI_and_DATA_CUSTOMER_BASE_EN (1).csv", sep=";", encoding = 'unicode_escape')
dataset = pd.DataFrame(dataset)

##### Deleting risk churn and leaving
dataset4 = dataset.drop(dataset[dataset["CLC_STATUS"] == "4-Risk churn"].index)
dataset4 = dataset4.drop(dataset4[dataset4["CLC_STATUS"] == "5-Leaving"].index)



cluster_df  = dataset4.filter(["AVG_CONSUMPTION_GAS_M3", "AVG_CONSUMPTION_POWER_KWH", "CLC_STATUS", 'CUSTOMER_SENIORITY', 'BEHAVIOUR_SCORE', ], axis=1)
cluster_df = cluster_df.dropna()
from sklearn.preprocessing import OrdinalEncoder
ordinal = OrdinalEncoder()
ohe_df = pd.DataFrame(ordinal.fit_transform(cluster_df), columns = cluster_df.columns)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
ohe_df = pd.DataFrame(sc.fit_transform(ohe_df), columns = cluster_df.columns)
