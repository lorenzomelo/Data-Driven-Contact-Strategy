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

#DROPS RICK CHURNS AND LEAVING
dataset4 = dataset.drop(dataset[dataset["CLC_STATUS"] == "4-Risk churn"].index)
dataset4 = dataset4.drop(dataset4[dataset4["CLC_STATUS"] == "5-Leaving"].index)
dataset4["CLC_STATUS"].value_counts()
len(dataset4)



#DATASET DIVISION
'''
DATASET SOLUTION
drop the people with already a solution --> "SOLUTIONS" == 1
'''

solution_dataset = dataset4.drop(dataset4[dataset4["SOLUTIONS"] == 1].index)
set(solution_dataset["SOLUTIONS"])
len(solution_dataset)

#solution_dataset.to_csv("SOLUTION_DATASET.csv")

#DATASET CROSS_SELLING                                 
cross_selling_dataset = dataset4.drop(dataset4[dataset4["COMMODITY"] == "DUAL"].index)
set(cross_selling_dataset["COMMODITY"])
len(cross_selling_dataset)
#cross_selling_dataset.to_csv("CROSS_SELLING_DATASET.csv")

                                 
#### RISK CHURN CORRELATIONS
#drop the variables not useful for the correlation matrix
corr_dataset = dataset3.drop(['GENRE', 'ID', 'DATE_LAST_VISIT_DESK','DATE_LAST_REQUEST_CC','FIRST_ACTIVATION_DATE','DATE_LAST_CAMPAIGN','SUPPLY_START_DATE','YEAR_BIRTH','ZONE','AREA'], axis = 1)
#get dummies of categorical variables
df_dummies = pd.get_dummies(corr_dataset, columns=['CONSENSUS_PRIVACY','COMMODITY','CUSTOMER_SENIORITY','PHONE_VALIDATED','BEHAVIOUR_SCORE','CLC_STATUS','ACQUISITION_CHANNEL','LAST_GAS_PRODUCT','LAST_POWER_PRODUCT', 'LAST_CAMPAIGN_TIPOLOGY'])
#drop the other values of the variable CLC status
riskchurn_corr_df = df_dummies.drop(['CLC_STATUS_3-Customer Loyalty','CLC_STATUS_1-New','CLC_STATUS_2-Customer','CLC_STATUS_5-Leaving'],axis=1)
#correlation matrix
corrmatr = riskchurn_corr_df.corr()
#sort by correlation index to see most correlated variables
riskchurn_corr = corrmatr['CLC_STATUS_4-Risk churn']
riskchurn_corr_sorted = riskchurn_corr.sort_values(ascending=False)

                                 
                                 
                                 
                                 
exp_dataset4 = dataset4.filter(['LAST_MONTH_DESK_VISITS', 'LAST_3MONTHS_DESK_VISITS',
       'LAST_YEAR_DESK_VISITS', 'LAST_MONTH_CC_REQUESTS',
       'LAST_3MONTHS_CC_REQUESTS', 'LAST_YEAR_CC_REQUESTS','N_GAS_POINTS', 'N_POWER_POINTS', 'N_DISUSED_GAS_POINTS',
       'N_DISUSED_POWER_POINTS', 'N_TERMINATED_GAS_PER_SWITCH',
       'N_TERMINATED_POWER_PER_SWITCH', 'N_TERMINATED_GAS_PER_VOLTURA',
       'N_TERMINATED_POWER_PER_VOLTURA', 'INBOUND_CONTACTS_LAST_MONTH',
       'INBOUND_CONTACTS_LAST_2MONTHS', 'INBOUND_CONTACTS_LAST_YEAR', 'N_RISK_CASES_CHURN_GAS', 'N_RISK_CASES_CHURN_POWER',
       'N_MISSED_PAYMENTS', 'N_SWITCH_ANTI_CHURN',  'N_CAMPAIGN_SENT', 'N_CAMPAIGN_CLICKED', 'N_CAMPAIGN_OPENED',
       'N_DEM_CARING', 'N_SMS_CARING', 'N_TLS_CARING', 'N_DEM_RENEWAL',
       'N_SMS_RENEWAL', 'N_TLS_RENEWAL', 'N_DEM_CROSS_SELLING',
       'N_SMS_CROSS_SELLING', 'N_TLS_CROSS_SELLING', 'N_DEM_SOLUTION',
       'N_SMS_SOLUTION', 'N_TLS_SOLUTION','AVG_CONSUMPTION_GAS_M3',
       'AVG_CONSUMPTION_POWER_KWH' ], axis = 1)

for column in exp_dataset4.columns:
    Q1=exp_dataset4[column].quantile(0.05)
    Q3=exp_dataset4[column].quantile(0.95)
    Q2=exp_dataset4[column].quantile(0.5)
    IQR=Q3-Q1
    exp_dataset4[column] = np.where(exp_dataset4[column] < Q1, Q2, exp_dataset4[column])
    exp_dataset4[column] = np.where(exp_dataset4[column] > Q3, Q2, exp_dataset4[column])
    if column == 'AVG_CONSUMPTION_POWER_KWH':
        Q3=exp_dataset4[column].quantile(0.75)
        exp_dataset4[column] = np.where(exp_dataset4[column] > Q3, Q2, exp_dataset4[column])


descr = exp_dataset4.describe().T
exp_dataset4.boxplot(vert=False)

dataset4 = dataset4.drop(exp_dataset4.columns, axis = 1)
dataset4 = pd.concat([dataset4.reset_index(drop=True),exp_dataset4.reset_index(drop = True)], axis=1)

descr2 = dataset4.describe().T


### Analizziamo cosa caratterizza i "DUAL" e i "SOLUTIONS_1" guardando le medie 

## Alcune variabili che torneranno utili più tardi.
# Facciamo una distinzione tra AVG_CONSUPTION e le altre variabili per via della "scala" dei valori 
dataset5 = dataset4.drop(['GENRE', 'ID', 'CONSENSUS_PRIVACY', 'DATE_LAST_VISIT_DESK','DATE_LAST_REQUEST_CC', 'ZONE', 'AREA', 'FIRST_ACTIVATION_DATE', 'DATE_LAST_CAMPAIGN', 'SUPPLY_START_DATE', 'EMAIL_VALIDATED', 'PHONE_VALIDATED', 'YEAR_BIRTH', "AVG_CONSUMPTION_GAS_M3", "AVG_CONSUMPTION_POWER_KWH"], axis = 1)
dummy_df = pd.get_dummies(dataset5)

avg_df = dataset4.filter(["AVG_CONSUMPTION_GAS_M3", "AVG_CONSUMPTION_POWER_KWH", "SOLUTIONS"], axis=1)
dummy_avg_df = pd.get_dummies(avg_df)

avg_df_2 = dataset4.filter(["AVG_CONSUMPTION_GAS_M3", "AVG_CONSUMPTION_POWER_KWH", "COMMODITY"], axis=1)
dummy_avg_df_2 = pd.get_dummies(avg_df_2)
dummy_avg_df_2 = dummy_avg_df_2.drop(["COMMODITY_GAS", "COMMODITY_POWER"], axis=1)


## SOLUTIONS
df_mean = dummy_df.groupby('SOLUTIONS').mean()
df_mean.reset_index(drop=True, inplace=True)
columns = list(df_mean.columns)
df_mean = df_mean.T
df_mean["variables"] = columns
df_mean["difference"] = abs(df_mean[0] - df_mean[1])
sort_df = df_mean.sort_values(by=['difference'], ascending=False)
difference_dataset = sort_df.head(20)
difference_dataset.plot(x="variables", y=[0, 1], kind="barh")

avg_df_mean = dummy_avg_df.groupby('SOLUTIONS').mean()
avg_df_mean.reset_index(drop=True, inplace=True)
columns2 = list(avg_df_mean.columns)
avg_df_mean = avg_df_mean.T
avg_df_mean["variables"] = columns2
avg_df_mean["difference"] = abs(avg_df_mean[0] - avg_df_mean[1])
avg_df_mean.plot(x="variables", y=[0, 1], kind="barh")


## DUAL
df_mean_2 = dummy_df.groupby('COMMODITY_DUAL').mean()
df_mean_2.reset_index(drop=True, inplace=True)
columns = list(df_mean_2.columns)
df_mean_2 = df_mean_2.T
df_mean_2["variables"] = columns
df_mean_2["difference"] = abs(df_mean_2[0] - df_mean_2[1])
sort_df_2 = df_mean_2.sort_values(by=['difference'], ascending=False)
difference_dataset_2 = sort_df_2.head(20)
difference_dataset_2.plot(x="variables", y=[0, 1], kind="barh", color = ["green", "red"])

avg_df_mean_2 = dummy_avg_df_2.groupby('COMMODITY_DUAL').mean()
avg_df_mean_2.reset_index(drop=True, inplace=True)
columns2 = list(avg_df_mean_2.columns)
avg_df_mean_2 = avg_df_mean_2.T
avg_df_mean_2["variables"] = columns2
avg_df_mean_2["difference"] = abs(avg_df_mean_2[0] - avg_df_mean_2[1])
avg_df_mean_2.plot(x="variables", y=[0, 1], kind="barh", color = ["green", "red"])
