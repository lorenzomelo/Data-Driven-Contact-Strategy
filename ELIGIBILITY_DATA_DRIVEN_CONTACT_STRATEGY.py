import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("DLT_AI_and_DATA_CUSTOMER_BASE_EN.csv", sep=";", encoding = 'unicode_escape')
dataset = pd.DataFrame(dataset)

###### EXPLORATORY DATA ANALYSIS #######

#### SOLUTIONS ####
'''
#COMMODITY
crosstab_commodity_solutions = pd.crosstab(index=dataset4["SOLUTIONS"],
                                    columns=dataset4["COMMODITY"], normalize='index')
crosstab_commodity_solutions.plot.bar(figsize=(6, 4),
                               rot=0).set(ylabel="Count", xlabel = "Solution")
plt.title('Solutions by commodity type', fontweight="bold", fontsize =10)
plt.show()
#PRIVACY
crosstab_privacy_solutions = pd.crosstab(index=dataset4["SOLUTIONS"],
                                    columns=dataset4["CONSENSUS_PRIVACY"], normalize='index')
crosstab_privacy_solutions.plot.bar(figsize=(6, 4),
                               rot=0, color=('red', 'green')).set(ylabel="Percentage", xlabel = "Solution")
plt.title('Solutions by Privacy Consensus', fontweight="bold", fontsize =10)
plt.show()
#CUSTOMER SENIORITY
crosstab_custsen_solutions = pd.crosstab(index=dataset4["SOLUTIONS"],
                                    columns=dataset4["CUSTOMER_SENIORITY"], normalize='index')
crosstab_custsen_solutions.plot.bar(figsize=(6, 4),
                               rot=0, color=('hotpink', 'deeppink', 'mediumvioletred')).set(ylabel="Percentage", xlabel = "Solution")
plt.title('Solutions by Customer Seniority', fontweight="bold", fontsize =10)
plt.show()
#GENDER
crosstab_genre_solutions = pd.crosstab(index=dataset4["SOLUTIONS"],
                                    columns=dataset4["GENRE"], normalize='index')
crosstab_genre_solutions.plot.bar(figsize=(6, 4),
                               rot=0, color=('lightcoral', 'gold')).set(ylabel="Percentage", xlabel = "Solution")
plt.title('Solutions by Genre', fontweight="bold", fontsize =10)
plt.show()

#LOYALITY PROGRAM
crosstab_loyalityprog_solutions = pd.crosstab(index=dataset4["SOLUTIONS"],
                                    columns=dataset4["LOYALTY_PROGRAM"], normalize='index')
crosstab_loyalityprog_solutions.plot.bar(figsize=(6, 4),
                               rot=0, color=('darkorchid', 'royalblue')).set(ylabel="Percentage", xlabel = "Solution")
plt.title('Solutions by Loyalty Program', fontweight="bold", fontsize =10)
plt.show()

#AREA
crosstab_area_solutions = pd.crosstab(index=dataset4["SOLUTIONS"],
                                    columns=dataset4["AREA"], normalize='index')
crosstab_area_solutions.plot.bar(figsize=(6, 4),
                               rot=0, color=('limegreen', 'lightcoral', 'deepskyblue', 'gold')).set(ylabel="Percentage", xlabel = "Solution")
plt.title('Solutions by Area of Italy', fontweight="bold", fontsize =10)
plt.show()

#FLAG_BAD_CUSTOMER
crosstab_flag_solutions = pd.crosstab(index=dataset4["SOLUTIONS"],
                                    columns=dataset4["FLAG_BAD_CUSTOMER"], normalize='index')
crosstab_flag_solutions.plot.bar(figsize=(6, 4),
                               rot=0, color=('yellowgreen', 'crimson')).set(ylabel="Percentage", xlabel = "Solution")
plt.title('Solutions by Bad Customer Flag', fontweight="bold", fontsize =10)
plt.show()

#AVG CONSUMPTION GAS and AVG CONSUMPTION POWER
bp = sns.boxplot(data = dataset, x = "SOLUTIONS", y= "AVG_CONSUMPTION_GAS_M3", hue="SOLUTIONS")  # RUN PLOT
bp.set_ylim([0, 200000])
plt.show()

#INBOUND CONTACTS LAST YEAR
hp = sns.histplot(data=dataset, x="INBOUND_CONTACTS_LAST_YEAR")
hp.set_xlim([0, 20])
plt.show()
#N MISSED PAYMENTS

#N CAMPAIGN SENT

#N DEM, N SMS, N TLS SOLUTION

#BEHAVIOUR SCORE

#CLC STATUS

#ACQUISITION CHANNEL


'''

#### COMMODITY ####





##### Deleting risk churn and leaving
dataset4 = dataset.drop(dataset[dataset["CLC_STATUS"] == "4-Risk churn"].index)
dataset4 = dataset4.drop(dataset4[dataset4["CLC_STATUS"] == "5-Leaving"].index)



##### OUTLIER DETECTION ######
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



#### dataset for dummies
df_for_dummies = dataset4.drop(['ID', 'CONSENSUS_PRIVACY', 'DATE_LAST_VISIT_DESK','DATE_LAST_REQUEST_CC',
                          'FIRST_ACTIVATION_DATE', 'DATE_LAST_CAMPAIGN', 'SUPPLY_START_DATE', 'EMAIL_VALIDATED',
                          'PHONE_VALIDATED', 'YEAR_BIRTH', "AVG_CONSUMPTION_GAS_M3", "AVG_CONSUMPTION_POWER_KWH"], axis = 1)
dummy_df = pd.get_dummies(df_for_dummies)




##### MEAN ANALYSIS #####

### Analizziamo cosa caratterizza i "DUAL" e i "SOLUTIONS_1" guardando le medie

## Alcune variabili che torneranno utili più tardi.
# Facciamo una distinzione tra AVG_CONSUPTION e le altre variabili per via della "scala" dei valori
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
