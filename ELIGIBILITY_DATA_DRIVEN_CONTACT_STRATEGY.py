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
                          'PHONE_VALIDATED', 'YEAR_BIRTH', 
                          'LAST_POWER_PRODUCT', 'LAST_MONTH_DESK_VISITS',
                          'LAST_3MONTHS_DESK_VISITS', 'LAST_YEAR_DESK_VISITS',
                          'LAST_MONTH_CC_REQUESTS', 'LAST_3MONTHS_CC_REQUESTS',
                          'LAST_YEAR_CC_REQUESTS', 'INBOUND_CONTACTS_LAST_MONTH', 'INBOUND_CONTACTS_LAST_2MONTHS',
                          'INBOUND_CONTACTS_LAST_YEAR','N_CAMPAIGN_SENT', 'N_CAMPAIGN_CLICKED', 'N_CAMPAIGN_OPENED',
                          'N_DEM_CARING', 'N_SMS_CARING', 'N_TLS_CARING', 'N_DEM_RENEWAL',
                          'N_SMS_RENEWAL', 'N_TLS_RENEWAL'], axis = 1)


dummy_df = pd.get_dummies(df_for_dummies)


# UNDERSAMPLING 
class_2,class_1 = dummy_df.SOLUTIONS.value_counts()
c2 = dummy_df[dummy_df['SOLUTIONS'] == 0]
c1 = dummy_df[dummy_df['SOLUTIONS'] == 1]
df_2 = c2.sample(class_1)
under_sol = pd.concat([df_2,c1],axis=0)
under_sol.SOLUTIONS.value_counts()
under_sol.to_csv("Logistic_solution.csv")

class_2,class_1 = dummy_df.COMMODITY_DUAL.value_counts()
c2 = dummy_df[dummy_df['COMMODITY_DUAL'] == 0]
c1 = dummy_df[dummy_df['COMMODITY_DUAL'] == 1]
df_2 = c2.sample(class_1)
under_dual = pd.concat([df_2,c1],axis=0)
under_dual.COMMODITY_DUAL.value_counts()
under_dual.to_csv("Logistic_dual.csv")




### CORRELATIONS 
import seaborn as sn

def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr(method = "spearman").abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
var_corr = get_top_abs_correlations(under_dual, 50)
var_corr_2 = get_top_abs_correlations(under_sol, 50)

sol_corr = under_sol[under_sol.columns].corr(method = "spearman")['SOLUTIONS'][:]
dual_corr = under_dual[under_dual.columns].corr(method = "spearman")['COMMODITY_DUAL'][:]
sol_corr = pd.DataFrame(sol_corr).sort_values(by = 'SOLUTIONS', ascending=False)
dual_corr = pd.DataFrame(dual_corr).sort_values(by = 'COMMODITY_DUAL', ascending=False)




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
df_mean = dummy_df.drop(["AVG_CONSUMPTION_GAS_M3", "AVG_CONSUMPTION_POWER_KWH", 'N_DEM_CROSS_SELLING',
                          'N_SMS_CROSS_SELLING', 'N_TLS_CROSS_SELLING', 'N_DEM_SOLUTION',
                          'N_SMS_SOLUTION', 'N_TLS_SOLUTION'], axis = 1)
df_mean = df_mean.groupby('SOLUTIONS').mean()
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
df_mean_2 = dummy_df.drop(["AVG_CONSUMPTION_GAS_M3", "AVG_CONSUMPTION_POWER_KWH", 'N_DEM_CROSS_SELLING',
                          'N_SMS_CROSS_SELLING', 'N_TLS_CROSS_SELLING', 'N_DEM_SOLUTION',
                          'N_SMS_SOLUTION', 'N_TLS_SOLUTION'], axis = 1)
df_mean_2 = df_mean_2.groupby('COMMODITY_DUAL').mean()
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



#FEATURE SELECTION
#RANDOM FOREST

# DUAL
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score


# df_random_dual = pd.read_csv("project_deloitte/Logistic_dual.csv")
under_dual = under_dual.drop(['N_DEM_CROSS_SELLING', 'N_SMS_CROSS_SELLING',
                              'N_TLS_CROSS_SELLING', 'N_DEM_SOLUTION',
                              'N_SMS_SOLUTION', 'N_TLS_SOLUTION'], axis=1)
X = under_dual.iloc[:, under_dual.columns != "COMMODITY_DUAL"].values
y = under_dual["COMMODITY_DUAL"]

#train test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Scale

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# GRID SEARCH / OPTIMIZATION
rf_final = RandomForestClassifier(random_state=0)
param_grid = {
    'n_estimators': [100, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rf_final, param_grid=param_grid, cv=5)
CV_rfc.fit(X_train, y_train)
print(CV_rfc.best_params_)

#MODEL
rfc1=RandomForestClassifier(random_state=0, max_features='auto', n_estimators= 500, max_depth=7, criterion='gini')
rfc1.fit(X_train, y_train)
y_pred = rfc1.predict(X_test)

# Evaluation metrics
#CONFUSION MATRIX
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.title("Random Forest Confusion Matrix", fontweight="bold", fontsize=12)
plt.show()

#ACCURACY
accuracy_train = rfc1.score(X_train, y_train)
print("Random Forest - Accuracy on the training set: " + str(accuracy_train))
print("Random Forest - Accuracy on the test set: " + str(accuracy_score(y_test, y_pred)))
print("Random Forest - Precision: " + str(precision_score(y_test, y_pred)))
print("Random Forest - Recall: " + str(recall_score(y_test, y_pred)))

#VISUALIZATION OF THE FEATURE_IMPORTANCE
rfc1.feature_importances_
sorted_idx = rfc1.feature_importances_.argsort()
yaxis1 = pd.DataFrame(X, columns=['LOYALTY_PROGRAM', 'SOLUTIONS', 'NEW_CUSTOMER',
                                  'WEB_PORTAL_REGISTRATION', 'FLAG_BAD_CUSTOMER', 'N_GAS_POINTS',
                                  'N_POWER_POINTS', 'N_DISUSED_GAS_POINTS', 'N_DISUSED_POWER_POINTS',
                                  'N_TERMINATED_GAS_PER_SWITCH', 'N_TERMINATED_POWER_PER_SWITCH',
                                  'N_TERMINATED_GAS_PER_VOLTURA', 'N_TERMINATED_POWER_PER_VOLTURA',
                                  'N_RISK_CASES_CHURN_GAS', 'N_RISK_CASES_CHURN_POWER',
                                  'N_MISSED_PAYMENTS', 'N_SWITCH_ANTI_CHURN', 'AVG_CONSUMPTION_GAS_M3',
                                  'AVG_CONSUMPTION_POWER_KWH', 'GENRE_F', 'GENRE_M',
                                  'COMMODITY_GAS', 'COMMODITY_POWER', 'ZONE_Abruzzo', 'ZONE_Basilicata',
                                  'ZONE_Calabria', 'ZONE_Campania', 'ZONE_Emilia-Romagna',
                                  'ZONE_Friuli-Venezia Giulia', 'ZONE_Lazio', 'ZONE_Liguria',
                                  'ZONE_Lombardia', 'ZONE_Marche', 'ZONE_Molise', 'ZONE_Piemonte',
                                  'ZONE_Puglia', 'ZONE_Sardegna', 'ZONE_Sicilia', 'ZONE_Toscana',
                                  'ZONE_Trentino-Alto Adige', 'ZONE_Umbria',
                                  "ZONE_Valle d'Aosta/VallÇ¸e d'Aoste", 'ZONE_Veneto', 'AREA_Center',
                                  'AREA_North-East', 'AREA_North-West', 'AREA_South',
                                  'CUSTOMER_SENIORITY_1-3 YEARS', 'CUSTOMER_SENIORITY_<1 YEAR',
                                  'CUSTOMER_SENIORITY_>3 YEARS', 'BEHAVIOUR_SCORE_BAD PAYER',
                                  'BEHAVIOUR_SCORE_GOOD PAYER', 'BEHAVIOUR_SCORE_LATECOMER',
                                  'CLC_STATUS_1-New', 'CLC_STATUS_2-Customer',
                                  'CLC_STATUS_3-Customer Loyalty', 'ACQUISITION_CHANNEL_Agency',
                                  'ACQUISITION_CHANNEL_CC', 'ACQUISITION_CHANNEL_Desk',
                                  'ACQUISITION_CHANNEL_Teleselling', 'ACQUISITION_CHANNEL_WEB',
                                  'LAST_GAS_PRODUCT_Digital', 'LAST_GAS_PRODUCT_Fidelityÿ',
                                  'LAST_GAS_PRODUCT_Green', 'LAST_GAS_PRODUCT_Traditional',
                                  'LAST_CAMPAIGN_TIPOLOGY_Caring', 'LAST_CAMPAIGN_TIPOLOGY_Communication',
                                  'LAST_CAMPAIGN_TIPOLOGY_Comunicazione',
                                  'LAST_CAMPAIGN_TIPOLOGY_Cross-Selling',
                                  'LAST_CAMPAIGN_TIPOLOGY_Renewal', 'LAST_CAMPAIGN_TIPOLOGY_Rinnovo',
                                  'LAST_CAMPAIGN_TIPOLOGY_Solution'])

plt.figure(figsize=(15, 15))
sns.barplot(x=rfc1.feature_importances_[sorted_idx], y=yaxis1.columns[sorted_idx], orient="h",
            palette="gist_rainbow")
plt.xlabel("Random Forest Feature Importance")
plt.title("Random Forest Features Importance DUAL", fontweight="bold", fontsize=12)
sns.set(font_scale=0.3)
plt.rcParams['figure.dpi'] = 300
plt.show()

#RANDOM
#SOLUTION

under_sol = under_sol.drop(['N_DEM_CROSS_SELLING', 'N_SMS_CROSS_SELLING',
                            'N_TLS_CROSS_SELLING', 'N_DEM_SOLUTION',
                            'N_SMS_SOLUTION', 'N_TLS_SOLUTION'], axis=1)
X_sol = under_sol.iloc[:, under_sol.columns != "SOLUTIONS"].values
y_sol = under_sol["SOLUTIONS"]

#split
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sol, y_sol, test_size=0.3, random_state=0)

#scale
sc = StandardScaler()
X_train_s = sc.fit_transform(X_train_s)
X_test_s = sc.transform(X_test_s)

#GRID SEARCH
rf_final_sol = RandomForestClassifier(0)
param_grid = {
    'n_estimators': [20, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}
CV_rfc_sol = GridSearchCV(estimator=rf_final_sol, param_grid=param_grid, cv=5)
CV_rfc_sol.fit(X_train_s, y_train_s)
print(CV_rfc_sol.best_params_)

#MODEL
rfc2=RandomForestClassifier(random_state=0, max_features='log2', n_estimators= 20, max_depth=6, criterion='entropy')
rfc2.fit(X_train_s, y_train_s)
y_pred_s = rfc2.predict(X_test_s)

# Evaluation metrics
mat = confusion_matrix(y_test_s, y_pred_s)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.title("Random Forest Confusion Matrix", fontweight="bold", fontsize=12)
plt.show()
#ACCURACY
accuracy_train = rfc2.score(X_train_s, y_train_s)
print("Random Forest - Accuracy on the training set: " + str(accuracy_train))
print("Random Forest - Accuracy on the test set: " + str(accuracy_score(y_test_s, y_pred_s)))
print("Random Forest - Precision: " + str(precision_score(y_test_s, y_pred_s)))
print("Random Forest - Recall: " + str(recall_score(y_test_s, y_pred_s)))

#VISUALIZATION OF FEATURE IMPORTANCE
rfc2.feature_importances_
sorted_idx = rfc2.feature_importances_.argsort()
yaxis1 = pd.DataFrame(X_sol, columns=['LOYALTY_PROGRAM', 'NEW_CUSTOMER',
                                      'WEB_PORTAL_REGISTRATION', 'FLAG_BAD_CUSTOMER', 'N_GAS_POINTS',
                                      'N_POWER_POINTS', 'N_DISUSED_GAS_POINTS', 'N_DISUSED_POWER_POINTS',
                                      'N_TERMINATED_GAS_PER_SWITCH', 'N_TERMINATED_POWER_PER_SWITCH',
                                      'N_TERMINATED_GAS_PER_VOLTURA', 'N_TERMINATED_POWER_PER_VOLTURA',
                                      'N_RISK_CASES_CHURN_GAS', 'N_RISK_CASES_CHURN_POWER',
                                      'N_MISSED_PAYMENTS', 'N_SWITCH_ANTI_CHURN', 'AVG_CONSUMPTION_GAS_M3',
                                      'AVG_CONSUMPTION_POWER_KWH', 'GENRE_F', 'GENRE_M', 'COMMODITY_DUAL',
                                      'COMMODITY_GAS', 'COMMODITY_POWER', 'ZONE_Abruzzo', 'ZONE_Basilicata',
                                      'ZONE_Calabria', 'ZONE_Campania', 'ZONE_Emilia-Romagna',
                                      'ZONE_Friuli-Venezia Giulia', 'ZONE_Lazio', 'ZONE_Liguria',
                                      'ZONE_Lombardia', 'ZONE_Marche', 'ZONE_Molise', 'ZONE_Piemonte',
                                      'ZONE_Puglia', 'ZONE_Sardegna', 'ZONE_Sicilia', 'ZONE_Toscana',
                                      'ZONE_Trentino-Alto Adige', 'ZONE_Umbria',
                                      "ZONE_Valle d'Aosta/VallÇ¸e d'Aoste", 'ZONE_Veneto', 'AREA_Center',
                                      'AREA_North-East', 'AREA_North-West', 'AREA_South',
                                      'CUSTOMER_SENIORITY_1-3 YEARS', 'CUSTOMER_SENIORITY_<1 YEAR',
                                      'CUSTOMER_SENIORITY_>3 YEARS', 'BEHAVIOUR_SCORE_BAD PAYER',
                                      'BEHAVIOUR_SCORE_GOOD PAYER', 'BEHAVIOUR_SCORE_LATECOMER',
                                      'CLC_STATUS_1-New', 'CLC_STATUS_2-Customer',
                                      'CLC_STATUS_3-Customer Loyalty', 'ACQUISITION_CHANNEL_Agency',
                                      'ACQUISITION_CHANNEL_CC', 'ACQUISITION_CHANNEL_Desk',
                                      'ACQUISITION_CHANNEL_Teleselling', 'ACQUISITION_CHANNEL_WEB',
                                      'LAST_GAS_PRODUCT_Digital', 'LAST_GAS_PRODUCT_Fidelityÿ',
                                      'LAST_GAS_PRODUCT_Green', 'LAST_GAS_PRODUCT_Traditional',
                                      'LAST_CAMPAIGN_TIPOLOGY_Caring', 'LAST_CAMPAIGN_TIPOLOGY_Communication',
                                      'LAST_CAMPAIGN_TIPOLOGY_Comunicazione',
                                      'LAST_CAMPAIGN_TIPOLOGY_Cross-Selling',
                                      'LAST_CAMPAIGN_TIPOLOGY_Renewal', 'LAST_CAMPAIGN_TIPOLOGY_Rinnovo',
                                      'LAST_CAMPAIGN_TIPOLOGY_Solution'])

plt.figure(figsize=(20, 20))
sns.barplot(x=rfc2.feature_importances_[sorted_idx], y=yaxis1.columns[sorted_idx], orient="h",
            palette="gist_rainbow")
plt.xlabel("Random Forest Feature Importance")
plt.title("Random Forest Features Importance SOLUTIONS", fontweight="bold", fontsize=12)
sns.set(font_scale=0.3)
plt.rcParams['figure.dpi'] = 300
plt.show()


#FEAUTURE SELECTED FOR RECCOMENDATION

recommend_sol = dummy_df.filter(['AVG_CONSUMPTION_GAS_M3', "COMMODITY_DUAL", 'ZONE_Piemonte', 'WEB_PORTAL_REGISTRATION', 
                                 'AREA_North-West', 'CLC_STATUS_3-Customer Loyalty', 'BEHAVIOUR_SCORE_GOOD PAYER', 'LOYALTY_PROGRAM', 'AREA_SOUTH', 'ZONE_VENETO', 
                                 'LAST_CAMPAIGN_TIPOLOGY_Caring', 'AREA_North-East',
                                 'LAST_CAMPAIGN_TIPOLOGY_Cross-Selling','CUSTOMER_SENIORITY_>3 YEARS', 'CUSTOMER_SENIORITY_<1 YEAR',
                                 'ACQUISITION_CHANNEL_CC', 'BEHAVIOUR_SCORE_BAD PAYER', "AREA_CENTER" ], axis=1)

recommend_dual = dummy_df.filter(['CLC_STATUS_3-Customer Loyalty', 'AREA_North-West',
                                 'WEB_PORTAL_REGISTRATION', 'LAST_CAMPAIGN_TIPOLOGY_Cross-Selling',
                                 'N_DISUSED_GAS_POINTS', 
                                 'LAST_CAMPAIGN_TIPOLOGY_Caring', 'SOLUTIONS', 'CUSTOMER_SENIORITY_>3 YEARS', 'LAST_CAMPAIGN_TIPOLOGY_Renewal', 
                                 'CUSTOMER_SENIORITY_1-3 YEARS', 'AVG_CONSUMPTION_GAS_M3' 
                                 'LAST_GAS_PRODUCT_Traditional', "COMMODITY_DUAL"], axis=1)


#ELIGIBILITY

column_names = ["ID", "Month_1", "Month_2", "Month_3", "Month_4", "Month_5", "Month_6",
                "Month_7", "Month_8", "Month_9", "Month_10", "Month_11", "Month_12"]
Cross_Selling_DEM = pd.DataFrame(columns=column_names)
Cross_Selling_SMS = pd.DataFrame(columns=column_names)
Cross_Selling_TLS = pd.DataFrame(columns=column_names)
Solution_DEM = pd.DataFrame(columns=column_names)
Solution_SMS = pd.DataFrame(columns=column_names)
Solution_TLS = pd.DataFrame(columns=column_names)


# CLEAN FOR CONSENSUS_PRIVACY, CLC_STATUS
def clean_data_for_eligibility(df):
    df = df.drop(df[df["CLC_STATUS"] == "4-Risk churn"].index)
    df2 = df.drop(df[df["CLC_STATUS"] == "5-Leaving"].index)
    df3 = df.drop(df[df["CONSENSUS_PRIVACY"] == "NO"].index)
    return df3


dataset_eligible = clean_data_for_eligibility(dataset)


# CLEAN PER MAIL E PHONE
def clean_phone(df):
    df = df.drop(df[(df["PHONE_VALIDATED"] == "KO") & (df["EMAIL_VALIDATED"] == 0)].index)
    return df

dataset_final_eligible = clean_phone(dataset_eligible)

# MERGE THE TWO
dataset_final_eligible['randNumCol'] = np.random.randint(0, 2, size=len(dataset_final_eligible))  # add column with random number to proxy propensity

### DIVIDING DATASET ACCORDING TO THE ELIGIBILITY
pd.options.mode.chained_assignment = None
timefmt = "%d/%m/%Y"
dataset_final_eligible['DATE_LAST_CAMPAIGN'] = pd.to_datetime(dataset_final_eligible['DATE_LAST_CAMPAIGN'], format = timefmt)

datatypes = dataset_final_eligible.dtypes

dataset_final_eligible['REFERENCE_DATE'] = "26/04/2022"
dataset_final_eligible['REFERENCE_DATE'] = pd.to_datetime(dataset_final_eligible['REFERENCE_DATE'], format = timefmt)

dataset_final_eligible['N_months'] = ((dataset_final_eligible.DATE_LAST_CAMPAIGN - dataset_final_eligible.REFERENCE_DATE)/np.timedelta64(1, 'M'))
dataset_final_eligible['N_months'] = dataset_final_eligible['N_months'].astype(int).abs()

cross_selling_tls_general = dataset_final_eligible[(dataset_final_eligible.N_months >= 6) & (dataset_final_eligible["PHONE_VALIDATED"] != "KO")]
cross_selling_dem_general = dataset_final_eligible[(dataset_final_eligible.N_months >= 2) & (dataset_final_eligible["EMAIL_VALIDATED"] != 0)] 
cross_selling_sms_general = dataset_final_eligible[(dataset_final_eligible.N_months >= 2) & (dataset_final_eligible["PHONE_VALIDATED"] != "KO")]
solution_tls_general = dataset_final_eligible[(dataset_final_eligible.N_months >= 12) & (dataset_final_eligible["PHONE_VALIDATED"] != "KO")]
solution_dem_general = dataset_final_eligible[(dataset_final_eligible.N_months >= 6) & (dataset_final_eligible["EMAIL_VALIDATED"] != 0)]
solution_sms_general = dataset_final_eligible[(dataset_final_eligible.N_months >= 6) & (dataset_final_eligible["PHONE_VALIDATED"] != "KO")]

#CROSS SELLING
for index, row in cross_selling_dem_general.iterrows():
    l = [row["ID"],1,0,1,0,1,0,1,0,1,0,1,0]
    Cross_Selling_DEM.loc[len(Cross_Selling_DEM)] = l
    
for index, row in cross_selling_sms_general.iterrows():
    l = [row["ID"],1,0,1,0,1,0,1,0,1,0,1,0]
    Cross_Selling_SMS.loc[len(Cross_Selling_SMS)] = l

for index, row in cross_selling_tls_general.iterrows():
    l = [row["ID"],0,0,0,0,0,1,0,0,0,0,0,1]
    Cross_Selling_TLS.loc[len(Solution_TLS)] = l

#SOLUTION
for index, row in solution_dem_general.iterrows():
    if (row["PHONE_VALIDATED"] == "KO") & (row["COMMODITY"] != "DUAL"):
        l = [row["ID"],0,1,0,1,0,0,0,0,0,0,0,0]
        Solution_DEM.loc[len(Solution_DEM)] = l
    elif (row["PHONE_VALIDATED"] == "KO") & (row["COMMODITY"] == "DUAL"):
        m = [row["ID"],1,0,1,0,0,0,0,0,0,0,0,0]
        Solution_DEM.loc[len(Solution_DEM)] = m
    elif (row["PHONE_VALIDATED"] != "KO") & (row["COMMODITY"] == "DUAL"):
        n = [row["ID"],0,0,1,0,1,0,0,0,0,0,0,0]
        Solution_DEM.loc[len(Solution_DEM)] = n
    elif (row["PHONE_VALIDATED"] != "KO") & (row["COMMODITY"] != "DUAL"):
        o = [row["ID"],0,0,0,1,0,0,0,1,0,0,0,0]
        Solution_DEM.loc[len(Solution_DEM)] = o

for index, row in solution_sms_general.iterrows():
    if row["COMMODITY"] == "DUAL":
        l = [row["ID"],0,0,1,0,1,0,0,0,0,0,0,0]
        Solution_SMS.loc[len(Solution_SMS)] = l
    elif row["COMMODITY"] != "DUAL":
        m = [row["ID"],0,0,0,1,0,0,0,1,0,0,0,0]
        Solution_SMS.loc[len(Solution_SMS)] = m

for index, row in solution_tls_general.iterrows():
    if row["COMMODITY"] == "DUAL":
        l = [row["ID"],1,0,0,0,0,0,0,0,0,0,0,0]
        Cross_Selling_TLS.loc[len(Solution_TLS)] = l
    elif row["COMMODITY"] != "DUAL":
        m = [row["ID"],0,1,0,0,0,0,0,0,0,0,0,0]
        Cross_Selling_TLS.loc[len(Solution_TLS)] = m

  
  
#PROPENSITY 2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

#dataset = pd.read_csv("dataset4.csv", sep=",", index_col=0)
#dummy_df = pd.read_csv("dummy_df.csv", sep=",", index_col=0)

different_cols = dummy_df.columns.difference(dataset4.columns)
dataset_difference = dummy_df[different_cols]

dataset_whole = pd.merge(dataset4, dataset_difference, left_index=True,
                     right_index=True, how='inner')

dataset_dual = dataset_whole.filter(['CLC_STATUS_3-Customer Loyalty', 'AREA_North-West','WEB_PORTAL_REGISTRATION', 'LAST_CAMPAIGN_TIPOLOGY_Cross-Selling',
                                     'N_DISUSED_GAS_POINTS', 
                                     'LAST_CAMPAIGN_TIPOLOGY_Caring', 'SOLUTIONS', 'CUSTOMER_SENIORITY_>3 YEARS', 'LAST_CAMPAIGN_TIPOLOGY_Renewal', 
                                     'CUSTOMER_SENIORITY_1-3 YEARS', 'AVG_CONSUMPTION_GAS_M3','LAST_GAS_PRODUCT_Traditional',
                                     "CONSENSUS_PRIVACY", "ID", "COMMODITY_DUAL", "PHONE_VALIDATED", "EMAIL_VALIDATED"], axis=1)  
def non_elegible(df):
    df = df.drop(df[(df["CONSENSUS_PRIVACY"] == "YES") & (df["COMMODITY_DUAL"] == 0)].index)
    return df
dataset_non_elegible = non_elegible(dataset_dual)
dataset_non_elegible = dataset_non_elegible.dropna(axis=0)

def elegible(df):
    df = df.drop(df[(df["CONSENSUS_PRIVACY"] == "NO") | (df["COMMODITY_DUAL"] == 1)].index)
    df = df.drop(df[(df["PHONE_VALIDATED"] == "KO") & (df["EMAIL_VALIDATED"] == 0)].index)
    return df
dataset_elegible = elegible(dataset_dual)
dataset_elegible = dataset_elegible.dropna(axis=0)

class_2, class_1 = dataset_non_elegible.COMMODITY_DUAL.value_counts()
c2 = dataset_non_elegible[dataset_non_elegible['COMMODITY_DUAL'] == 0]
c1 = dataset_non_elegible[dataset_non_elegible['COMMODITY_DUAL'] == 1]
df_2 = c2.sample(class_1)
under_dual = pd.concat([df_2, c1], axis=0)

X = under_dual.iloc[:, :-5].values
y = under_dual["COMMODITY_DUAL"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#SCALE
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#GRIDSEARCH
classifier = RandomForestClassifier(random_state = 0)
param_grid = {
    'n_estimators': [90,95,100,105,110],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8,9,10],
    'criterion': ['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=3)
CV_rfc.fit(X_train, y_train)
print(CV_rfc.best_params_)

#MODEL
rfc1 = RandomForestClassifier(random_state=0, max_features='auto', n_estimators=100,
                              max_depth=8, criterion='gini')
rfc1.fit(X_train, y_train)
y_pred = rfc1.predict(X_test)

accuracy_train = rfc1.score(X_train, y_train)
print("Random Forest - Accuracy on the training set: " + str(accuracy_train))
print("Random Forest - Accuracy on the test set: " + str(accuracy_score(y_test, y_pred)))
print("Random Forest - Precision: " + str(precision_score(y_test, y_pred)))
print("Random Forest - Recall: " + str(recall_score(y_test, y_pred)))

X_pred = dataset_elegible.iloc[:, :-5].values
X_pred = sc.transform(X_pred)
predicted = rfc1.predict_proba(X_pred)
predicted = pd.DataFrame(predicted)
prova_propensity = predicted[predicted[1] > 0.5]
    
    
    
