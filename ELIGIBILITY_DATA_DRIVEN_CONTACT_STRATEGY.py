# Import the required libraries and load the dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("DLT_AI_and_DATA_CUSTOMER_BASE_EN (1).csv", sep=";", encoding = 'unicode_escape')
dataset = pd.DataFrame(dataset)


###### EXPLORATORY DATA ANALYSIS #######


#### COMMODITY ####

df_start = dataset.copy()
df_dummy_visual = pd.get_dummies(df_start["COMMODITY"])
df_commodity_eda = pd.concat([df_start.reset_index(drop=True), df_dummy_visual.reset_index(drop=True)], axis=1)

sns.countplot("DUAL", data = df_commodity_eda, palette = "hls")
plt.title('COMMODITY_DUAL', fontweight="bold", fontsize =10)
plt.show()

#SOLUTION/COMMODITY
crosstab_commodity_solutions = pd.crosstab(index=df_commodity_eda["DUAL"],
                                           columns=df_commodity_eda["SOLUTIONS"], normalize='index')
crosstab_commodity_solutions.plot.bar(figsize=(6, 4),
                                      rot=0).set(ylabel="%", xlabel="")
plt.title('Solution by commodity type', fontweight="bold", fontsize=10)
plt.xticks([0, 1], ['NON DUAL', 'DUAL'])
plt.show()

# DUAL/LOYALTY
crosstab_loyalty_dual = pd.crosstab(index=df_commodity_eda["DUAL"],
                                    columns=df_commodity_eda["LOYALTY_PROGRAM"],
                                    normalize="index")
crosstab_loyalty_dual.plot.bar(figsize=(6, 4),
                               rot=0, color=('green', 'red')).set(ylabel="%", xlabel="")
plt.title('Loyalty by commodity', fontweight="bold", fontsize=10)
plt.xticks([0, 1], ['NON DUAL', 'DUAL'])
plt.show()

# DUAL/WEB REGISTRATION
crosstab_loyalty_dual = pd.crosstab(index=df_commodity_eda["DUAL"],
                                    columns=df_commodity_eda["WEB_PORTAL_REGISTRATION"],
                                    normalize="index")
crosstab_loyalty_dual.plot.bar(figsize=(6, 4),
                               rot=0, color=('green', 'red')).set(ylabel="%", xlabel="")
plt.title('DUAL AND WEB PORTAL', fontweight="bold", fontsize=10)
plt.xticks([0, 1], ['NON DUAL', 'DUAL'])
plt.show()

# DUAL/SENIORITY
crosstab_custsen_solutions = pd.crosstab(index=df_commodity_eda["DUAL"],
                                         columns=df_commodity_eda["CUSTOMER_SENIORITY"], normalize='index')
crosstab_custsen_solutions.plot.bar(figsize=(6, 4),
                                    rot=0, color=('hotpink', 'deeppink', 'mediumvioletred')).set(ylabel="Percentage",
                                                                                                 xlabel="")
plt.title('DUAL by Customer Seniority', fontweight="bold", fontsize=10)
plt.xticks([0, 1], ['NON DUAL', 'DUAL'])
plt.show()

#PRIVACY
crosstab_privacy_solutions = pd.crosstab(index=dataset["DUAL"],
                                    columns=dataset["CONSENSUS_PRIVACY"], normalize='index')
crosstab_privacy_solutions.plot.bar(figsize=(6, 4),
                               rot=0, color=('red', 'green')).set(ylabel="Percentage", xlabel = "Solution")
plt.title('DUAL by Privacy Consensus', fontweight="bold", fontsize =10)
plt.show()

# DUAL/AREA
crosstab_area_solutions = pd.crosstab(index=df_commodity_eda["DUAL"],
                                      columns=df_commodity_eda["AREA"], normalize='index')
crosstab_area_solutions.plot.bar(figsize=(6, 4),
                                 rot=0, color=('limegreen', 'lightcoral', 'deepskyblue', 'gold')).set(
    ylabel="Percentage", xlabel="Solution")
plt.title('DUAL by Area of Italy', fontweight="bold", fontsize=10)
plt.xticks([0, 1], ['NON DUAL', 'DUAL'])
plt.show()

# DUAL/FLAG_BAD_CUSTOMER
crosstab_flag_solutions = pd.crosstab(index=df_commodity_eda["DUAL"],
                                      columns=df_commodity_eda["FLAG_BAD_CUSTOMER"], normalize='index')
crosstab_flag_solutions.plot.bar(figsize=(6, 4),
                                 rot=0, color=('yellowgreen', 'crimson')).set(ylabel="Percentage", xlabel="")
plt.title('Solutions by Bad Customer Flag', fontweight="bold", fontsize=10)
plt.xticks([0, 1], ['NON DUAL', 'DUAL'])
plt.show()

# AVG CONSUMPTION GAS and AVG CONSUMPTION POWER
bp = sns.boxplot(data=df_commodity_eda, x="DUAL", y="AVG_CONSUMPTION_GAS_M3")  # RUN PLOT
bp.set_ylim([0, 200000])
plt.xticks([0, 1], ['NON DUAL', 'DUAL'])
plt.show()

#DUAL/EMAIL_VALIDATED
crosstab_dual_mail = pd.crosstab(index=df_commodity_eda["DUAL"],
                                 columns=df_commodity_eda["EMAIL_VALIDATED"], normalize='index')
crosstab_dual_mail.plot.bar(figsize=(6, 4),
                             rot=0, color=('green', 'red')).set(ylabel="Percentage", xlabel="")
plt.title('Dual by email_validated', fontweight="bold", fontsize=10)
plt.xticks([0, 1], ['NON DUAL', 'DUAL'])
plt.show()

#DUAL/GENRE
crosstab_dual_mail = pd.crosstab(index=df_commodity_eda["DUAL"],
                                 columns=df_commodity_eda["GENRE"], normalize='index')
crosstab_dual_mail.plot.bar(figsize=(6, 4),
                             rot=0, color=('yellowgreen', 'crimson')).set(ylabel="Percentage", xlabel="")
plt.title('Dual by genre', fontweight="bold", fontsize=10)
plt.xticks([0, 1], ['NON DUAL', 'DUAL'])
plt.show()

#DUAL/CLC_STATUS
crosstab_commodity_solutions = pd.crosstab(index=df_commodity_eda["DUAL"],
                                    columns=df_commodity_eda["CLC_STATUS"], normalize='index')
crosstab_commodity_solutions.plot.bar(figsize=(6, 4),
                               rot=0).set(ylabel="Count")
plt.xticks([0, 1], ['NON DUAL', 'DUAL'])
plt.title('dual by clc_status', fontweight="bold", fontsize =10)
plt.show()

# AVG CONSUMPTION GAS and AVG CONSUMPTION POWER
bp = sns.boxplot(data=df_commodity_eda, x="DUAL", y="AVG_CONSUMPTION_GAS_M3")  # RUN PLOT
bp.set_ylim([0, 200000])
plt.xticks([0, 1], ['NON DUAL', 'DUAL'])
plt.show()

##### SOLUTIONS #####

sns.countplot("SOLUTIONS", data = dataset, palette = "hls")
plt.title('SOLUTIONS', fontweight="bold", fontsize =10)
plt.show()

#The following graphs show the distribution of variables per solution values

#COMMODITY
crosstab_commodity_solutions = pd.crosstab(index=dataset["SOLUTIONS"],
                                    columns=dataset["COMMODITY"], normalize='index')
crosstab_commodity_solutions.plot.bar(figsize=(6, 4),
                               rot=0).set(ylabel="Count", xlabel = "Solution")
plt.title('Solutions by Commodity type', fontweight="bold", fontsize =10)
plt.show()

#PRIVACY
crosstab_privacy_solutions = pd.crosstab(index=dataset["SOLUTIONS"],
                                    columns=dataset["CONSENSUS_PRIVACY"], normalize='index')
crosstab_privacy_solutions.plot.bar(figsize=(6, 4),
                               rot=0, color=('red', 'green')).set(ylabel="Percentage", xlabel = "Solution")
plt.title('Solutions by Privacy Consensus', fontweight="bold", fontsize =10)
plt.show()


#CUSTOMER SENIORITY
crosstab_custsen_solutions = pd.crosstab(index=dataset["SOLUTIONS"],
                                    columns=dataset["CUSTOMER_SENIORITY"], normalize='index')
crosstab_custsen_solutions.plot.bar(figsize=(6, 4),
                               rot=0, color=('hotpink', 'deeppink', 'mediumvioletred')).set(ylabel="Percentage", xlabel = "Solution")
plt.title('Solutions by Customer Seniority', fontweight="bold", fontsize =10)
plt.show()

#GENDER
crosstab_genre_solutions = pd.crosstab(index=dataset["SOLUTIONS"],
                                    columns=dataset["GENRE"], normalize='index')
crosstab_genre_solutions.plot.bar(figsize=(6, 4),
                               rot=0, color=('lightcoral', 'gold')).set(ylabel="Percentage", xlabel = "Solution")
plt.title('Solutions by Genre', fontweight="bold", fontsize =10)
plt.show()

#LOYALITY PROGRAM
crosstab_loyalityprog_solutions = pd.crosstab(index=dataset["SOLUTIONS"],
                                    columns=dataset["LOYALTY_PROGRAM"], normalize='index')
crosstab_loyalityprog_solutions.plot.bar(figsize=(6, 4),
                               rot=0, color=('darkorchid', 'royalblue')).set(ylabel="Percentage", xlabel = "Solution")
plt.title('Solutions by Loyalty Program', fontweight="bold", fontsize =10)
plt.show()

#AREA
crosstab_area_solutions = pd.crosstab(index=dataset["SOLUTIONS"],
                                    columns=dataset["AREA"], normalize='index')
crosstab_area_solutions.plot.bar(figsize=(6, 4),
                               rot=0, color=('limegreen', 'lightcoral', 'deepskyblue', 'gold')).set(ylabel="Percentage", xlabel = "Solution")
plt.title('Solutions by Area of Italy', fontweight="bold", fontsize =10)
plt.show()

#FLAG_BAD_CUSTOMER
crosstab_flag_solutions = pd.crosstab(index=dataset["SOLUTIONS"],
                                    columns=dataset["FLAG_BAD_CUSTOMER"], normalize='index')
crosstab_flag_solutions.plot.bar(figsize=(6, 4),
                               rot=0, color=('yellowgreen', 'crimson')).set(ylabel="Percentage", xlabel = "Solution")
plt.title('Solutions by Bad Customer Flag', fontweight="bold", fontsize =10)
plt.show()

#AVG CONSUMPTION GAS and AVG CONSUMPTION POWER
bp = sns.boxplot(data = dataset, x = "SOLUTIONS", y= "AVG_CONSUMPTION_GAS_M3", hue="SOLUTIONS")  
bp.set_ylim([0, 200000])
plt.show()


'''
f, axes = plt.subplots(1, 2, figsize=(15, 6))
f.tight_layout()
g1 = sns.histplot(data = dataset[dataset["SOLUTIONS"] == 1], x = "AVG_CONSUMPTION_GAS_M3", kde=True, color = "green", ax=axes[0], bins=100)
g1.set_title('Average consumption gas (Solution = 1)')
g2 = sns.histplot(data = dataset[dataset["SOLUTIONS"] == 0], x = "AVG_CONSUMPTION_GAS_M3", kde=True, color = "green", ax=axes[1])
g2.set_title('Average consumption gas (Solution = 0)')
plt.ylim(0, 3000)
plt.xlim(0, 50000)
plt.show()
'''

#BEHAVIOUR SCORE
crosstab_commodity_solutions = pd.crosstab(index=dataset["SOLUTIONS"],
                                    columns=dataset["BEHAVIOUR_SCORE"], normalize='index')
crosstab_commodity_solutions.plot.bar(figsize=(6, 4), color = ["skyblue","blue","dodgerblue"],
                                      rot=0).set(ylabel="Count", xlabel = "Solution")
plt.title('Solutions by Behaviour Score', fontweight="bold", fontsize =10)
plt.show()

#CLC STATUS
crosstab_commodity_solutions = pd.crosstab(index=dataset["SOLUTIONS"],
                                    columns=dataset["CLC_STATUS"], normalize='index')
crosstab_commodity_solutions.plot.bar(figsize=(6, 4), color = ["skyblue","orchid", "yellowgreen","blue","dodgerblue"],
                                      rot=0).set(ylabel="Count", xlabel = "Solution")
plt.title('Solutions by CLC Status', fontweight="bold", fontsize =10)
plt.show()



##### Deleting customers at risk of being churned or leaving (they cannot be contacted for commercial purposes)
dataset4 = dataset.drop(dataset[dataset["CLC_STATUS"] == "4-Risk churn"].index)
dataset4 = dataset4.drop(dataset4[dataset4["CLC_STATUS"] == "5-Leaving"].index)



##### OUTLIER DETECTION ######

# Replace the outliers with the respective median

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

dataset4 = dataset4.drop(exp_dataset4.columns, axis = 1)
dataset4 = pd.concat([dataset4.reset_index(drop=True),exp_dataset4.reset_index(drop = True)], axis=1)

descr2 = dataset4.describe().T


#### Create a dataset made of dummy variables deriving from the original ones
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

# to overcome the problem of the unbalanced dataset, we randomly selecte observation from the majority class and delet them from the training dataset (random undersampling)
class_2,class_1 = dummy_df.SOLUTIONS.value_counts() # The variables on the first line are of int datatype and shall be used in order to tell how much of a sample we want
c2 = dummy_df[dummy_df['SOLUTIONS'] == 0] # # While between the second and the third line are of DataFrame datatype each is a slice of the DataFrame containing only one type of class. 
c1 = dummy_df[dummy_df['SOLUTIONS'] == 1]
df_2 = c2.sample(class_1) # On the fourth line, we re-assign the DataFrames to new ones but we will apply the sample function to it and pass to it the int value of the least class, in this case, class_1.
under_sol = pd.concat([df_2,c1],axis=0) # Lastly, we will concatenate the last new two DataFrames as well as one of the original Dataframes, the one which contains the minority class label.
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

# check for correlation among the independent variables
import seaborn as sn

# Get diagonal and lower triangular pairs of correlation matrix
def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

# List top absolute correlations
def get_top_abs_correlations(df, n=5):
    au_corr = df.corr(method = "spearman").abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
var_corr = get_top_abs_correlations(under_dual, 50)
var_corr_2 = get_top_abs_correlations(under_sol, 50)

# check for correlation between the independent variables and the dependent ones
sol_corr = under_sol[under_sol.columns].corr(method = "spearman")['SOLUTIONS'][:]
dual_corr = under_dual[under_dual.columns].corr(method = "spearman")['COMMODITY_DUAL'][:]
sol_corr = pd.DataFrame(sol_corr).sort_values(by = 'SOLUTIONS', ascending=False)
dual_corr = pd.DataFrame(dual_corr).sort_values(by = 'COMMODITY_DUAL', ascending=False)




##### FEATURE SELECTION #####

### MEAN ANALYSIS ###

## let's start by defining some variables that will come in handy later.
# We make a distinction between AVG_CONSUPTION and the other variables due to the different ranges of values that characterize them
avg_df = dataset4.filter(["AVG_CONSUMPTION_GAS_M3", "AVG_CONSUMPTION_POWER_KWH", "SOLUTIONS"], axis=1)
dummy_avg_df = pd.get_dummies(avg_df)

avg_df_2 = dataset4.filter(["AVG_CONSUMPTION_GAS_M3", "AVG_CONSUMPTION_POWER_KWH", "COMMODITY"], axis=1)
dummy_avg_df_2 = pd.get_dummies(avg_df_2)
dummy_avg_df_2 = dummy_avg_df_2.drop(["COMMODITY_GAS", "COMMODITY_POWER"], axis=1)


## SOLUTIONS
# remove the variables belonging to the customer care interactions category, which we hypothesized not be relevant to in this case
df_mean = dummy_df.drop(["AVG_CONSUMPTION_GAS_M3", "AVG_CONSUMPTION_POWER_KWH", 'N_DEM_CROSS_SELLING',
                          'N_SMS_CROSS_SELLING', 'N_TLS_CROSS_SELLING', 'N_DEM_SOLUTION',
                          'N_SMS_SOLUTION', 'N_TLS_SOLUTION'], axis = 1)
# divide the dataset into those who have already purchased the contract and those who have not. 
# then, for both groups we compute the average value of each of the features present in the dataset
df_mean = df_mean.groupby('SOLUTIONS').mean() 
df_mean.reset_index(drop=True, inplace=True)
columns = list(df_mean.columns)
df_mean = df_mean.T
df_mean["variables"] = columns
df_mean["difference"] = abs(df_mean[0] - df_mean[1])
# The results are  sorted in descending order on the basis of the difference between the means of the two groups.
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



### RANDOM FOREST ###

# DUAL
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score


# df_random_dual = pd.read_csv("project_deloitte/Logistic_dual.csv")
# once again, we remove the variables belonging to the customer care interactions category
under_dual = under_dual.drop(['N_DEM_CROSS_SELLING', 'N_SMS_CROSS_SELLING',
                              'N_TLS_CROSS_SELLING', 'N_DEM_SOLUTION',
                              'N_SMS_SOLUTION', 'N_TLS_SOLUTION'], axis=1)

# define the set of (potential) independent variables 
X = under_dual.iloc[:, under_dual.columns != "COMMODITY_DUAL"].values
# define the dependent variable
y = under_dual["COMMODITY_DUAL"]

# split our dataset into two sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Standardize features by removing the mean and scaling to unit variance
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Before actually being trained, the algorithm goes through a tuning phase in order to maximize its performance.
# without overfitting or creating too high of a variance.
'''
# GRID SEARCH / OPTIMIZATION
rf_final = RandomForestClassifier(random_state=0)
# manually defining a subset of the hyperparametric space 
param_grid = {
    'n_estimators': [100, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}
# Each combination’s performance is then evaluated using cross-validation and the best performing hyperparametric combination is chosen.
CV_rfc = GridSearchCV(estimator=rf_final, param_grid=param_grid, cv=5)
CV_rfc.fit(X_train, y_train)
print(CV_rfc.best_params_)
'''

# fit the model with the best performing hyperparametric combination
rfc1=RandomForestClassifier(random_state=0, max_features='auto', n_estimators= 500, max_depth=7, criterion='gini')
rfc1.fit(X_train, y_train)
y_pred = rfc1.predict(X_test)

# Evaluate the model
mat = confusion_matrix(y_test, y_pred) # compute the confusion matrix
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.title("Random Forest Confusion Matrix", fontweight="bold", fontsize=12)
plt.show()

accuracy_train = rfc1.score(X_train, y_train)
print("Random Forest - Accuracy on the training set: " + str(accuracy_train))
print("Random Forest - Accuracy on the test set: " + str(accuracy_score(y_test, y_pred)))
print("Random Forest - Precision: " + str(precision_score(y_test, y_pred)))
print("Random Forest - Recall: " + str(recall_score(y_test, y_pred)))

# visualize how much each variable is contributing to the decision.
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

# Interestingly, the results deriving from the "mean analysis" have been almost completely confirmed.

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

'''
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
'''

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


# Create the set of predictors on the basis of the results obtained from the feature selection.
# The variables that lack of an economic meaning, together with the strongly correlated ones (e.g. the dummy variables deriving from the same original feature), are removed
# The variables that would have helped us to identify non-elegible customers (e.g. "CONSENSUS_PRIVACY" and "PHONE_VALIDATED") have been added

different_cols = dummy_df.columns.difference(dataset4.columns)
dataset_difference = dummy_df[different_cols]

dataset_whole = pd.merge(dataset4, dataset_difference, left_index=True,
                     right_index=True, how='inner')

dataset_dual = dataset_whole.filter(['CLC_STATUS_3-Customer Loyalty', 'AREA_North-West','WEB_PORTAL_REGISTRATION', 'LAST_CAMPAIGN_TIPOLOGY_Cross-Selling',
                                     'N_DISUSED_GAS_POINTS', 
                                     'LAST_CAMPAIGN_TIPOLOGY_Caring', 'SOLUTIONS', 'CUSTOMER_SENIORITY_>3 YEARS', 'LAST_CAMPAIGN_TIPOLOGY_Renewal', 
                                     'CUSTOMER_SENIORITY_1-3 YEARS', 'AVG_CONSUMPTION_GAS_M3','LAST_GAS_PRODUCT_Traditional',
                                     "CONSENSUS_PRIVACY", "ID", "COMMODITY_DUAL", "PHONE_VALIDATED", "EMAIL_VALIDATED"], axis=1)



### DUAL PROPENSITY ###

# constitute a dataset made up of non-eligible customers
def non_elegible(df):
    df = df.drop(df[(df["CONSENSUS_PRIVACY"] == "YES") & (df["COMMODITY_DUAL"] == 0)].index)
    return df
dataset_non_elegible = non_elegible(dataset_dual)
dataset_non_elegible = dataset_non_elegible.dropna(axis=0)

#  constitute a dataset made up of eligible customers
def elegible(df):
    df = df.drop(df[(df["CONSENSUS_PRIVACY"] == "NO") | (df["COMMODITY_DUAL"] == 1)].index)
    df = df.drop(df[(df["PHONE_VALIDATED"] == "KO") & (df["EMAIL_VALIDATED"] == 0)].index)
    return df
dataset_elegible = elegible(dataset_dual)
dataset_elegible = dataset_elegible.dropna(axis=0)

# The training set and the test set derive from dataset_non_elegible. 
# Therefore, also in this case, we perform a random undersampling
class_2, class_1 = dataset_non_elegible.COMMODITY_DUAL.value_counts()
c2 = dataset_non_elegible[dataset_non_elegible['COMMODITY_DUAL'] == 0]
c1 = dataset_non_elegible[dataset_non_elegible['COMMODITY_DUAL'] == 1]
import random
random.seed(123)
df_2 = c2.sample(class_1)
under_dual = pd.concat([df_2, c1], axis=0)
    

    
#Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
X = under_dual.iloc[:, :-5].values
y = under_dual["COMMODITY_DUAL"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
rfc1 = RandomForestClassifier(random_state=0, max_features='auto', n_estimators=100,
                              max_depth=8, criterion='gini')
rfc1.fit(X_train, y_train)
y_pred = rfc1.predict(X_test)

accuracy_train = rfc1.score(X_train, y_train)
print("Random Forest - Accuracy on the training set: " + str(accuracy_train))
print("Random Forest - Accuracy on the test set: " + str(accuracy_score(y_test, y_pred)))
print("Random Forest - Precision: " + str(precision_score(y_test, y_pred)))
print("Random Forest - Recall: " + str(recall_score(y_test, y_pred)))

X_pred = dataset_elegible.iloc[:, :-5].values #create a subset of dataset_elegible  
X_pred = sc.transform(X_pred) 
predicted = rfc1.predict_proba(X_pred) # predict the probability that the customer will subscribe or not the contract 
predicted = pd.DataFrame(predicted)
predicted['RANDOM PREDICTION'] = np.where(predicted[0] >= 0.5, 0, 1) # round the probabilities on the basis of the selected threshold and add a column with the final result
dataset_elegible['RANDOM PREDICTION'] = predicted['RANDOM PREDICTION'].values


ensemble = pd.DataFrame(y_test)
ensemble['RANDOM PREDICTION'] = pd.Series(y_pred).values


#KNN
from sklearn.neighbors import KNeighborsClassifier

'''
X = under_dual.iloc[:, :-5].values
y = under_dual["COMMODITY_DUAL"]
X_train, X_test, y_train_dual, y_test_dual = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
X_train_dual = sc.fit_transform(X_train)
X_test_dual = sc.transform(X_test)
'''

knn_model_dual = KNeighborsClassifier(n_neighbors = 19)
knn_model_dual.fit(X_train, y_train)
y_pred_knn_dual =knn_model_dual.predict(X_test)

# Evaluation metrics
accuracy_train = knn_model_dual.score(X_train, y_train)
print("KNN - Accuracy on the training set: " + str(accuracy_train))
from sklearn.metrics import accuracy_score, precision_score, recall_score
print("KNN - Accuracy on the test set: " + str(accuracy_score(y_test, y_pred_knn_dual)))
print("KNN - Precision: " + str(precision_score(y_test, y_pred_knn_dual)))
print("KNN - Recall: " + str(recall_score(y_test,y_pred_knn_dual)))

# because we need the probability of the event besides its classification, 
# in the case of KNN (and later also of SVC) we recur to calibrated probabilities.
from sklearn.calibration import CalibratedClassifierCV
calib_clf_dual = CalibratedClassifierCV(knn_model_dual, cv=3, method='sigmoid') # fit and calibrate model on training data
calib_clf_dual.fit(X_train, y_train)
y_calibprob_dual = calib_clf_dual.predict_proba(X_test) # evaluate the model
y_calibprob_dual = pd.DataFrame(y_calibprob_dual)
y_test_dual = pd.DataFrame(y_test)

predicted_knn_dual = pd.concat([y_calibprob_dual.reset_index(drop=True),y_test_dual.reset_index(drop=True)], axis=1)
predicted_knn_dual['predicted'] = np.where(predicted_knn_dual[0]>= 0.5, 0,1)

confusion_matrix_dual = pd.crosstab(predicted_knn_dual['COMMODITY_DUAL'], predicted_knn_dual['predicted'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix_dual)

'''
X_pred_dual = dataset_elegible.iloc[:,:-5]
X_pred_dual = sc.transform(X_pred_dual)
'''

pred_dual = calib_clf_dual.predict_proba(X_pred)
pred_dual = pd.DataFrame(pred_dual)
ID_column = dataset_elegible["ID"]
pred_dual = pd.concat([pred_dual, ID_column.reset_index(drop=True)], axis = 1)

pred_dual['KNN PREDICTION'] = np.where(pred_dual[0] >= 0.5, 0, 1)
#dataset_elegible['KNN PREDICTION'] = pred_dual['KNN PREDICTION'].values

ensemble['KNN PREDICTION'] = predicted_knn_dual['predicted'].values


#SVM
from sklearn.svm import SVC
svm_model = SVC(C=1, coef0=0.01, degree=3, gamma='auto', kernel='rbf')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

clf = CalibratedClassifierCV(svm_model, cv=3) 
clf.fit(X_train, y_train)
y_proba = clf.predict_proba(X_test)
y_proba = pd.DataFrame(y_proba)
y_proba['SVM PREDICTION'] = np.where(y_proba[0] >= 0.5, 0, 1)

accuracy_train = svm_model.score(X_train, y_train)
print("Accuracy on the training set: " + str(accuracy_train))
print("Accuracy on the test set: " + str(accuracy_score(y_test, y_pred_svm)))
print("Precision: " + str(precision_score(y_test, y_pred_svm)))
print("Recall: " + str(recall_score(y_test, y_pred_svm)))

pred_dual_svm = clf.predict_proba(X_pred)
pred_dual_svm = pd.DataFrame(pred_dual_svm)
pred_dual_svm['ID']= dataset_elegible['ID'].values
#pred_dual_svm = pred_dual_svm[pred_dual_svm[1]>0.2]

pred_dual_svm['SVM PREDICTION'] = np.where(pred_dual_svm[0] >= 0.5, 0, 1)
#dataset_elegible['SVM PREDICTION'] = pred_dual_svm['SVM PREDICTION'].values

ensemble['SVM PREDICTION'] = y_proba['SVM PREDICTION'].values


#Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train.values.ravel())
y_pred_log = logreg.predict(X_test)
y_pred_log = pd.DataFrame(y_pred_log)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
print("Logistic Regression - Accuracy on the test set: " + str(accuracy_score(y_test, y_pred_log)))
print("Logistic Regression - Precision: " + str(precision_score(y_test, y_pred_log)))
print("Logistic Regression - Recall: " + str(recall_score(y_test, y_pred_log)))

ensemble['LOG PREDICTION'] = y_pred_log[0].values

pred_dual_log = logreg.predict_proba(X_pred)
pred_dual_log = pd.DataFrame(pred_dual_log)
pred_dual_log['ID']= dataset_elegible['ID'].values
#pred_dual_log = pred_dual_log[pred_dual_log[1] > 0.2]
pred_dual_log['LOG PREDICTION'] = np.where(pred_dual_log[0] >= 0.5, 0, 1)
#dataset_elegible['LOG PREDICTION'] = pred_dual_log['LOG PREDICTION'].values


#Ensemble method
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
eclf_dual = VotingClassifier(estimators=[('Random Forest', rfc1), ('KNN', calib_clf_dual), ('SVM', clf), ('Logistic Regression', logreg)], voting='soft')
eclf_dual = eclf_dual.fit(X_train, y_train)
print(eclf_dual.predict(X_test))

y_proba_ens = eclf_dual.predict_proba(X_test)
y_proba_ens = pd.DataFrame(y_proba_ens)
y_proba_ens['ENSEMBLE PREDICTION'] = np.where(y_proba_ens[0] >= 0.5, 0, 1)
ensemble['ENSEMBLE PREDICTION'] = y_proba_ens['ENSEMBLE PREDICTION'].values
pd.crosstab(ensemble['COMMODITY_DUAL'], ensemble['ENSEMBLE PREDICTION'])

(453+517)/(453+517+99+14)
517/(517+99)
517/(517+14)

#create a dataset to compare the algorithm performance 
metrics = {'accuracy': [0.8919667590027701, 0.866112650046168, 0.8910433979686058, 0.8919667590027701, 0.8910433979686058], 'precision': [0.8349514563106796, 0.8177496038034865, 0.8314606741573034, 0.8306709265175719, 0.8293460925039873], 'recall': [0.9717514124293786, 0.943502824858757, 0.975517890772128, 0.9792843691148776, 0.9792843691148776]}
metrics_dual = pd.DataFrame.from_dict(metrics)
metrics_dual = metrics_dual.set_axis(['Random Forest', 'KNN', 'SVM', 'Log Regression', "Ensemble"])
metrics_dual

# as it is the ensemble that has the highest performance, we choose this to predict who will sign the contract and who will not
predictions = eclf_dual.predict_proba(X_pred)
predictions = pd.DataFrame(predictions)
predictions['ID']= dataset_elegible['ID'].values 
predictions['PREDICTION_DUAL'] = np.where(predictions[0] >= 0.5, 0, 1) 

predictions = predictions.drop([0,1], axis = 1)

# perform a left join between dataset and predictions on the basis of "ID", returning a dataframe containing all the rows of the left dataframe (dataset)
# All the non-matching rows of the left dataframe contain NaN for the columns in the right dataframe.
dataset = dataset.merge(predictions,how='left', left_on='ID', right_on='ID') 
dataset.columns


### SOLUTION PROPENSITY ###
def non_elegible_sol(df):
    df = df.drop(df[(df["CONSENSUS_PRIVACY"] == "YES") & (df["SOLUTIONS"] == 0)].index)
    return df
def elegible_sol(df):
    df = df.drop(df[(df["CONSENSUS_PRIVACY"] == "NO") | (df["SOLUTIONS"] == 1)].index)
    df = df.drop(df[(df["PHONE_VALIDATED"] == "KO") & (df["EMAIL_VALIDATED"] == 0)].index)
    return df

dataset_sol = dataset_whole.filter(['AVG_CONSUMPTION_GAS_M3', "COMMODITY_DUAL", 'ZONE_Piemonte', 'WEB_PORTAL_REGISTRATION', 
                                 'AREA_North-West', 'CLC_STATUS_3-Customer Loyalty', 'BEHAVIOUR_SCORE_GOOD PAYER', 'LOYALTY_PROGRAM', 'AREA_SOUTH', 'ZONE_VENETO', 
                                 'LAST_CAMPAIGN_TIPOLOGY_Caring', 'AREA_North-East',
                                 'LAST_CAMPAIGN_TIPOLOGY_Cross-Selling','CUSTOMER_SENIORITY_>3 YEARS', 'CUSTOMER_SENIORITY_<1 YEAR',
                                 'ACQUISITION_CHANNEL_CC', 'BEHAVIOUR_SCORE_BAD PAYER', "AREA_CENTER", "CONSENSUS_PRIVACY", "ID" 
                                 ,"PHONE_VALIDATED", "EMAIL_VALIDATED", "SOLUTIONS"], axis=1)

dataset_elegib_sol = elegible_sol(dataset_sol)
dataset_non_elegib_sol = non_elegible_sol(dataset_sol)
dataset_elegib_sol.dropna(axis=0, inplace=True)
dataset_non_elegib_sol.dropna(axis=0, inplace=True)

class_2, class_1 = dataset_non_elegib_sol.SOLUTIONS.value_counts()
c2 = dataset_non_elegib_sol[dataset_non_elegib_sol['SOLUTIONS'] == 0]
c1 = dataset_non_elegib_sol[dataset_non_elegib_sol['SOLUTIONS'] == 1]
random.seed(123)
df_3 = c2.sample(class_1)
under_sol = pd.concat([df_3, c1], axis=0)

X_sol = under_sol.iloc[:, :-5].values
y_sol = under_sol["SOLUTIONS"]

#under_sol.to_csv("under_sol.csv")
#dataset_elegib_sol.to_csv("dataset_elegib_sol.csv")

X_train_sol, X_test_sol, y_train_sol, y_test_sol = train_test_split(X_sol, y_sol, test_size=0.3, random_state=0)
sc = StandardScaler()
X_train_sol = sc.fit_transform(X_train_sol)
X_test_sol = sc.transform(X_test_sol)

#Random Forest
rfc2 = RandomForestClassifier(random_state=0, max_features='auto', n_estimators=105,
                              max_depth=4, criterion='gini')
rfc2.fit(X_train_sol, y_train_sol)
y_pred_sol = rfc2.predict(X_test_sol)
y_pred_sol = pd.DataFrame(y_pred_sol)

accuracy_train = rfc2.score(X_train_sol, y_train_sol)
print("Random Forest - Accuracy on the training set: " + str(accuracy_train))
print("Random Forest - Accuracy on the test set: " + str(accuracy_score(y_test_sol, y_pred_sol)))
print("Random Forest - Precision: " + str(precision_score(y_test_sol, y_pred_sol)))
print("Random Forest - Recall: " + str(recall_score(y_test_sol, y_pred_sol)))

X_pred_sol = dataset_elegib_sol.iloc[:, :-5].values
X_pred_sol = sc.transform(X_pred_sol)

ensemble_sol = pd.DataFrame(y_test_sol)
ensemble_sol['RANDOM PREDICTION'] = y_pred_sol[0].values


#SVM
svm_model = SVC(C=1, coef0=0.5, degree=3, gamma='auto', kernel='poly')
svm_model.fit(X_train_sol, y_train_sol)
y_pred_svm = svm_model.predict(X_test_sol)

accuracy_train = svm_model.score(X_train_sol, y_train_sol)
print("Accuracy on the training set: " + str(accuracy_train))
print("Accuracy on the test set: " + str(accuracy_score(y_test_sol, y_pred_svm)))
print("Precision: " + str(precision_score(y_test_sol, y_pred_svm)))
print("Recall: " + str(recall_score(y_test_sol, y_pred_svm)))

svm = SVC(C=1, coef0=0.5, degree=3, gamma='auto', kernel='poly')
clf_sol = CalibratedClassifierCV(svm_model, cv=3) 
clf_sol.fit(X_train_sol, y_train_sol)
y_proba_sol = clf_sol.predict_proba(X_test_sol)
y_proba_sol = pd.DataFrame(y_proba_sol)
y_test_sol = pd.DataFrame(y_test_sol)
y_proba_sol['SVM PREDICTION'] = np.where(y_proba_sol[0] >= 0.5, 0, 1)
ensemble_sol['SVM PREDICTION'] = y_proba_sol['SVM PREDICTION'].values


#Logistic Regression
mod_log_sol = LogisticRegression()
mod_log_sol.fit(X_train_sol, y_train_sol.values.ravel())
mod_pred_log = mod_log_sol.predict(X_test_sol)

print("Accuracy on the test set: " + str(accuracy_score(y_test_sol, mod_pred_log)))
print("Precision: " + str(precision_score(y_test_sol, mod_pred_log)))
print("Recall: " + str(recall_score(y_test_sol, mod_pred_log)))


mod_pred_log = pd.DataFrame(mod_pred_log)
ensemble_sol['LOG PREDICTION'] = mod_pred_log[0].values

#KNN
knn_model_sol = KNeighborsClassifier(n_neighbors = 9)
knn_model_sol.fit(X_train_sol, y_train_sol)
y_pred_knn_sol =knn_model_sol.predict(X_test_sol)

accuracy_train = knn_model_sol.score(X_train_sol, y_train_sol)
print("KNN - Accuracy on the training set: " + str(accuracy_train))
from sklearn.metrics import accuracy_score, precision_score, recall_score
print("KNN - Accuracy on the test set: " + str(accuracy_score(y_test_sol, y_pred_knn_sol)))
print("KNN - Precision: " + str(precision_score(y_test_sol, y_pred_knn_sol)))
print("KNN - Recall: " + str(recall_score(y_test_sol,y_pred_knn_sol)))

from sklearn.calibration import CalibratedClassifierCV
calib_clf_sol = CalibratedClassifierCV(knn_model_sol, cv=3, method='sigmoid')
calib_clf_sol.fit(X_train_sol, y_train_sol)
y_calibprob_sol = calib_clf_sol.predict_proba(X_test_sol)
y_calibprob_sol = pd.DataFrame(y_calibprob_sol)
y_test_sol = pd.DataFrame(y_test_sol)

predicted_knn_sol = pd.concat([y_calibprob_sol.reset_index(drop=True),y_test_sol.reset_index(drop=True)], axis=1)
predicted_knn_sol['predicted'] = np.where(predicted_knn_sol[0]>= 0.5, 0,1)

ensemble_sol['KNN PREDICTION'] = predicted_knn_sol['predicted'].values


#Ensemble method
eclf_sol = VotingClassifier(estimators=[('Random Forest', rfc2), ('KNN', calib_clf_sol), ('SVM', clf_sol), ('Logistic Regression', mod_log_sol)], voting='soft')
eclf_sol = eclf_sol.fit(X_train_sol, y_train_sol)
print(eclf_sol.predict(X_test_sol))

y_proba_ens_sol = eclf_sol.predict_proba(X_test_sol)
y_proba_ens_sol = pd.DataFrame(y_proba_ens_sol)
y_proba_ens_sol['ENSEMBLE PREDICTION'] = np.where(y_proba_ens_sol[0] >= 0.5, 0, 1)
ensemble_sol['ENSEMBLE PREDICTION'] = y_proba_ens_sol['ENSEMBLE PREDICTION'].values
pd.crosstab(ensemble_sol['SOLUTIONS'], ensemble_sol['ENSEMBLE PREDICTION'])

(42+33)/(42+33+12+18)
33/(33+12)
33/(33+18)

metrics_sol = {'accuracy': [0.6857142857142857, 0.6761904761904762, 0.6857142857142857, 0.6571428571428571, 0.7142857142857143], 'precision': [0.725, 0.6808510638297872, 0.6730769230769231, 0.6530612244897959, 0.7333333333333333], 'recall': [0.5686274509803921, 0.6274509803921569, 0.6862745098039216, 0.6274509803921569, 0.6470588235294118]}
metrics_sol = pd.DataFrame.from_dict(metrics_sol)
metrics_sol = metrics_sol.set_axis(['Random Forest', 'KNN', 'SVM', 'Log Regression', "Ensemble"])
metrics_sol


predictions_sol = eclf_sol.predict_proba(X_pred_sol)
predictions_sol = pd.DataFrame(predictions_sol)
predictions_sol['ID']= dataset_elegib_sol['ID'].values
predictions_sol['PREDICTION_SOL'] = np.where(predictions_sol[0] >= 0.5, 0, 1)

predictions_sol = predictions_sol.drop([0,1], axis = 1)

dataset = dataset.merge(predictions_sol,how='left', left_on='ID', right_on='ID') 
dataset.columns


dataset.value_counts('PREDICTION_SOL')
dataset.value_counts('PREDICTION_DUAL')


###ELIGIBILITY###

# create the dataset in which monthly contact strategis will be inserted
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


### DIVIDING DATASET ACCORDING TO THE ELIGIBILITY

# filtering the costumers chosen by the propensity model on the basis of the "general rule": 
# those customers whose last contact dates back to at least N months ago are eligible, where N = # months / # contacts per year.
pd.options.mode.chained_assignment = None
timefmt = "%d/%m/%Y" # set dates format
dataset_final_eligible['DATE_LAST_CAMPAIGN'] = pd.to_datetime(dataset_final_eligible['DATE_LAST_CAMPAIGN'], format = timefmt)

datatypes = dataset_final_eligible.dtypes

dataset_final_eligible['REFERENCE_DATE'] = "26/04/2022" # as reference date we take the day on which the project was delivered
dataset_final_eligible['REFERENCE_DATE'] = pd.to_datetime(dataset_final_eligible['REFERENCE_DATE'], format = timefmt)

dataset_final_eligible['N_months'] = ((dataset_final_eligible.DATE_LAST_CAMPAIGN - dataset_final_eligible.REFERENCE_DATE)/np.timedelta64(1, 'M')) # compute the difference between the reference date and the date on which the last contact took place
dataset_final_eligible['N_months'] = dataset_final_eligible['N_months'].astype(int).abs() # compute the absolute value of the result

# create six subset of dataset_final_eligible (ne for every possible combination of marketing campaign and communication channel) 
# on the basis of which the monthly contact strategies will be defined
cross_selling_tls_general = dataset_final_eligible[(dataset_final_eligible.N_months >= 6) & (dataset_final_eligible["PHONE_VALIDATED"] != "KO") & (dataset_final_eligible["PREDICTION_DUAL"] == 1)]
cross_selling_dem_general = dataset_final_eligible[(dataset_final_eligible.N_months >= 2) & (dataset_final_eligible["EMAIL_VALIDATED"] != 0) & (dataset_final_eligible["PREDICTION_DUAL"] == 1)] 
cross_selling_sms_general = dataset_final_eligible[(dataset_final_eligible.N_months >= 2) & (dataset_final_eligible["PHONE_VALIDATED"] != "KO") & (dataset_final_eligible["PREDICTION_DUAL"] == 1)]
solution_tls_general = dataset_final_eligible[(dataset_final_eligible.N_months >= 12) & (dataset_final_eligible["PHONE_VALIDATED"] != "KO") & (dataset_final_eligible["PREDICTION_SOL"] == 1)]
solution_dem_general = dataset_final_eligible[(dataset_final_eligible.N_months >= 6) & (dataset_final_eligible["EMAIL_VALIDATED"] != 0) & (dataset_final_eligible["PREDICTION_SOL"] == 1)]
solution_sms_general = dataset_final_eligible[(dataset_final_eligible.N_months >= 6) & (dataset_final_eligible["PHONE_VALIDATED"] != "KO") & (dataset_final_eligible["PREDICTION_SOL"] == 1)]


#CROSS SELLING
# The criterion is to maximize the number of contacts, while remaining within the limits imposed by both the campaign rules and the cross-campaign rules. 

for index, row in cross_selling_dem_general.iterrows(): # iterate through the dataset of the combination in question
    l = [row["ID"],1,0,1,0,1,0,1,0,1,0,1,0] # create a list that represents the monthly contact strategy for each customer of the respective dataset 
    Cross_Selling_DEM.loc[len(Cross_Selling_DEM)] = l # append the monthly contact strategy to the final dataset
    
for index, row in cross_selling_sms_general.iterrows():
    l = [row["ID"],1,0,1,0,1,0,1,0,1,0,1,0]
    Cross_Selling_SMS.loc[len(Cross_Selling_SMS)] = l

for index, row in cross_selling_tls_general.iterrows():
    l = [row["ID"],0,0,0,0,0,1,0,0,0,0,0,1]
    Cross_Selling_TLS.loc[len(Cross_Selling_TLS)] = l

#SOLUTION
# in order to respect the criterion, we have to take into consideration if the consumer was contactable for both campaigns (solution and cross-selling), if he was “email validated” and, finally, if he was “phone validated”. 
# we repeat what we did before, adjusting the strategy for each of the possible combinations of the factors just mentioned

for index, row in solution_dem_general.iterrows(): 
    if (row["PHONE_VALIDATED"] == "KO") & (row["PREDICTION_DUAL"] != 1):
        l = [row["ID"],0,1,0,1,0,0,0,0,0,0,0,0]  
        Solution_DEM.loc[len(Solution_DEM)] = l
    elif (row["PHONE_VALIDATED"] == "KO") & (row["PREDICTION_DUAL"] == 1):
        m = [row["ID"],1,0,1,0,0,0,0,0,0,0,0,0]
        Solution_DEM.loc[len(Solution_DEM)] = m
    elif (row["PHONE_VALIDATED"] != "KO") & (row["PREDICTION_DUAL"] == 1):
        n = [row["ID"],0,0,1,0,1,0,0,0,0,0,0,0]
        Solution_DEM.loc[len(Solution_DEM)] = n
    elif (row["PHONE_VALIDATED"] != "KO") & (row["PREDICTION_DUAL"] != 1):
        o = [row["ID"],0,0,0,1,0,0,0,1,0,0,0,0]
        Solution_DEM.loc[len(Solution_DEM)] = o

for index, row in solution_sms_general.iterrows():
    if row["PREDICTION_DUAL"] == 1:
        l = [row["ID"],0,0,1,0,1,0,0,0,0,0,0,0]
        Solution_SMS.loc[len(Solution_SMS)] = l
    elif row["PREDICTION_DUAL"] != 1:
        m = [row["ID"],0,0,0,1,0,0,0,1,0,0,0,0]
        Solution_SMS.loc[len(Solution_SMS)] = m

for index, row in solution_tls_general.iterrows():
    if row["PREDICTION_DUAL"] == 1:
        l = [row["ID"],1,0,0,0,0,0,0,0,0,0,0,0]
        Solution_TLS.loc[len(Solution_TLS)] = l
    elif row["PREDICTION_DUAL"] != 1:
        m = [row["ID"],0,1,0,0,0,0,0,0,0,0,0,0]
        Solution_TLS.loc[len(Solution_TLS)] = m

# finally we export the six datasets as files in csv format 
Cross_Selling_DEM.to_csv("Cross_Selling_DEM.csv")
Cross_Selling_SMS.to_csv("Cross_Selling_SMS.csv")
Cross_Selling_TLS.to_csv("Cross_Selling_TLS.csv")
Solution_DEM.to_csv("Solution_DEM.csv")
Solution_SMS.to_csv("Solution_SMS.csv")
Solution_TLS.to_csv("Solution_TLS.csv")


