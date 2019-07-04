
#IMPORTING ALL THE REQUIRED PACKAGES

import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE, ADASYN   #for oversampling minority dataset
#Dependencies for imblearn package -> https://pypi.org/project/imbalanced-learn/

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_recall_curve,roc_auc_score
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score,classification_report,auc
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score
from sklearn.cross_validation import KFold, cross_val_score,train_test_split
from sklearn.preprocessing import StandardScaler as ss
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score

#Read the Dataset from a csv file

data = pd.read_csv("Desktop/new.csv")
data.head()

sns.countplot(x='SAR_FLG', data=data)



X = data.iloc[:,0:50]
y = data.iloc[:,54]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y)

ad = ADASYN()    
X_ad, y_ad = ad.fit_sample(X_train, y_train)

X_ad = pd.DataFrame(data = X_ad, columns = X_train.columns)

print(X_train.shape)
print(X_ad.shape)


print("Number transactions X_train dataset: ", len(X_train))
print("Number transactions X_test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))

X_train_sampled, X_test_sampled, y_train_sampled, y_test_sampled = train_test_split(X_ad,y_ad,test_size = 0.3,random_state = 0)
print("")
print("Number transactions y_train dataset: ", len(X_train_sampled))
print("Number transactions y_test dataset: ", len(X_test_sampled))
print("Total number of transactions: ", len(X_train_sampled)+len(X_test_sampled))

X_train_sampled_df = pd.DataFrame(X_train_sampled)
y_train_sampled_df = pd.DataFrame(y_train_sampled)
X_test_sampled_df = pd.DataFrame(X_test_sampled)
y_test_sampled_df = pd.DataFrame(y_test_sampled)



Source_data_no_fraud_count = len(data[data.ATM_TRXN_OUT_CT_JMP_P100____	==0])
Source_data_fraud_count = len(data[data.ATM_TRXN_OUT_CT_JMP_P100____	!=0])
print('Percentage of counts in original dataset:{}%'.format((Source_data_fraud_count*100)/(Source_data_no_fraud_count+Source_data_fraud_count)))

Sampled_data_no_fraud_count = len(X_ad[y_ad==0])
Sampled_data_fraud_count = len(X_ad[y_ad==1])
print('Percentage of counts in the new data:{}%'.format((Sampled_data_fraud_count*100)/(Sampled_data_no_fraud_count+Sampled_data_fraud_count)))




#Function to find best C-parameter for Logistic regression model

def printing_Kfold_scores(x_train_data,y_train_data):
    fold = KFold(len(y_train_data),5,shuffle=False) 

    c_param_range = [100,75,50,25,10]

    results_table = pd.DataFrame(index = range(len(c_param_range),2), columns = ['C_parameter','Mean recall score'])
    results_table['C_parameter'] = c_param_range

    j = 0
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('C PARAMETER: ', c_param)
        print('-------------------------------------------')
        print('')

        recall_accs = []
        for iteration, indices in enumerate(fold,start=1):

            
            lr = LogisticRegression(C = c_param, penalty = 'l1')

            
            lr.fit(x_train_data.iloc[indices[0],:],y_train_data.iloc[indices[0],:].values.ravel())

            
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1],:].values)

           
            recall_acc = recall_score(y_train_data.iloc[indices[1],:].values,y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration ', iteration,': recall score = ', recall_acc*100)

        
        results_table.ix[j,'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs)*100)
        print('')
        
        
    results_table['Mean recall score']=results_table['Mean recall score'].astype('float64')
    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
    
    
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')
    
    return best_c
best_c = printing_Kfold_scores(X_train_sampled_df,y_train_sampled_df)




#XGB CLASSIFIER MODEL

model = XGBClassifier()
kfold = StratifiedKFold(n_splits=2, random_state=1)
model.fit(X_train_sampled_df, y_train_sampled_df)

scoring = 'roc_auc'
results = cross_val_score(model, X_train_sampled_df, y_train_sampled_df, cv=kfold, scoring = scoring)
print( "AUC: %.3f  ,  STD: %.3f" % (results.mean(), results.std()) )




#Receiver Operating Characteristic (ROC) Plot for XGB classifier model


fig_size = plt.rcParams["figure.figsize"] 

old_fig_params = fig_size
fig_size[0] = 7
fig_size[1] = 5
   
plt.rcParams["figure.figsize"] = fig_size 


mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
lw = 2

i = 0
for (train, test), color in zip(kfold.split(X_train_sampled_df, y_train_sampled_df), colors):
    probas_ = model.fit(X_train_sampled_df, y_train_sampled_df).predict_proba(X_train_sampled_df)
    fpr, tpr, thresholds = roc_curve(y_train_sampled_df, probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=color,
             label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
         label='Luck')

mean_tpr /= kfold.get_n_splits(X_train_sampled_df, y_train_sampled_df)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()




#RANDOMFOREST CLASSIFIER MODEL

model = RandomForestClassifier()
kfold = StratifiedKFold(n_splits=2, random_state=1)
model.fit(X_train_sampled_df, y_train_sampled_df)

scoring = 'roc_auc'
results = cross_val_score(model, X_train_sampled_df, y_train_sampled_df, cv=kfold, scoring = scoring)
print( "AUC: %.3f  ,  STD: %.3f" % (results.mean(), results.std()) )




#Receiver Operating Characteristic (ROC) Plot for Randomforest classifier model

fig_size = plt.rcParams["figure.figsize"] 

old_fig_params = fig_size
fig_size[0] = 7
fig_size[1] = 5
   
plt.rcParams["figure.figsize"] = fig_size 


mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
lw = 2

i = 0
for (train, test), color in zip(kfold.split(X_train_sampled_df, y_train_sampled_df), colors):
    probas_ = model.fit(X_train_sampled_df, y_train_sampled_df).predict_proba(X_train_sampled_df)
    fpr, tpr, thresholds = roc_curve(y_train_sampled_df, probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=color,
             label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
         label='Luck')

mean_tpr /= kfold.get_n_splits(X_train_sampled_df, y_train_sampled_df)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()




#LOGISTIC REGRESSION MODEL 

model = LogisticRegression()
kfold = StratifiedKFold(n_splits=2, random_state=1)
model.fit(X_train_sampled_df, y_train_sampled_df)

scoring = 'roc_auc'
results = cross_val_score(model, X_train_sampled_df, y_train_sampled_df, cv=kfold, scoring = scoring)
print( "AUC: %.3f  ,  STD: %.3f" % (results.mean(), results.std()) )




#Receiver Operating Characteristic (ROC) Plot for Logistic Reression classifier model

fig_size = plt.rcParams["figure.figsize"] 

old_fig_params = fig_size
fig_size[0] = 7
fig_size[1] = 5
   
plt.rcParams["figure.figsize"] = fig_size 


mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
lw = 2

i = 0
for (train, test), color in zip(kfold.split(X_train_sampled_df, y_train_sampled_df), colors):
    probas_ = model.fit(X_train_sampled_df, y_train_sampled_df).predict_proba(X_train_sampled_df)
    fpr, tpr, thresholds = roc_curve(y_train_sampled_df, probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=color,
             label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
         label='Luck')

mean_tpr /= kfold.get_n_splits(X_train_sampled_df, y_train_sampled_df)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()




# CONSOLIDATED VALUES FOR DIFFERENT MODELS

###########################################################################################################################

# WITH SMOTE TECHNIQUE

#                                  AUC VALUES              MEAN ROC VALUES
# 1)LOGISTIC REGRESSION               0.850                      0.903
# 2)XGBOOST                           0.912                     0.951
# 3)RANDOMFOREST                      0.901                     0.960

###########################################################################################################################

# WITH ADASYN TECHNIQUE

#                                  AUC VALUES              MEAN ROC VALUES
# 1)LOGISTIC REGRESSION               0.814                     0.821
# 2)XGBOOST                           0.917                     0.972
# 3)RANDOMFOREST                      0.927                     0.984

###########################################################################################################################
# BEST MODEL- RANDOMFOREST CLASSIFIER WITH ADASYN OVERSAMPLING TECHNIQUE
###########################################################################################################################

