# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# %%
data_classification = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'
#wget data_classification
df=pd.read_csv(data_classification)


# %%


# %% [markdown]
# Data cleaning

# %%
print (df.dtypes)

df.columns= df.columns.str.lower().str.replace(" ","_").str.replace("-","_")
for clm in df.dtypes[df.dtypes=='object'].index:
    df[clm]= df[clm].str.lower().str.replace(' ','_').str.replace('-','_')
df[df.isnull()==True].count()




# %%
#replacing errors by null and then by zeros
df.totalcharges=pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges=df.totalcharges.fillna(0)

# %%
'''binary=df.nunique()[df.nunique()==2].index

for cat in binary:
    v=df[cat][0]
    df[cat]=(df[cat]==v).astype('int')

df.head()
df.nunique()
df.multiplelines.unique()'''
df.churn = (df.churn == 'yes').astype(int)
df.iloc[1].to_dict()

# %% [markdown]
# split the data using scikit-learn 

# %%
from sklearn.model_selection import  train_test_split

# %% [markdown]
# set the train/validation/test

# %%
df_train_full, df_test= train_test_split(df,test_size=0.2,random_state=1)
df_train, df_validation= train_test_split(df_train_full,test_size=0.25,random_state=1)

df_train.reset_index(drop=True)
df_validation.reset_index(drop=True)
df_test.reset_index(drop=True)


# %%
y_train=df_train.churn.values
y_train_full=df_train_full.churn.values
y_validation=df_validation.churn.values
y_test=df_test.churn.values

del (df_train['churn'])
del (df_test['churn'])
del (df_validation['churn'])

# %% [markdown]
# Exploratory data analysis EDA

# %%
df_train_full.churn.value_counts(normalize=True)
global_churn_rate=df_train_full.churn.mean().round(2)
global_churn_rate


# %%
print(df_train_full.columns)
numerical=['tenure','monthlycharges','totalcharges']
categorical=[ 'gender', 'seniorcitizen', 'partner', 'dependents',
     'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']

# %%
for c in categorical:
    df_group=df_train_full.groupby(c).churn.agg(['mean','count'])
    df_group['diff']=df_group['mean']-global_churn_rate
    df_group['risk']=df_group['mean']/global_churn_rate
    print(df_group) 

# %% [markdown]
# Importance of features 

# %%
from sklearn.metrics import mutual_info_score

# %%

def mutual_info_churn_score(series):
    return mutual_info_score(series,df_train_full.churn)
mi = df_train_full[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending=False)
mi
#

# %% [markdown]
# Correlation between numerical variables and churn rate.

# %%
print(numerical)
df_train_full[numerical].corrwith(df_train_full.churn)

# %% [markdown]
# tenure is important 

# %%
df_train_full[df_train_full.tenure<3].churn.mean()

# %% [markdown]
# One-Hot Encoding

# %%
from sklearn.feature_extraction import DictVectorizer

# %%
train_dicts =df_train[categorical+numerical].to_dict(orient='records')
dv=DictVectorizer(sparse=False)
dv.fit(train_dicts) #show him how the data look like 
x_train=dv.transform(train_dicts)

# %%
dv.get_feature_names_out()

# %%
x_train.shape

# %%
validation_dicts= df_validation[numerical+categorical].to_dict(orient='records')
x_validation=dv.transform(validation_dicts)
x_validation.shape

# %% [markdown]
# Logistic regression 
# similar to linear regression but the output is between 0 and 1 
# i.e. signoid score 
# 
# y_predict= signoid(w0+X.dot(w) )

# %%
from sklearn.linear_model import LogisticRegression
model =LogisticRegression(max_iter=100000)
model.fit(x_train,y_train)
model.score

# %%
model.intercept_[0] #w0

# %%
model.coef_[0].round(3)#w

# %%
model.predict(x_train)# hard prediction


# %%
y_predict=model.predict_proba(x_validation)[:,1]# soft predictions

# %%


# %% [markdown]
# 

# %% [markdown]
# accuracy of the model

# %%
#accuracy =(y_validation==churn_decision.astype(int)).mean()
#accuracy

# %% [markdown]
# do the same to test the model

# %%
full_train_dict= df_train_full.to_dict(orient='records')
x_full_train=dv.transform(full_train_dict)
model.fit(x_full_train,y_train_full)

# %%
test_dict= df_test.to_dict(orient='records')
x_test=dv.transform(test_dict)
test_predict=model.predict(x_test)

(y_test==test_predict).mean()


# %% [markdown]
# Confusion matrix 

# %%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, test_predict)
print("Confusion Matrix:\n", cm)
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# %%
from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_test, test_predict, average='binary')
recall = recall_score(y_test, test_predict, average='binary')

print("Precision:", precision)
print("Recall:", recall)

# %% [markdown]
# ROC curves 

# %%
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_validation, y_predict)

# %%
plt.figure(figsize=(5, 5))

plt.plot(fpr, tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()

# %% [markdown]
# cross validation 

# %%
from sklearn.model_selection import KFold

# %%
from tqdm.auto import tqdm

# %%
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=10000)
    model.fit(X_train, y_train)
    
    return dv, model

# %%


# %%
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

# %%
from sklearn.metrics import roc_auc_score
n_splits = 5

for C in tqdm([0.001, 0.01, 0.1, 0.5, 1]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores = []

    for train_idx, val_idx in kfold.split(df_train_full):
        df_train = df_train_full.iloc[train_idx]
        df_val = df_train_full.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

# %% [markdown]
# Deployement of a model

# %%
import pickle

# %%
output_file=f'model_C={C}.bin'

# %%
f_out=open(output_file,'wb')
pickle.dump((dv,model),f_out)
f_out.close()

 

