# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 20:11:44 2021

@author: lord of music
"""
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score
######################################
#####################################
#-----READING DATA---------
boston=load_boston()
boston.keys()
description=str(boston.DESCR)
data=pd.DataFrame(data=boston.data,columns=boston.feature_names)
data['price']=boston.target

###################################
##################################
#------UNDERSTANDING THE DATA --------

data.info()
data.isna().sum()
#######################################
######################################
#----------DATA VISUALISATION------------
sns.pairplot(data)
#----DATA DISTRIBUTION ---------
#Plotting the data distrubution in 2 rows and 7 columns
rows=2
cols=7
fig,ax=plt.subplots(nrows=rows,ncols=cols,figsize=(20,5))
index=0
columns=data.columns
for i in range(rows):
    for j in range(cols):
        sns.distplot(data[columns[index]],ax=ax[i][j])
        index+=1
plt.tight_layout()

#######################

fig,ax=plt.subplots(figsize=(20,10))
sns.heatmap(data.corr(),annot=True,annot_kws={'size':4})


#Cleaning data based on distribution
data=data.drop(['CRIM','ZN','CHAS','B'],axis=1)
cormat=data.corr()
 
def getCorrolatedFeatures(corrdata,corrvalue):
    features=[]
    values=[]
    for i, index in enumerate(corrdata.index): #(indexCol ,nameCol)
        if (abs(corrdata[index])>corrvalue):
            features.append(index)
            values.append(corrdata[index])
    df=pd.DataFrame(data=values,index=features,columns=['Corr Value'])
    return df


############################################
############################################

corrvalue=0.2
corrData=getCorrolatedFeatures(cormat['price'],corrvalue)
corrolated_data=data[corrData.index]
sns.pairplot(corrolated_data)
fig,ax=plt.subplots(figsize=(20,10))
sns.heatmap(corrolated_data.corr(),annot=True)


##################################
#################################
#Shuffle and Splitting data 

X=corrolated_data.drop('price',axis=1).values
y=corrolated_data['price'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#Training the model

regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
score_non_opt=r2_score(y_test,y_pred)

############################"



def BackwardElimination(X,y):
    import statsmodels.api as stm
    import numpy as np
    
    X=np.append(arr=np.ones((X.shape[0],1)).astype(int),values=X,axis=1)
    ColumnIndexList=[i for i in range(X.shape[1])]
    X_opt=X[:,ColumnIndexList]
    #Step2 : Fit the model with all possible predictors
    regressor_opt=stm.OLS(y,X_opt).fit()
    #Step3:Consider the prdictor with the highest P value if P>sl=0.05 Go to step 4 else Finish 
    pValuesList=list(regressor_opt.pvalues)
   
    while(True):
        verification=False
        for e in pValuesList:
            if e>0.05 :   
                ColumnIndexList.pop(pValuesList.index(e))
                pValuesList.remove(e)
                X_opt=X[:,ColumnIndexList]
                regressor_opt=stm.OLS(y,X_opt).fit()
            else :
                continue
        pValuesList=list(regressor_opt.pvalues)
        for e in pValuesList:
            if e>0.05 :
                verification=True
        if verification==False :
            break 
        else :
            continue
    return(pValuesList,regressor_opt)
pv,regressor_opt=BackwardElimination(X,y)
y_opt=regressor_opt.predict()
regressor_opt.summary()