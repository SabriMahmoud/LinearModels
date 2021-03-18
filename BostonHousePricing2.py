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




##################################
#################################
#Shuffle and Splitting data 

X=data.drop('price',axis=1).values
y=data['price'].values
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