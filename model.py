# -*- coding: utf-8 -*-
"""
Created on Wed May 27 09:42:11 2020

@author: SSN
"""

import pandas as pd

data = pd.read_csv('hiring.csv')


x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.linear_model import LinearRegression
regression = LinearRegression()



regression.fit(x,y)

regression.predict([[3,6,9]])

import pickle
pickle.dump(regression, open('model.pkl','wb'))
##########################

model = pickle.load(open('model.pkl','rb'))

model.predict([[3,6,9]])









