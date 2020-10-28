# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 13:14:24 2020

@author: hp user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sal = pd.read_csv("Salary_Data.csv")
sal.columns

sal= sal.rename(columns= {"YearsExperience" : "yrsexp", "Salary" : "slry"})
sal.columns

plt.hist(sal.yrsexp)
plt.boxplot(sal.slry)
plt.hist(sal.slry)
plt.boxplot(sal.slry)

sal.kurt()
sal.skew()

plt.plot(sal.yrsexp, sal.slry, "ro"); plt.xlabel("yers exp"); plt.ylabel("salry")

sal.slry.corr(sal.yrsexp)

import statsmodels.formula.api as smf
model= smf.ols("slry ~ yrsexp", data = sal).fit()
model.params
model.summary()

print(model.conf_int(0.05))

pred= model.predict(sal.yrsexp)
pred

import matplotlib.pylab as plt
plt.scatter(sal['yrsexp'],sal['slry'],color='red');plt.plot(sal['yrsexp'],pred,color='black');plt.xlabel('yrs of exp');plt.ylabel('salry')

pred.corr(sal.slry)

model2 = smf.ols("slry ~ np.log(yrsexp)", data= sal).fit()
model2.params
model2.summary()

print(model.conf_int(0.05))

pred2= model2.predict(sal.yrsexp)
pred2

plt.scatter(sal.yrsexp, sal.slry, c="b"); plt.plot(sal.yrsexp, pred2, c="r");plt.xlabel("Years of experience");plt.ylabel("Salary")

pred2.corr(sal.slry)

model3= smf.ols('np.log(slry)~ yrsexp', data=sal).fit()
model3.params
model3.summary()

print(model.conf_int(0.05))

predlog= model3.predict(sal.yrsexp)
predlog
pred3= np.exp(predlog)
pred3

plt.scatter(sal.yrsexp, sal.slry, c="b"); plt.plot(sal.yrsexp, pred3, c="r"); plt.xlabel("yrs exp"); plt.ylabel("salry")

pred3.corr(sal.slry)

sal["yrsexp_sq"]= sal.yrsexp*sal.yrsexp

model4= smf.ols('np.log(slry)~yrsexp+yrsexp_sq', data= sal).fit()
model4.params
model4.summary()

print(model4.conf_int(0.05))

pred_quad= model4.predict(sal)
pred4= np.exp(pred_quad)
pred4

plt.scatter(sal.yrsexp, sal.slry, c="b"); plt.plot(sal.yrsexp,pred4, c="r"); plt.xlabel("yrs exp"); plt.ylabel("salry")

pred4.corr(sal.slry)

data = {"Model":pd.Series(["model1_linear","model2_exponential","model3_exponential","model4_quadratic"]),"R_square_values":(0.957,0.854,0.932,0.949)}

table_rsquare = pd.DataFrame(data)
table_rsquare

std_resid = model.resid_pearson       # Calculating the standard residuals of the best model
std_resid

plt.plot(std_resid, "ro");plt.axhline(y=0, color="blue");plt.xlabel("Observation number");plt.ylabel("Standard residuals")

plt.scatter(pred, sal.slry, color="blue"); plt.xlabel("Predicted"); plt.ylabel("Actual")

plt.hist(std_resid)
