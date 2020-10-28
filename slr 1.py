
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

deli = pd.read_csv("delivery.csv")
deli.columns
deli= deli.rename(columns = {"Deliverytime" : "deltime", "Sorting " : "sort"})
deli.columns

plt.hist(deli.deltime)
plt.boxplot(deli.deltime)

plt.hist(deli.sort)
plt.boxplot(deli.sort)

deli.skew()
deli.kurt()

plt.plot(deli.sort, deli.deltime, "bo"); plt.xlabel("Sorting time"); plt.ylabel("Delivery time")

deli.deltime.corr(deli.sort)

import statsmodels.formula.api as smf
model= smf.ols("deltime ~ sort", data = deli).fit()
model.params
model.summary()

print(model.conf_int(0.05))

pred= model.predict(deli.sort)
pred

error_model= deli.sort
import matplotlib.pylab as plt
plt.scatter(x=deli['sort'],y=deli['deltime'],color='red');plt.plot(deli['sort'],pred,color='black');plt.xlabel('Delivery time');plt.ylabel('Sorting time')

pred.corr(deli.deltime)

model2= smf.ols('deltime~np.log(sort)', data=deli).fit()
model2.params
model2.summary()

print(model2.conf_int(0.05))

pred2= model2.predict(deli.sort)
pred2

plt.scatter(deli.sort, deli.deltime, c="b"); plt.plot(deli.sort,pred2, c="r"); plt.xlabel("delivery time"); plt.ylabel("sorting time")
pred2.corr(deli.deltime)

model3= smf.ols('np.log(deltime)~sort', data= deli).fit()
model3.params
model3.summary()

print(model3.conf_int(0.05))

pred3= model3.predict(deli.sort)
pred3
pred3= np.exp(pred3)
pred3

plt.scatter(deli.sort,deli.deltime, c="b"); plt.plot(deli.sort,pred3, c="r"); plt.xlabel("delivery time"); plt.ylabel("sorting time")
pred3.corr(deli.deltime)

deli["sort_sq"] = deli.sort*deli.sort

model4= smf.ols('np.log(deltime)~sort+sort_sq', data=deli).fit()
model4.params
model4.summary()

print(model4.conf_int(0.05))

pred4= model4.predict(deli)
pred4
pred4= np.exp(pred4)
pred4

plt.scatter(deli.sort, deli.deltime, c="r"); plt.plot(deli.sort,pred4, c="b");plt.xlabel("delivery time"); plt.ylabel("sorting time")
pred4.corr(deli.deltime)


data = {"Model":pd.Series(["model1_linear","model2_exponential","model3_exponential","model4_quadratic"]),"R_square_values":(0.682,0.695,0.711,0.765)}


table_rsquare = pd.DataFrame(data)
table_rsquare

student_resid = model4.resid_pearson 
student_resid




plt.plot(student_resid, "o"); plt.axhline(y=0, color="red");plt.xlabel("Observation number");plt.ylabel("Standard residuals")


plt.scatter(pred4, deli.deltime, c="b");plt.xlabel("Predicted");plt.ylabel("Actual")

plt.hist(student_resid)
