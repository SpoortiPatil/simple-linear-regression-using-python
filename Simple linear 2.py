
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
sal = pd.read_csv("Salary_Data.csv")
sal.columns

# Renaming the columns for convinience
sal= sal.rename(columns= {"YearsExperience" : "yrsexp", "Salary" : "slry"})
sal.columns

# Graphical exploration of data
plt.hist(sal.yrsexp)
plt.boxplot(sal.slry)
plt.hist(sal.slry)
plt.boxplot(sal.slry)

# Calculation of skewness and kurtosis to check the type of distribution
sal.kurt()
sal.skew()

plt.plot(sal.yrsexp, sal.slry, "ro"); plt.xlabel("yers exp"); plt.ylabel("salry")

sal.slry.corr(sal.yrsexp)           # checking the correlation between the dependent and independent variables

# For preparing linear regression model importing the statsmodels.formula.api
import statsmodels.formula.api as smf
model= smf.ols("slry ~ yrsexp", data = sal).fit()
model.params
model.summary()

print(model.conf_int(0.05))

# Predicting the values using the model
pred= model.predict(sal.yrsexp)
pred

import matplotlib.pylab as plt
plt.scatter(sal['yrsexp'],sal['slry'],color='red');plt.plot(sal['yrsexp'],pred,color='black');plt.xlabel('yrs of exp');plt.ylabel('salry')

pred.corr(sal.slry)

# bulinding another model(exponential) to check whether the R_square value can be increased
model2 = smf.ols("slry ~ np.log(yrsexp)", data= sal).fit()
model2.params
model2.summary()

print(model.conf_int(0.05))

pred2= model2.predict(sal.yrsexp)
pred2

plt.scatter(sal.yrsexp, sal.slry, c="b"); plt.plot(sal.yrsexp, pred2, c="r");plt.xlabel("Years of experience");plt.ylabel("Salary")

# Predicting the values using the model
pred2.corr(sal.slry)

# bulinding another model(exponential) to check whether the R_square value can be increased
model3= smf.ols('np.log(slry)~ yrsexp', data=sal).fit()
model3.params
model3.summary()

print(model.conf_int(0.05))

# Predicting the values using the model
predlog= model3.predict(sal.yrsexp)
predlog
pred3= np.exp(predlog)
pred3

plt.scatter(sal.yrsexp, sal.slry, c="b"); plt.plot(sal.yrsexp, pred3, c="r"); plt.xlabel("yrs exp"); plt.ylabel("salry")

pred3.corr(sal.slry)

sal["yrsexp_sq"]= sal.yrsexp*sal.yrsexp

# bulinding another model(quadratic) to check whether the R_square value can be increased
model4= smf.ols('np.log(slry)~yrsexp+yrsexp_sq', data= sal).fit()
model4.params
model4.summary()

print(model4.conf_int(0.05))

# Predicting the values using the model
pred_quad= model4.predict(sal)
pred4= np.exp(pred_quad)
pred4

plt.scatter(sal.yrsexp, sal.slry, c="b"); plt.plot(sal.yrsexp,pred4, c="r"); plt.xlabel("yrs exp"); plt.ylabel("salry")

pred4.corr(sal.slry)

# Creating the table of models and thier R_square values
data = {"Model":pd.Series(["model1_linear","model2_exponential","model3_exponential","model4_quadratic"]),"R_square_values":(0.957,0.854,0.932,0.949)}

table_rsquare = pd.DataFrame(data)
table_rsquare


# Out of all the 4 models, last model (quadratic_model) has the highest R-square value, hence it is the best fit model
std_resid = model.resid_pearson       # Calculating the standard residuals of the best model
std_resid

plt.plot(std_resid, "ro");plt.axhline(y=0, color="blue");plt.xlabel("Observation number");plt.ylabel("Standard residuals")

plt.scatter(pred, sal.slry, color="blue"); plt.xlabel("Predicted"); plt.ylabel("Actual")

plt.hist(std_resid)
