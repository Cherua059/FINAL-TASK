#-----------------provided in .txt file------------------------

# Attributes for student-math.csv (Math course) dataset:
# 1 school - student's school (binary: "GP" - Gabriel Pereira or "MS" - Mousinho da Silveira)
# 2 sex - student's sex (binary: "F" - female or "M" - male)
# 3 age - student's age (numeric: from 15 to 22)
# 4 address - student's home address type (binary: "U" - urban or "R" - rural)
# 5 famsize - family size (binary: "LE3" - less or equal to 3 or "GT3" - greater than 3)
# 6 Pstatus - parent's cohabitation status (binary: "T" - living together or "A" - apart)
# 7 Medu - mother's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
# 8 Fedu - father's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
# 9 Mjob - mother's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
# 10 Fjob - father's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
# 11 reason - reason to choose this school (nominal: close to "home", school "reputation", "course" preference or "other")
# 12 guardian - student's guardian (nominal: "mother", "father" or "other")
# 13 traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
# 14 studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
# 15 failures - number of past class failures (numeric: n if 1<=n<3, else 4)
# 16 schoolsup - extra educational support (binary: yes or no)
# 17 famsup - family educational support (binary: yes or no)
# 18 paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
# 19 activities - extra-curricular activities (binary: yes or no)
# 20 nursery - attended nursery school (binary: yes or no)
# 21 higher - wants to take higher education (binary: yes or no)
# 22 internet - Internet access at home (binary: yes or no)
# 23 romantic - with a romantic relationship (binary: yes or no)
# 24 famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
# 25 freetime - free time after school (numeric: from 1 - very low to 5 - very high)
# 26 goout - going out with friends (numeric: from 1 - very low to 5 - very high)
# 27 Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 28 Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 29 health - current health status (numeric: from 1 - very bad to 5 - very good)
# 30 absences - number of school absences (numeric: from 0 to 93)
# 31 G1 - first period grade (numeric: from 0 to 20)
# 32 G2 - second period grade (numeric: from 0 to 20)
# 33 G3 - final grade (numeric: from 0 to 20)

#_________________________________CODE____________________________________

#importing the libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

#_________________________________PART1___________________________________

#changing the working directory to current direntory, where the code and the .csv files are present
os.chdir(os.path.dirname(os.path.abspath("__file__")));

#reading the data and creating a datframe
data = pd.read_csv(r"student-math.csv",sep=';',quotechar='"');
dataframe = pd.DataFrame(data,columns=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3']);


#label_encoding of columns with nominal type values
le = LabelEncoder(); #creates an instance of the label encoder function
dataframe.school = le.fit_transform(dataframe.school);
dataframe.sex = le.fit_transform(dataframe.sex);
dataframe.address = le.fit_transform(dataframe.address);
dataframe.famsize = le.fit_transform(dataframe.famsize);
dataframe.Pstatus = le.fit_transform(dataframe.Pstatus);
dataframe.Mjob = le.fit_transform(dataframe.Mjob);
dataframe.Fjob = le.fit_transform(dataframe.Fjob);
dataframe.reason = le.fit_transform(dataframe.reason);
dataframe.guardian = le.fit_transform(dataframe.guardian);
dataframe.schoolsup = le.fit_transform(dataframe.schoolsup);
dataframe.famsup = le.fit_transform(dataframe.famsup);
dataframe.paid = le.fit_transform(dataframe.paid);
dataframe.activities = le.fit_transform(dataframe.activities);
dataframe.nursery = le.fit_transform(dataframe.nursery);
dataframe.higher = le.fit_transform(dataframe.higher);
dataframe.internet = le.fit_transform(dataframe.internet);
dataframe.romantic = le.fit_transform(dataframe.romantic);

#creating a new column which is the sum of G1,G2 and G3 elements
final_grade = [];
for i in range(0,len(data)):
    sum = data.G1[i] + data.G2[i] + data.G3[i];
    final_grade.append(sum);
    sum=0;
dataframe['final_grade'] = final_grade;

#assigning input and output features
X = dataframe.drop(['G3','final_grade'],axis=1); #input
Y = dataframe['final_grade']; #output

#one hot encoding to avoid ranking of values
# ohe = OneHotEncoder(categories='auto'); #creates an instance of the one hot encoder function
# X = ohe.fit_transform(X).toarray();

#-------------------------------------------------------------------------------------
#spliting the dataset. 10% = testing data, 90% = training data 
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.1);

#____________________PART2______________________

alg = linear_model.LinearRegression() #creates an instance of the linear regressiion function
alg.fit(x_train,y_train); #fitting the model with training data
pred = alg.predict(x_test); #predicting the values with testing data
mse = mean_squared_error(y_test,pred);

#scatter plot between true vale and predicted value
plt.scatter(pred,y_test);
plt.xlabel('PREDICTED VALUE');
plt.ylabel('TRUE VALUE');
plt.legend(loc='best');

#printing the results
print('mean squared error = ',mse)
print('TRAIN SCORE = ',alg.score(x_train,y_train))
print('TEST SCORE = ',alg.score(x_test,y_test))
print('Difference = ',alg.score(x_train,y_train)-alg.score(x_test,y_test))
#-----------------------------------------------------------------------------------------

#-----bakward elimination----
# SIGNIFICANCE LEVEL = 0.05
l = len(dataframe);
X1 = np.append(arr=np.ones((l,1)).astype(int),values=X,axis=1)

#FIRST ITERATION
X_opt = X1[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 4

#SECOND ITERATION
X_opt = X1[:,[0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 5

#THIRD ITERATION
X_opt = X1[:,[0,1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 21

#FOURTH ITERATION
X_opt = X1[:,[0,1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 9

#FIFTH ITERATION
X_opt = X1[:,[0,1,2,3,6,7,8,10,11,12,13,14,15,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 26

#SIXTH ITERATION
X_opt = X1[:,[0,1,2,3,6,7,8,10,11,12,13,14,15,16,17,18,19,20,22,23,24,25,27,28,29,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 18

#SEVENTH ITERATION
X_opt = X1[:,[0,1,2,3,6,7,8,10,11,12,13,14,15,16,17,19,20,22,23,24,25,27,28,29,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 6

#EIGHT ITERATION
X_opt = X1[:,[0,1,2,3,7,8,10,11,12,13,14,15,16,17,19,20,22,23,24,25,27,28,29,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 12

#NINTH ITERATION
X_opt = X1[:,[0,1,2,3,7,8,10,11,13,14,15,16,17,19,20,22,23,24,25,27,28,29,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 25

#TENTH ITERATION
X_opt = X1[:,[0,1,2,3,7,8,10,11,13,14,15,16,17,19,20,22,23,24,27,28,29,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 13

#ELEVENTH ITERATION
X_opt = X1[:,[0,1,2,3,7,8,10,11,14,15,16,17,19,20,22,23,24,27,28,29,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 2

#TWELVETH ITERATION
X_opt = X1[:,[0,1,3,7,8,10,11,14,15,16,17,19,20,22,23,24,27,28,29,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 22

#THIRTEENTH ITERATION
X_opt = X1[:,[0,1,3,7,8,10,11,14,15,16,17,19,20,23,24,27,28,29,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 20

#FOURTEENTH ITERATION
X_opt = X1[:,[0,1,3,7,8,10,11,14,15,16,17,19,23,24,27,28,29,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 11

#FIFTEENTH ITERATION
X_opt = X1[:,[0,1,3,7,8,10,14,15,16,17,19,23,24,27,28,29,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 17

#SIXTEENTH ITERATION
X_opt = X1[:,[0,1,3,7,8,10,14,15,16,19,23,24,27,28,29,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 27

#SEVENTEENTH ITERATION
X_opt = X1[:,[0,1,3,7,8,10,14,15,16,19,23,24,28,29,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 29

#EIGHTEENTH ITERATION
X_opt = X1[:,[0,1,3,7,8,10,14,15,16,19,23,24,28,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 14

#NINETEENTH ITERATION
X_opt = X1[:,[0,1,3,7,8,10,15,16,19,23,24,28,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 10

#TWENTIETH ITERATION
X_opt = X1[:,[0,1,3,7,8,15,16,19,23,24,28,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 7

#21ST ITERATION
X_opt = X1[:,[0,1,3,8,15,16,19,23,24,28,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 8

#22ND ITERATION
X_opt = X1[:,[0,1,3,15,16,19,23,24,28,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 16

#23RD ITERATION
X_opt = X1[:,[0,1,3,15,19,23,24,28,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 15

#24TH ITERATION
X_opt = X1[:,[0,1,3,19,23,24,28,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 1

#25TH ITERATION
X_opt = X1[:,[0,3,19,23,24,28,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 23

#26TH ITERATION
X_opt = X1[:,[0,3,19,24,28,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 28

#27TH ITERATION
X_opt = X1[:,[0,3,19,24,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#P>SL for column 19

#28TH ITERATION
X_opt = X1[:,[0,3,24,30,31,32]];
regressor_ols = sm.OLS(endog=Y,exog=X_opt).fit();
print(regressor_ols.summary());
#FINAL ITERATION AS ALL P VALUES ARE LESS THAN SL.
#age,famrel,absences,g1,g2 are the features with maximum impact.

#fitting the model with updated features
X_opt = X1[:,[3,24,30,31,32]];#selecting features from the original input feature set(ignoring the first)

#train,test,split the data with the chosen features
x_train1,x_test1,y_train1,y_test1 = train_test_split(X_opt,Y,test_size=0.1);
alg.fit(x_train1,y_train1); #fitting the model with training data
pred1 = alg.predict(x_test1); #predicting the values with testing data
mse1 = mean_squared_error(y_test1,pred1);

#printing the values
print('mean squared error = ',mse1);
print('TRAIN SCORE = ',alg.score(x_train1,y_train1));
print('TEST SCORE = ',alg.score(x_test1,y_test1));
print('Difference = ',alg.score(x_train1,y_train1)-alg.score(x_test1,y_test1));

#----xxxxxx---------end of task----------xxxx

#--------OTHER MODELS AND ALGORITHMS--------
#1.LOGISTIC REGRESSION
xtr,xts,ytr,yts = train_test_split(X_opt,Y,test_size=0.1);
from sklearn.linear_model import LogisticRegression
alg1 = LogisticRegression(solver='liblinear')
alg1.fit(xtr,ytr)
pred3 = alg1.predict(xts)
alg1.score(xts,yts)

#2.DECISION TREE REGRESSION
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor() 
xtr,xts,ytr,yts = train_test_split(X_opt,Y,test_size=0.1);
regressor.fit(xtr, ytr);
pred_reg=regressor.predict(xts)
regressor.score(xts,yts)

#3.RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
regressor1 = RandomForestRegressor(n_estimators = 100)
xtr,xts,ytr,yts = train_test_split(X_opt,Y,test_size=0.1);
regressor1.fit(xtr, ytr);
pred_reg1=regressor1.predict(xts)
regressor1.score(xts,yts)

