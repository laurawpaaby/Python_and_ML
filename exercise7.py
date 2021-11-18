#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:23:23 2021

@author: laura
"""
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
import scipy as sp
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#%%

# Exercises and objectives

# 1) Estimate bias and variance based on a true underlying function  
# 2) Fitting training data and applying it to test sets with and without regularization  

# For each question and sub-question, please indicate one of the three following answers:  
#     i. I understood what was required of me  
#     ii. I understood what was required of me, but I did not know how to fulfil the requirement  
#     iii. I did not understand what was required of me  


#%%

# # EXERCISE 1 - Estimate bias and variance based on a true underlying function  

# We can express regression as $y = f(x) + \epsilon$ with $E[\epsilon] = 0$ and $var(\epsilon) = \sigma^2$ ($E$ means expected value)  
# For a given point: $x_0$, we can decompose the expected prediction error , $E[(y_0 - \hat{f}(x_0))^2]$ into three parts - __bias__, __variance__ and __irreducible error__ (the first two together are the __reducible error__):

# The expected prediction error is, which we also call the __Mean Squared Error__:  
# $E[(y_0 - \hat{f}(x_0))^2] =  bias(\hat{f}(x_0))^2 + var(\hat{f}(x_0)) + \sigma^2$
  
# where __bias__ is;
  
# $bias(\hat{f}(x_0)) = E[\hat{f}(x_0)] - f(x_0)$

#%%
# 1) Create a function, $f(x)$ that squares its input. This is our __true__ function  
def our_sqrt(x):
    """this functions take the sqrt of the input"""
    return x**2


#     i. generate data, $y$, based on an input range of [0, 6] with a spacing of 0.1. Call this $x$
x = np.arange(0,6, 0.1)
y_true = our_sqrt(x)



#     ii. add normally distributed noise to $y$ with $\sigma=5$ (set a seed to 7 `np.random.seed(7)`) to $y$ and call it $y_{noisy}$
noise_vectors = [] #making the empty vector 
for i in range(6):  
   np.random.seed(7) 
   noise_vector = np.random.normal(loc=0, scale =5, size = 60) #making a list of normal distributed numbers loc=mean, scale=sd
   noise_vectors.append(noise_vector)


y_noise = y_true + noise_vector



#     iii. plot the true function and the generated points  
plt.figure()
plt.plot(x, y_true, 'r-')
plt.plot(x, y_noise, 'bo')
plt.legend(['true_data', 'noisy_data'])
plt.show()



#%%
# 2) Fit a linear regression using `LinearRegression` from `sklearn.linear_model` based on $y_{noisy}$ and $x$ (see code chunk below associated with Exercise 1.2)  
regressor = LinearRegression()
x = x.reshape(60,1) #reshaping x
y_fit_model = regressor.fit(x, y_noise) ## what goes in here?

##predicting new data points on this model:
y_pred = y_fit_model.predict(x)


#     i. plot the fitted line (see the `.intercept_` and `.coef_` attributes of the `regressor` object) on top of the plot (from 1.1.iii)
#instead here i plottet the predicted y-hats from the model (the same think)

plt.figure()
plt.plot(x, y_true, 'r-')
plt.plot(x, y_noise, 'bo')
plt.plot(x, y_pred, 'g-')
plt.legend(['true_data', 'noisy_data', 'fitted_line'])
plt.show()




#%%
#     ii. now run the code chunk below associated with Exercise 1.2.ii - what does X_quadratic amount to?
quadratic = PolynomialFeatures(degree=2)
X_quadratic = quadratic.fit_transform(x.reshape(-1, 1))


reg_1 = LinearRegression()
reg_1.fit(X_quadratic, y_noise) # fit quadratic features

y_quad_hat = reg_1.predict(X_quadratic) # calculate this



#%%

#     iii. do a quadratic and a fifth order fit as well and plot them (on top of the plot from 1.2.i)
five_deg = PolynomialFeatures(degree=5)
X_five = five_deg.fit_transform(x.reshape(-1, 1))

reg_5 = LinearRegression()
reg_5.fit(X_five, y_noise) # fit quadratic features

y_hat_5 = reg_5.predict(X_five) # estimating 

plt.figure()
plt.plot(x, y_true, 'r-')
plt.plot(x, y_noise, 'bo')
plt.plot(x, y_pred, 'g-')
plt.plot(x, y_quad_hat, 'y-')
plt.plot(x, y_hat_5, 'c-')
plt.legend(['true data', 'data with noise', 'fitted line', 'quadratic fit', 'Fifth order poly'])
plt.show()

#%%

# 3) Simulate 100 samples, each with sample size `len(x)` with $\sigma=5$ normally distributed noise added on top of the true function    

sim_data = np.zeros((60,100)) #making the empty data frame with the right shape 

for i in range(100):
    noise = np.random.normal(loc=0, scale = 5, size = len(x)) #making noise for all columns
    sim_data[:,i] = y_true + noise #adding the noise just made to all the true noise


#%%
#     i. do linear, quadratic and fifth-order fits for each of the 100 samples  


#### LINEAR FIT
sim_data_yhat_lin  = np.zeros((60,100)) #making new empty frame for the y hats
#making a loop making the lin reg for each:
for i in range(100):
    reg_3 = LinearRegression() 
    reg_3.fit(x, sim_data[:,i]) #fitting the model to the data from sim_data
    sim_data_yhat_lin[:,i] = reg_3.predict(x) #predicting estimated y's and saving them in the new data y frame.



#### QUADARATIC FIT
sim_data_yhat_quad  = np.zeros((60,100)) #making new empty frame for the y hats
#making a loop making the lin reg for each:
for i in range(100):
    quadratic = PolynomialFeatures(degree=2)
    X_quad = quadratic.fit_transform(x.reshape(-1, 1))#making the fitted model with quad features 
    reg_3 = LinearRegression() 
    reg_3.fit(X_quad, sim_data[:,i]) 
    sim_data_yhat_quad[:,i] = reg_3.predict(X_quad)
    

#### FIFTH POLYNOMIAL FIT
sim_data_yhat_five  = np.zeros((60,100)) #making new empty frame for the y hats
#making a loop making the lin reg for each:
for i in range(100):
    poly_five = PolynomialFeatures(degree=5)
    X_five = poly_five.fit_transform(x.reshape(-1, 1))#making the fitted model with fifth poly features 
    reg_3 = LinearRegression() 
    reg_3.fit(X_five, sim_data[:,i]) 
    sim_data_yhat_five[:,i] = reg_3.predict(X_five) #remember we predict on the new x's now :D
    
    

#%%

#     ii. create a __new__ figure, `plt.figure`, and plot the linear and the quadratic fits (colour them appropriately); highlight the true value for $x_0=3$. From the graphics alone, judge which fit has the highest bias and which has the highest variance for $x_0$  

#this was a try to do it in a work ... it does not work :) it just gives 100 plots 
for i in range (100):
    plt.figure()
    plt.plot(x, y_true, 'r-')
    plt.plot(x,sim_data_yhat_quad[:,i] , 'b-')
    plt.show()
    
#%%

#plotting the quadratic fit (for all 100 samples)
plt.figure()
plt.plot(x, sim_data, 'lightgrey')
plt.plot(x, sim_data_yhat_quad[0:99] , 'g-')
plt.plot(x, y_true, 'r-')
plt.plot(3, our_sqrt(3), 'ro') #adding x0 = 3 and the y value for that calculated on the function (here our square function)
plt.legend(['grey = all data','green = 100 samples of quad fit', 'red =True y values'])
plt.show()


#plotting the linear fit (for all 100 samples)
plt.figure()
plt.plot(x, sim_data, 'lightgray')
plt.plot(x,sim_data_yhat_lin[0:99] , 'g-')
plt.plot(x, y_true, 'r-')
plt.plot(3, our_sqrt(3), 'ro')
plt.legend(['grey = all data','green = 100 samples of lin fit', 'red =True y values'])
plt.show()

##### CONCLUSION: 
# THE LINEAR APPEARS TO BE MORE BIASED, BUT WITH A VERY LITTLE VARIANCE (THE LINES ARE FAR OFF THE POINT)
# THE QUAD FIT HAS A LOW BIAS AND VARIANCE, AND SEEMS TO BE THE BETTER

#(this could be in one plot but then it is hard to interpret)

#%%
#     iii. create a __new__ figure, `plt.figure`, and plot the quadratic and the fifth-order fits (colour them appropriately); highlight the true value for $x_0=3$. From the graphics alone, judge which fit has the highest bias and which has the highest variance for $x_0$  
#plotting the quadratic fit (for all 100 samples)
plt.figure()
plt.plot(x, sim_data, color='lightgray')
plt.plot(x, sim_data_yhat_five[0:99] , 'b-')
plt.plot(x, sim_data_yhat_quad[0:99] , 'g-')
plt.plot(x, y_true, 'r-')
plt.plot(3, our_sqrt(3), 'ro') #adding x0 = 3 and the y value for that calculated on the function (here our square function)
plt.legend(['grey = all data','blue = fifth poly fit','green = quad lin fit', 'red =True y values'])
plt.show()

#### CONCLUSION
# THE QUAD IS STILL THE BETTER FIIIIIT 
#%%
#     iv. estimate the __bias__ and __variance__ at $x_0$ for the linear, the quadratic and the fifth-order fits (the expected value $E[\hat{f}(x_0)] - f(x_0)$ is found by taking the mean of all the simulated, $\hat{f}(x_0)$, differences)  
###estimating bias is the yhat - true y :) cause it gives the difference 

#oki so we have actually chosen to do it for all true y's and not for x0 = 3 only. this could have been done by setting y_true in the loop to our_sqrt(3) instead

#### LINEAR REGRESSION:  
    
#calculating the bias and variance (this gives one mean number for each sample)
lin_bias_variance = np.zeros((3,100))
for i in range(100):
    bias = sim_data_yhat_lin[:,i] - y_true
    lin_bias_variance[0,i] = statistics.mean(bias) #bias is here one number for each sample (instead of having it for each points)
    # calculating the squared bias 
    bias_sqrt = statistics.mean(bias**2)
    lin_bias_variance[1,i] = bias_sqrt
    #calculating the variance 
    variance = statistics.variance(sim_data_yhat_lin[:,i])
    lin_bias_variance[2,i] = variance
   
#the the bias is the first row and the variance is now the second row.
#### finding the mean variance and bias across all samples for comparison 
lin_bias_mean = statistics.mean(lin_bias_variance[0,:])
print(lin_bias_mean)

#finding the sqrt bias across all samples
lin_sqrt_bias =statistics.mean(lin_bias_variance[1,:])

#finding the variance across all samples 
lin_variance_mean = statistics.mean(lin_bias_variance[2,:])


#%%

#### QUADRATIC REGRESSION
#calculating the bias (this gives one mean number for each sample)
quad_bias_variance = np.zeros((3,100))
for i in range(100):
    bias = sim_data_yhat_quad[:,i] - y_true
    quad_bias_variance[0,i] = statistics.mean(bias) #bias is here one number for each sample (instead of having it for each points)
    # calculating the squared bias 
    bias_sqrt = statistics.mean(bias**2)
    quad_bias_variance[1,i] = bias_sqrt
    #calculating the variance 
    variance = statistics.variance(sim_data_yhat_quad[:,i])
    quad_bias_variance[2,i] = variance
    
    

quad_bias_mean = statistics.mean(quad_bias_variance[0,:])
print(quad_bias_mean)

#finding the sqrt bias across all samples
quad_sqrt_bias =statistics.mean(quad_bias_variance[1,:])


#finding the variance across all samples 
quad_variance_mean = statistics.mean(quad_bias_variance[2,:])


#%%
#### FIVE DEGREE REGRESSION
#calculating the bias (this gives one mean number for each sample)
five_bias_variance = np.zeros((3,100))
for i in range(100):
    bias = sim_data_yhat_five[:,i] - y_true
    five_bias_variance[0,i] = statistics.mean(bias) #bias is here one number for each sample (instead of having it for each points)
     # calculating the squared bias 
    bias_sqrt = statistics.mean(bias**2)
    five_bias_variance[1,i] = bias_sqrt
    #calculating the variance 
    variance = statistics.variance(sim_data_yhat_five[:,i])
    five_bias_variance[2,i] = variance

five_bias_mean = statistics.mean(five_bias_variance[0,:])
print(quad_bias_mean)

#finding the sqrt bias across all samples
five_sqrt_bias =statistics.mean(five_bias_variance[1,:])

#finding the variance across all samples i
five_variance_mean = statistics.mean(five_bias_variance[2,:])



#### it seams as if our biases are more or less the same, and the variance are smallest in the linear regression - the bias part are hard for me to explain though. 

#%%
#     v. show how the __squared bias__ and the __variance__ is related to the complexity of the fitted models
### we would expect the bias to be smallest in the most complex model (this fits the most data, maybe even too much, but can therefore probably explain data. )
print(lin_sqrt_bias)
print(quad_sqrt_bias)
print(five_sqrt_bias)

print(lin_variance_mean)
print(quad_variance_mean)
print(five_variance_mean)


#%%  
#     vi. simulate __epsilon__: `epsilon = np.random.normal(scale=5, size=100)`. 
##Based on your simulated values of __bias, variance and epsilon__, what is the __Mean Squared Error__ for each of the three fits? Which fit is better according to this measure? 

### LINEAR FIT MSE
MSE_lin = []
for i in range(100):
    MSE = np.square(np.subtract(y_true,sim_data_yhat_lin[:,i])).mean()
    MSE_lin.append(MSE)

MSE_lin_avg = statistics.mean(MSE_lin)
print(MSE_lin_avg)

### QUAD FIT 
MSE_quad = []
for i in range(100):
    MSE = np.square(np.subtract(y_true,sim_data_yhat_quad[:,i])).mean()
    MSE_quad.append(MSE)

MSE_quad_avg = statistics.mean(MSE_quad)
print(MSE_quad_avg)


### fIFTH 
MSE_five = []
for i in range(100):
    MSE = np.square(np.subtract(y_true,sim_data_yhat_five[:,i])).mean()
    MSE_five.append(MSE)

MSE_five_avg = statistics.mean(MSE_five)
print(MSE_five_avg)


###### CONCLUSION: THE QUAD FIT IS THE BETTER FIT ACCORDING TO THE MEAN SQUARE ERROR 

#%%
# EXERCISE 2: Fitting training data and applying it to test sets with and without regularization

# All references to pages are made to this book:
# Raschka, S., 2015. Python Machine Learning. Packt Publishing Ltd.  

#%%
# 1) Import the housing dataset using the upper chunk of code from p. 280 
df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-2nd-edition'
                 '/master/code/ch10/housing.data.txt',
                 header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# 1. CRIM      per capita crime rate by town
# 2. ZN        proportion of residential land zoned for lots over 
#                  25,000 sq.ft.
# 3. INDUS     proportion of non-retail business acres per town
# 4. CHAS      Charles River dummy variable (= 1 if tract bounds 
#                  river; 0 otherwise)
# 5. NOX       nitric oxides concentration (parts per 10 million)
# 6. RM        average number of rooms per dwelling
# 7. AGE       proportion of owner-occupied units built prior to 1940
# 8. DIS       weighted distances to five Boston employment centres
# 9. RAD       index of accessibility to radial highways
# 10. TAX      full-value property-tax rate per $10,000
# 11. PTRATIO  pupil-teacher ratio by town
# 12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
#                  by town
# 13. LSTAT    % lower status of the population
# 14. MEDV     Median value of owner-occupied homes in $1000s


#     i. and define the correlation matrix `cm` as done on p. 284  
#CORRELATION MATRIX 
cm = np.corrcoef(df[columns].values.T)
#sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)

plt.tight_layout()
plt.show()

#     ii. based on this matrix, do you expect collinearity can be an issue if we run multiple linear regression  by fitting MEDV on LSTAT, INDUS, NOX and RM?  

#%%
# 2) Fit MEDV on  LSTAT, INDUS, NOX and RM (standardize all five variables by using `StandardScaler.fit_transform`, (`from sklearn.preprocessing import StandardScaler`) by doing multiple linear regression using `LinearRegressionGD` as defined on pp. 285-286
#     i. how much does the solution improve in terms of the cost function if you go through 40 iterations instead of the default of 20 iterations?  
#     ii. how does the residual sum of squares based on the analytic solution (Ordinary Least Squares) compare to the cost after 40 iterations?
#     iii. Bonus question: how many iterations do you need before the Ordinary Least Squares and the Gradient Descent solutions result in numerically identical residual sums of squares?  
#

#%%
# 3) Build your own cross-validator function. This function should randomly split the data into $k$ equally sized folds (see figure p. 176) (see the code chunk associated with exercise 2.3). It should also return the Mean Squared Error for each of the folds
#     i. Cross-validate the fits of your model from Exercise 2.2. Run 11 folds and run 500 iterations for each fit  
#     ii. What is the mean of the mean squared errors over all 11 folds?  


def cross_validate(estimator, X, y, k): # estimator is the object created by initialising LinearRegressionGD
    mses = list() # we want to return k mean squared errors
    fold_size = y.shape[0] // k # we do integer division to get a whole number of samples
    for fold in range(k): # loop through each of the folds
        
        X_train = ?
        y_train = ?
        X_test = ?
        y_test = ?
        
        # fit training data
        # predict on test data
        # calculate MSE
        
    return mses



#%%
# 4) Now, we will do a Ridge Regression. Use `Ridge` (see code chunk associated with Exercise 2.4) to find the optimal `alpha` parameter ($\lambda$)
#     i. Find the _MSE_ (the mean of the _MSE's_ associated with each fold) associated with a reasonable range of `alpha` values (you need to find the lambda that results in the minimum _MSE_)  
#     ii. Plot the _MSE_ as a function of `alpha` ($\lambda$). Make sure to include an _MSE_ for `alpha=0` as well  
#     iii. Find the _MSE_ for the optimal `alpha`, compare its _MSE_ to that of the OLS regression
#     iv. Do the same steps for Lasso Regression `Lasso`  (2.4.i.-2.4.iii.)
#     v. Describe the differences between these three models, (the optimal Lasso, the optimal Ridge and the OLS)



