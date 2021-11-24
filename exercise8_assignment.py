#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:28:55 2021

@author: laura
"""
#%%
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
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
import os

#%%
# Exercises and objectives

# 1) Load the magnetoencephalographic recordings and do some initial plots to understand the data  
# 2) Do logistic regression to classify pairs of PAS-ratings  
# 3) Do a Support Vector Machine Classification on all four PAS-ratings  

# REMEMBER: In your report, make sure to include code that can reproduce the answers requested in the exercises below (__MAKE A KNITTED VERSION__)  
# REMEMBER: This is Assignment 3 and will be part of your final portfolio   

# # EXERCISE 1 - Load the magnetoencephalographic recordings and do some initial plots to understand the data  

# The files `megmag_data.npy` and `pas_vector.npy` can be downloaded here (http://laumollerandersen.org/data_methods_3/megmag_data.npy) and here (http://laumollerandersen.org/data_methods_3/pas_vector.npy)   

# 1) Load `megmag_data.npy` and call it `data` using `np.load`. You can use `join`, which can be imported from `os.path`, to create paths from different string segments  

#%%
# i. The data is a 3-dimensional array. The first dimension is number of repetitions of a visual stimulus , the second dimension is the number of sensors that record magnetic fields (in Tesla) that stem from neurons activating in the brain, and the third dimension is the number of time samples. How many repetitions, sensors and time samples are there?  
data = np.load("/Users/laura/Desktop/megmag_data.npy")

#numbers of repitions of visual stimuli: 682
#numbers of sensors (spots in the cap that record magnetic fields): 102
#numbers of time samples: 251
#### this is found by:
print(data.shape)

#%%
# ii. The time range is from (and including) -200 ms to (and including) 800 ms with a sample recorded every 4 ms. At time 0, the visual stimulus was briefly presented. Create a 1-dimensional array called `times` that represents this.  
##creating array of time samples:
time_array = np.arange(-200, 804, 4)



#%%
#     iii. Create the sensor covariance matrix $\Sigma_{XX}$: $$\Sigma_{XX} = \frac 1 N \sum_{i=1}^N XX^T$$ $N$ is the number of repetitions and $X$ has $s$ rows and $t$ columns (sensors and time), thus the shape is $X_{s\times t}$. Do the sensors pick up independent signals? (Use `plt.imshow` to plot the sensor covariance matrix)  
n = 682
cov_mat = []

#calculating the dot product for all rows i using all datapoints in the dimensions
for i in range(n):
    cov_mat.append(data[i,:,:] @ data[i,:,:].T)

#out of the loop the dot product of the the matrices for each i is summed and divided by n.
cov_mat = sum(cov_mat)/n

#plotting the covariance matrix
plt.imshow(cov_mat)


#%%
#     iv. Make an average over the repetition dimension using `np.mean` - use the `axis` argument. (The resulting array should have two dimensions with time as the first and magnetic field as the second)  

### we take the mean of all the repetions, the first axis = 0 - this now gives us all the values for sensors and time samples in this exact point (the mean) on the repetition axes 
rep_mean = np.mean(data, axis=0)
print(rep_mean.shape)

#%%
#     v. Plot the magnetic field (based on the average) as it evolves over time for each of the sensors (a line for each) (time on the x-axis and magnetic field on the y-axis). Add a horizontal line at $y = 0$ and a vertical line at $x = 0$ using `plt.axvline` and `plt.axhline`  
for i in range (102):
    plt.plot(time_array, rep_mean[i,:])
    plt.axvline(0,0, color = "black")
    plt.axhline(0,0, color = "black")


#%% 
#     vi. Find the maximal magnetic field in the average. 
## Then use `np.argmax` and `np.unravel_index` to find the sensor that has the maximal magnetic field.  
 
#finding the max - but this gives a coordinate in the dataframe of mean times:
maxi_mag = np.unravel_index(np.argmax(rep_mean), rep_mean.shape)
print(maxi_mag) #sensor 73, repition 112
#the coordinates can be printed as a number:
print(rep_mean[73,112])
# = 2.7886216843591933e-13 which is a super small number, yet it is our maximum. 

    

#%%
#     vii. Plot the magnetic field for each of the repetitions (a line for each) for the sensor that has the maximal magnetic field. Highlight the time point with the maximal magnetic field in the average (as found in 1.1.v) using `plt.axvline`  
## this is sensor 73 - we wanna plot the repitions and times for this exactly 
for i in range(len(data[:, 73, 0])):
    plt.plot(time_array, data[i,73,:])
    plt.axvline(time_array[112], color = "black")


#%%
#     viii. Describe in your own words how the response found in the average is represented in the single repetitions. 
#But do make sure to use the concepts _signal_ and _noise_ and comment on any differences on the range of values on the y-axis  

"""sooo we see the differences in the single repitions plot (the latter), and the average (the first with the black cross) in the distribution around y = 0. 
Firstly we found how sensor 73 is the one with the highest magnetic fields.

a lot other can be written here ) """


#%%

# 2) Now load `pas_vector.npy` (call it `y`). PAS is the same as in Assignment 2, describing the clarity of the subjective experience the subject reported after seeing the briefly presented stimulus  
vector_y = np.load("/Users/laura/Desktop/pas_vector.npy")

#     i. Which dimension in the `data` array does it have the same length as?  
print(vector_y.shape)
#the data is a vector of 682 rows - quite convenient since it matches the numbers of repitions 

#%%
#     ii. Now make four averages (As in Exercise 1.1.iii), one for each PAS rating, and plot the four time courses (one for each PAS rating) for the sensor found in Exercise 1.1.v  

##first we save a vector of the numbers of the placement for the passes:
## pass one
pas_one = [i for i, x in enumerate(vector_y) if x ==1 ]
#this list can now be used index the matching numbers out of the data - we should have 99 reps,since this is the amount of passes:
print(data[pas_one,:,:].shape)
pas_1 = data[pas_one,:,:] #saving this in new frame
#finding the mean of repitions for each pass:
print(np.mean(pas_1[0]))    # finding the mean
    

## pas two 
pas_two = [i for i, x in enumerate(vector_y) if x == 2 ]
print(data[pas_two,:,:].shape)
pas_2 = data[pas_two,:,:]#saving this in new frame
print(np.mean(pas_2[0]))    # finding the mean

## pas three
pas_three = [i for i, x in enumerate(vector_y) if x == 3]
print(data[pas_three,:,:].shape)
pas_3 = data[pas_three,:,:]#saving this in new frame
print(np.mean(pas_3[0]))    # finding the mean

## pas four
pas_four = [i for i, x in enumerate(vector_y) if x == 4]
print(data[pas_four,:,:].shape)
pas_4 = data[pas_four,:,:]#saving this in new frame
print(np.mean(pas_4[0]))    # finding the mean

# NOW WE HAVE FOUND THE MEAN VALUE OF REPITIONS FOR ALL PASSES

####### NOW we plot the means for each passes on top of the averages 
for i in range(102):
  plt.plot(time_array, rep_mean[i,:])
  plt.axhline(y = np.mean(pas_1[0]), color = "red")
  plt.axhline(y = np.mean(pas_2[0]), color = "blue")
  plt.axhline(y = np.mean(pas_3[0]), color = "green")
  plt.axhline(y = np.mean(pas_4[0]), color = "yellow")
  
 #"""We can see how the pas 2 = the blue line has the highest magnetic field""""  
  
#this visualize the means of all passes on top of the average magnitudes 
### maybe we should have done this exact thing but on the sensor with highest magnitude:


 #%% 
#doing the same on sensor 73
###finding the mean for each pass on sensor 73:
s_pas_one = [i for i, x in enumerate(vector_y) if x == 1 ]
s_pas_1 = data[s_pas_one,73,:]#saving this in new frame
1
s_pas_two = [i for i, x in enumerate(vector_y) if x == 2 ]
s_pas_2 = data[s_pas_two,73,:]#saving this in new frame

s_pas_three = [i for i, x in enumerate(vector_y) if x == 3 ]
s_pas_3 = data[s_pas_three,73,:]

s_pas_four = [i for i, x in enumerate(vector_y) if x == 4]
s_pas_4 = data[s_pas_four,73,:]
 
    
for i in range(len(data[:, 73, 0])):
    plt.plot(time_array, data[i,73,:])
    plt.axhline(y = np.mean(s_pas_1[0]), color = "red")
    plt.axhline(y = np.mean(s_pas_2[0]), color = "blue")
    plt.axhline(y = np.mean(s_pas_3[0]), color = "green")
    plt.axhline(y = np.mean(s_pas_4[0]), color = "yellow")
    
#this shows how pas2 has the lowest magnitude score, pas3 the highest, and 1 and 4 being in the middle - not as one might have expected. 
#Since the magnitude should seem to correspond with the clarity of the subjective experience. but this does not seem to be the case - wild guess here"""

#%%
#     iii. Notice that there are two early peaks (measuring visual activity from the brain), one before 200 ms and one around 250 ms. Describe how the amplitudes of responses are related to the four PAS-scores. Does PAS 2 behave differently than expected?  
#here a cool explaining could be given - not by me"""

#%%
# # EXERCISE 2 - Do logistic regression to classify pairs of PAS-ratings  

# 1) Now, we are going to do Logistic Regression with the aim of classifying the PAS-rating given by the subject  
#     i. We'll start with a binary problem - create a new array called `data_1_2` that only contains PAS responses 1 and 2. 
#      Similarly, create a `y_1_2` for the target vector  
list_y = np.argwhere((vector_y ==1) | (vector_y == 2))
y_1_2 = vector_y[list_y]

#%%

#taking the target values for each pass and saved them into data_1_2
data_one = data[pas_one,:,:]
data_two = data[pas_two,:,:]
data_1_2 = np.squeeze(np.concatenate((data_one,data_two), axis = 0))


##looking at the length to make sure they are the same
print(data_1_2.shape)

#%%
#     ii. Scikit-learn expects our observations (`data_1_2`) to be in a 2d-array, which has samples (repetitions) on dimension 1 and features (predictor variables) on dimension 2. Our `data_1_2` is a three-dimensional array. Our strategy will be to collapse our two last dimensions (sensors and time) into one dimension, while keeping the first dimension as it is (repetitions). Use `np.reshape` to create a variable `X_1_2` that fulfils these criteria.  
X_1_2 = data_1_2.transpose(0,1,2).reshape(-1,data_1_2.shape[0])
print(X_1_2.shape)
#this appears to be right since the shape now is the collapsed data in the 25602 row and the 214 repitions are the columns. 

#%%
#     iii. Import the `StandardScaler` and scale `X_1_2`  
#reshapingen 
X_1_2_reshaped = X_1_2.reshape(214, 25602)



## splitting the data so we can test how well it classifies the test set:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_1_2_reshaped, y_1_2, test_size=0.2,
                                                    random_state=0)

#making everything on the same scale
sc = StandardScaler()
sc.fit(X_1_2_reshaped)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)



#%%
#     iv. Do a standard `LogisticRegression` - can be imported from `sklearn.linear_model` - make sure there is no `penalty` applied  
logR = LogisticRegression(penalty='none') # no regularisation
logR.fit(X_train_std, y_train)


#%%
#     v. Use the `score` method of `LogisticRegression` to find out how many labels were classified correctly. Are we overfitting? Besides the score, what would make you suspect that we are overfitting?  
print(logR.score(X_test_std, y_test))
#### this means that 55,1% of the classifications are correct. 
## this is not very much, which might could be explained by overfitting of the test data. The model are therefore not generalisable, and classifies the test data quite badly.
## if the model was not overfitted to the train data, we would expect it to be a better classifier - maybe we should penalizzzzze it 


#%%
#     vi. Now apply the _L1_ penalty instead - how many of the coefficients (`.coef_`) are non-zero after this?  
log_l1 = LogisticRegression(penalty = 'l1', solver = 'liblinear') # set 
log_l1.fit(X_train_std, y_train)
print(log_l1.score(X_test_std, y_test))
#now that we have fitted the new penalized model we can check how many of the coefficients are not equal to zero
#this actually has a lower accuracy of classification 
#%%
#finding the zeros 
l1_coef = log_l1.coef_
print(np.count_nonzero(l1_coef == 0)) #this is the amounts of 0's 
print(np.count_nonzero(l1_coef != 0)) #this is the amounts of non 0's 


#%%
#     vii. Create a new reduced $X$ that only includes the non-zero coefficients - show the covariance of the non-zero features (two covariance matrices can be made; $X_{reduced}X_{reduced}^T$ or $X_{reduced}^TX_{reduced}$ (you choose the right one)) . Plot the covariance of the features using `plt.imshow`. Compared to the plot from 1.1.iii, do we see less covariance?  
log1_coef_0 = np.where(l1_coef!= 0)[1]
log1_coef_0.shape

reduced_X=X_1_2_reshaped[:,log1_coef_0]
reduced_X.shape

plt.imshow(np.cov(reduced_X))

#%%
# 2) Now, we are going to build better (more predictive) models by using cross-validation as an outcome measure 

   
#     i. Import `cross_val_score` and `StratifiedKFold` from `sklearn.model_selection`  
from sklearn.model_selection import cross_val_score, StratifiedKFold 

#%%
#     ii. To make sure that our training data sets are not biased to one target (PAS) or the other, create `y_1_2_equal`, which should have an equal number of each target. 
# Create a similar `X_1_2_equal`. The function `equalize_targets_binary` in the code chunk associated with Exercise 2.2.ii can be used. 
# Remember to scale `X_1_2_equal`!  
def equalize_targets_binary(data, y):
    np.random.seed(7)
    targets = np.unique(y) ## find the number of targets
    if len(targets) > 2:
        raise NameError ("can't have more than two targets") #may only be binary
    counts = list()
    indices = list()
    for target in targets:
        counts.append(np.sum(y == target)) ## find the number of each target
        indices.append(np.where(y == target)[0]) ## find their indices
    min_count = np.min(counts)
    # randomly choose trials
    first_choice = np.random.choice(indices[0], size=min_count, replace=False)
    second_choice = np.random.choice(indices[1], size=min_count,replace=False)
    # create the new data sets
    new_indices = np.concatenate((first_choice, second_choice))
    new_y = y[new_indices]

    new_data = data[new_indices, :, :]
    return new_data, new_y


#%%
data_1_2_equal, y_1_2_equal_new = equalize_targets_binary(data_1_2, y_1_2)


#%%
#     iii. Do cross-validation with 5 stratified folds doing standard `LogisticRegression` (See Exercise 2.1.iv)  
#     iv. Do L2-regularisation with the following `Cs=  [1e5, 1e1, 1e-5]`. Use the same kind of cross-validation as in Exercise 2.2.iii. In the best-scoring of these models, how many more/fewer predictions are correct (on average)?  
#     v. Instead of fitting a model on all `n_sensors * n_samples` features, fit  a logistic regression (same kind as in Exercise 2.2.iv (use the `C` that resulted in the best prediction)) for __each__ time sample and use the same cross-validation as in Exercise 2.2.iii. What are the time points where classification is best? Make a plot with time on the x-axis and classification score on the y-axis with a horizontal line at the chance level (what is the chance level for this analysis?)  
#     vi. Now do the same, but with L1 regression - set `C=1e-1` - what are the time points when classification is best? (make a plot)?  
#     vii. Finally, fit the same models as in Exercise 2.2.vi but now for `data_1_4` and `y_1_4` (create a data set and a target vector that only contains PAS responses 1 and 4). What are the time points when classification is best? Make a plot with time on the x-axis and classification score on the y-axis with a horizontal line at the chance level (what is the chance level for this analysis?)  
#%%
# 3) Is pairwise classification of subjective experience possible? Any surprises in the classification accuracies, i.e. how does the classification score fore PAS 1 vs 4 compare to the classification score for PAS 1 vs 2?  

#%%
# # EXERCISE 3 - Do a Support Vector Machine Classification on all four PAS-ratings  
# 1) Do a Support Vector Machine Classification  
# i. First equalize the number of targets using the function associated with each PAS-rating using the function associated with Exercise 3.1.i  
# ii. Run two classifiers, one with a linear kernel and one with a radial basis (other options should be left at their defaults) - the number of features is the number of sensors multiplied the number of samples. Which one is better predicting the category?
# iii. Run the sample-by-sample analysis (similar to Exercise 2.2.v) with the best kernel (from Exercise 3.1.ii). Make a plot with time on the x-axis and classification score on the y-axis with a horizontal line at the chance level (what is the chance level for this analysis?)
# iv. Is classification of subjective experience possible at around 200-250 ms?  


# 2) Finally, split the equalized data set (with all four ratings) into a training part and test part, where the test part if 30 % of the trials. Use `train_test_split` from `sklearn.model_selection`  
# i. Use the kernel that resulted in the best classification in Exercise 3.1.ii and `fit`the training set and `predict` on the test set. This time your features are the number of sensors multiplied by the number of samples.  
# ii. Create a _confusion matrix_. It is a 4x4 matrix. The row names and the column names are the PAS-scores. There will thus be 16 entries. The PAS1xPAS1 entry will be the number of actual PAS1, $y_{pas1}$ that were predicted as PAS1, $\hat y_{pas1}$. The PAS1xPAS2 entry will be the number of actual PAS1, $y_{pas1}$ that were predicted as PAS2, $\hat y_{pas2}$ and so on for the remaining 14 entries.  Plot the matrix
# iii. Based on the confusion matrix, describe how ratings are misclassified and if that makes sense given that ratings should measure the strength/quality of the subjective experience. Is the classifier biased towards specific ratings?  

def equalize_targets(data, y):
    np.random.seed(7)
    targets = np.unique(y)
    counts = list()
    indices = list()
    for target in targets:
        counts.append(np.sum(y == target))
        indices.append(np.where(y == target)[0])
    min_count = np.min(counts)
    first_choice = np.random.choice(indices[0], size=min_count, replace=False)
    second_choice = np.random.choice(indices[1], size=min_count, replace=False)
    third_choice = np.random.choice(indices[2], size=min_count, replace=False)
    fourth_choice = np.random.choice(indices[3], size=min_count, replace=False)
    
    new_indices = np.concatenate((first_choice, second_choice,
                                 third_choice, fourth_choice))
    new_y = y[new_indices]
    new_data = data[new_indices, :, :]
    
    return new_data, new_y
