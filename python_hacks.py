#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:16:20 2021

@author: laura
"""
# Remember
## = instead of ->

#%%
# Python commands:
## rgen --> random number generator, if normally distributed: rgen.normal(loc=0.0,scale=0.01, size=1 + X.shape[1])
## np.dot --> calculates the vector dot product
## df = pd.read_csv('data.csv', header=None) --> reading in dataframe 
## df.tail() --> checking out the dataset
## The % symbol in Python is called the Modulo Operator. It returns the remainder of dividing the left hand operand by right hand operand. It's used to get the remainder of a division problem


## yourmodel.fit(X,y) --> seem to give you the estimated/fitted values of y and X based on the model.
##yourmodel.predict() --> predicts new y values based on the training x's, example: y_pred = slr.predict(X), where slr is a model


#%%
# PLOTS 

#%%
### simple plots
#remember python builds up plots - so in each line you add something to your plot and show it in the end.
plt.figure() # create new figure
plt.plot([1, 2], [1, 2], 'b-') # plot a blue line
#plt.show() # show figure

plt.plot([2, 1], [2, 1], 'ro') # scatter plot (red)
#plt.show()
plt.xlabel('a label')
plt.title('a title')
plt.legend(['a legend', 'another legend'])
plt.show()

#%%
#######plot for decision regions 
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
        
#%%

##### plot for scatterplot matrix:
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']#names of the columns in the data you wanna check for pairwise correlation

sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.show()

#%%


