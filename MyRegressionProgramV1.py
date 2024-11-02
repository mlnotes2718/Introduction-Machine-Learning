import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Cost Function
def cost_function(X,y,b,w):
    '''
    Actual cost function for both single and multiple features
    X = matrix of training data, each training examples in rows (m) and features in column (n), 
        Single feature data must be in m by 1 vector, where m is total number of training examples.
    y = m by 1 vector, where m is total number of training examples.
    b = scalar
    w = n by 1 vector, where n is total number of features

    Return:
    Cost (scalar)
    '''
    
    m = X.shape[0]
    fx = (X@w)+b
    lossFunction = (fx - y) ** 2
    RSS = lossFunction.sum()
    cost = (1 / (2 * m)) * RSS

    return cost


# Compute Gradient
def compute_gradient(X,y,b,w):
    '''
    Compute gradient for each step size
    X = matrix of training data, each training examples in rows and features in column, 
        Single feature data must be in m by 1 vector, where m is total number of training examples.
    y = m by 1 vector, where m is total number of training examples.
    b = scalar
    w = n by 1 vector, where n is total number of features

    Return:
    db (scalar)
    dw (n by 1 vector)
    '''
    
    
    # total number of features
    n = w.shape[0]

    # Initialization of Variables 
    db = 0
    dw = 0
    temp_db = 0
    temp_dw = 0

    # Matrix computation
    fx = ((X@w) + b)
    temp_dw = (fx - y) * X
    temp_db = (fx - y)
    
    db = temp_db.mean() 
    dw = temp_dw.mean(axis=0).reshape([n,1]) 

    return db, dw


# Run Gradient Descent  
def compute_gradient_descent(X, y, iterations=10000, init_b=0., init_w = None, alpha=0.01, printProgress=False, printProgressRate = 1000, storeHistory=False):
    '''
    Runs Gradient Descent

    Compulsory Input
    X = matrix of training data, each training examples in rows and features in column, 
        Single feature data must be in m by 1 vector, where m is total number of training examples.
    y = m by 1 vector, where m is total number of training examples.
    
    Optional Input with Defauls
    iterations = total number of runs for the gradient descent (default: 10,000)
    init_b = initial b value (type: scalar) (default:0.)
    init_w = initial w value (type: n by 1 vector, where n is total number of features) (default:0.)
    alpha = learning rate / step size (default:0.01)

    Print Progress Options
    printProgress = To print the details while running gradient descent (type:Boolean) (default: False)
    printProgressRate = To print the details every n iterations (default:1000)

    Return:
    w = best w (n by 1 vector, where n is total number of features)
    b = best b (scalar)
    cost_history = Computed cost for each iterations (list)
    w_history = Computed w for each iterations (list)
    b_history = Computed g for each iterations (list)
    '''

    #### The following check for different data types and convert them to Numpy
    #### Also convert Pandas Series and DataFrame to Numpy
    #### Also convert 1D array to Numpy
    
    ### the following check if data type is Series
    if isinstance(X, pd.Series):
        #print('convert to numpy')
        X = X.to_frame()
    if isinstance(y, pd.Series):
        y = y.to_frame()
    
    ### the following check if data type is dataframe
    if isinstance(X, pd.DataFrame):
        #print('convert to numpy')
        X = X.to_numpy()
    if isinstance(y, pd.DataFrame):
        y = y.to_numpy()

    
    ### the following section convert 1d array to 2d array for ease of computation ###
    if X.ndim == 1:
        X = X.reshape(len(X),1)
    if y.ndim == 1:
        y = y.reshape(len(y),1)


    ### Check for init_w separately as the dimension of w depends on X
    ### Also perform copy to avoid changing the original array

    if np.any(init_w) == None:
        init_w = np.zeros([X.shape[1],1])

    if isinstance(init_w, pd.Series):
        init_w = init_w.copy().to_frame()

    if isinstance(init_w, pd.DataFrame):
        init_w = init_w.copy().to_numpy()

    if init_w.ndim == 1:
        init_w = init_w.copy()
        init_w = init_w.reshape(len(init_w),1)

    
    ### The following will check if the size match
    if X.shape[0] != y.shape[0]:
        print('Error, the size of X and y does not match!')
        return 
    
    if init_w.shape[0] != X.shape[1]:
        print('Error, the size of X features does not match with size of w !')
        print(init_w.shape)
        print(X.shape[1])
        return

    # Initialization of variables
    m,n = X.shape
    
    db = 0
    dw = 0

    b = init_b
    w = init_w
    
    cost_history = []
    w_history = np.zeros((1,n))
    b_history = []

    
    for j in range(iterations):

        # Compute Partial Derivatives
        db, dw = compute_gradient(X,y,b,w)

        b = b - (alpha * db)
        w = w - (alpha * dw)
        
        cost = cost_function(X,y,b,w)

        # Reshape w for printing and storing history
        w_convert = w.copy()
        w_convert = np.transpose(w_convert)
        
        if storeHistory == True: 
            cost_history.append(cost)
            b_history.append(b)
            w_history = np.vstack((w_history,w_convert))

        if printProgress == True:
            if j % printProgressRate == 0:
                print(f"iteration {j}: cost = {cost:.4e}: intercept = {b:.4e}: weights = {w_convert}")

    print(f"iteration {j}: Last cost = {cost:.4e}: intercept = {b:.4e}: weights = {w_convert}")
    print('best w', np.round(w,4))
    print('best b', np.round(b,4)) 

    return w, b, cost_history, w_history[1:], b_history


# Predict fx based on b and w
def prediction(X,b,w):
    '''
    Compute prediction of y_hat/y_predict based on input b and w
    X = matrix of training data, each training examples in rows and features in column, 
        Single feature data must be in m by 1 vector, where m is total number of training examples.
    b = scalar
    w = n by 1 vector, where n is total number of features

    Return:
    y_predict (m by 1 vector)
    '''
    
    y_predict = (X@w) + b
    return y_predict

def std_norm(X):
    """
    This function is z-score normalizer.
    Formula is x(scaled) = x - mean / {std deviation of x}
    There is similar scaler in sklearn is StandardScaler

    INPUT:
    X = The features dataset
    Dataset should be in column features with shape (m,n) where m is number of observations
    and n is total number of features

    RETURN:
    X_norm = Scaled dataset
    avg = Mean of each column features
    std = Standard Deviation of each column features
    """
    ### the following check if data type is Series
    ### if is Series convert to data frame
    if isinstance(X, pd.Series):
        X = X.to_frame()

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    if X.ndim == 1:
        np.array(X).reshape(-1,1)
        
    n = X.shape[1]
    
    avg = X.mean(axis=0).reshape((1,n))

    std = X.std(axis=0).reshape((1,n))
    
    X_norm = (X - avg) / std
    
    return X_norm, avg, std


def reverse_bw_std_norm(coef, intercept, avg, stddev):
    '''
    The following is to reverse the coeficient and intercept after using std_norm.
    Please use the same mean and std deviation from std_norm
    This function can use for one feature or multiple features regression

    INPUT:
    coef = The weights / coefficient of the trained dataset
    intercept = The intercept of the trained dataset
    avg = The average return from the function std_norm
    stddev = The standard deviation from the function std_norm

    RETURN:
    or_coef_gd = The weights / coefficient in un-scaled form
    or_intercept_gd = The intercept in un-scaled form
    
    '''
    n = coef.shape[0]
    ### reshape average and stddev
    avg = avg.reshape((n,1))
    stddev = stddev.reshape((n,1))

    or_coef_gd = coef / stddev
    or_intercept_gd = intercept + np.negative(coef * avg / stddev).sum()
    return or_coef_gd, or_intercept_gd

def minmax_scaling(X):
    """
    This function is to replicate the same method as SciKit Learn MinMaxScaler
    Formula is x(scaled) = x - min(x) / max(x) - min(x)
    This function produce similar result SciKit Learn MinMaxScaler

    INPUT:
    X = The features dataset
    Dataset should be in column features with shape (m,n) where m is number of observations
    and n is total number of features

    RETURN:
    scaled = Scaled dataset
    minimum = Minimum of each column features
    range = Maximum of each column features MINUS Minimum of each column features
    """
    ### the following check if data type is Series
    ### if is Series convert to data frame
    if isinstance(X, pd.Series):
        X = X.to_frame()

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    if X.ndim == 1:
        np.array(X).reshape(-1,1)
        
    n = X.shape[1]
    
    maximum = X.max(axis=0)
    minimum = X.min(axis=0)
    range = (maximum - minimum)
    scaled = (X - minimum) / range
    return scaled, minimum.reshape(-1,1), range.reshape(-1,1)

def reverse_bw_minmax_scale(coef, intercept, min, range):
    '''
    The following is to reverse the coeficient and intercept after using minmax_scaling.
    Please use the same minimum and range from minmax_scaling
    This function can use for one feature or multiple features regression

    INPUT:
    coef = The weights / coefficient of the trained dataset
    intercept = The intercept of the trained dataset
    min = The minimum return from the function minmax_scaling
    range = The range (max - min) for each column features from the function minmax_scaling

    RETURN:
    or_coef_gd = The weights / coefficient in un-scaled form
    or_intercept_gd = The intercept in un-scaled form
    
    '''
    or_coef_gd = coef / range
    or_intercept_gd = intercept + np.negative(coef * min / range).sum()
    return or_coef_gd, or_intercept_gd
