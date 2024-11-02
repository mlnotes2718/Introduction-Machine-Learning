## Basics
import math
import scipy
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm

## SciKit Learning Preprocessing 
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Pipeline and Train Test Split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

## SciKit Learn ML Models
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

############################################### Notes #################################################
## Please note that for this version regularization is included in cost function, 
## compute gradient and compute gradient Descent
#######################################################################################################


#######################################################################################################
## Linear Regression
#######################################################################################################

def cost_function(X,y,coefficient,intercept,reguLambda=0):
    '''
    Actual cost function for both single and multiple features
    X = matrix of training data, each training examples in rows (m) and features in column (n), 
        Single feature data must be in m by 1 vector, where m is total number of training examples.
    y = m by 1 vector, where m is total number of training examples.
    intercept (b) = scalar
    coefficient (w) = n by 1 vector, where n is total number of features
    reguLambda = default is 0, no regularization, please enter appropriate lambda value for regularization

    Return:
    Cost (scalar)
    '''

    ### the following section convert 1d array to 2d array for ease of computation ###
    if X.ndim == 1:
        X = X.reshape(-1,1)
    
    if y.ndim == 1:
        y = y.reshape(-1,1)
        
    if np.isscalar(coefficient) == True:
        coefficient = np.array([coefficient])
        
    if coefficient.ndim == 1:
        coefficient = coefficient.reshape(-1,1)
    
    m = X.shape[0]

    # Compute Normal Cost Function
    fx = (X@coefficient) + intercept
    lossFunction = (fx - y) ** 2
    RSS = lossFunction.sum()
    cost = (1 / (2 * m)) * RSS

    # Compute Regularization Term
    # lambda/2m * sum(w^2)
    regu = (reguLambda/ (2*m)) * sum(coefficient ** 2)

    totalCost = cost + regu[0]
    
    return totalCost

# Compute Gradient
def compute_gradient(X,y, coefficient, intercept=0., reguLambda=0.):
    '''
    Compute gradient for each step size
    X = matrix of training data, each training examples in rows and features in column, 
        Single feature data must be in m by 1 vector, where m is total number of training examples.
    y = m by 1 vector, where m is total number of training examples.
    w = n by 1 vector, where n is total number of features
    b = scalar (default = 0)
    reguLambda = scalar (default = 0), no regularization if default. Please enter appropriate lambda value for regularization.

    Return:
    db (scalar)
    dw (n by 1 vector)
    '''
    
    ### the following section convert 1d array to 2d array for ease of computation ###
    if X.ndim == 1:
        X = X.reshape(-1,1)
    
    if y.ndim == 1:
        y = y.reshape(-1,1)
        
    if np.isscalar(coefficient) == True:
        coefficient = np.array([coefficient])
        
    if coefficient.ndim == 1:
        coefficient = coefficient.reshape(-1,1)
    
    # total number of features and training examples
    m,n = X.shape

    # Initialization of Variables 
    db = 0
    dw = 0
    temp_db = 0
    temp_dw = 0

    # Matrix computation
    fx = ((X@coefficient) + intercept)
    temp_dw = (fx - y) * X
    temp_db = (fx - y)
    
    db = temp_db.mean() 
    dw = temp_dw.mean(axis=0).reshape([n,1]) 

    # Compute Regularization Term
    regu = coefficient * (reguLambda/m) 

    dwRegu = dw + regu

    return db, dwRegu



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



#######################################################################################################
## Logistic Regression
#######################################################################################################

def sigmoid(z):
    '''
    Sigmoid Function
    z = f(x) a matrix of m by 1, each training examples in rows (m) and fx in 1 column, 

    Return:
    Sigmoid (scalar)
    '''

    # there will be overflow error if we have data that is less than -709
    # to overcome this error we use np.clip to limit the data range from -700 to 700
    # The result will still be between 0 to 1
    z = np.clip(z, -700, 700)
    #print(z)
    sigmoid = 1 / (1 + np.exp(-z))
    
    return sigmoid


def compute_logistic_cost(X, y, coefficient, intercept, reguLambda = 0):
    '''
    Logistics cost function for both single and multiple features
    X = matrix of training data, each training examples in rows (m) and features in column (n), 
        Single feature data must be in m by 1 vector, where m is total number of training examples.
    y = m by 1 vector, where m is total number of training examples.
    coefficient (w) = n by 1 vector, where n is total number of features
    intercept (b) = scalar
    reguLambda = default is 0, no regularization, please enter appropriate lambda value for regularization

    Return:
    Cost (scalar)
    '''

    ### the following section convert 1d array to 2d array for ease of computation ###
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
        
    if isinstance(y, pd.DataFrame):
        y = y.to_numpy()
        
    if X.ndim == 1:
        X = X.reshape(-1,1)
    
    if y.ndim == 1:
        y = y.reshape(-1,1)
        
    if np.isscalar(coefficient) == True:
        coefficient = np.array([coefficient])
        
    if coefficient.ndim == 1:
        coefficient = coefficient.reshape(-1,1)


    m,n = X.shape


    z = np.matmul(X,coefficient) + intercept
    sig = sigmoid(z)
    cost = - (y * np.log(sig + 1e-20) + (1-y)*np.log(1-sig + 1e-20)).mean()
    
    # lambda/2m * sum(w^2)
    regu_cost = (reguLambda/(2 * m)) * (coefficient**2).sum() 
    
    return cost + regu_cost



def compute_logistic_gradient(X, y, coefficient, intercept, reguLambda=0.):
    '''
    Compute gradient for each step size
    X = matrix of training data, each training examples in rows and features in column, 
        Single feature data must be in m by 1 vector, where m is total number of training examples.
    y = m by 1 vector, where m is total number of training examples.
    w = n by 1 vector, where n is total number of features
    b = scalar (default = 0)
    reguLambda = scalar (default = 0), no regularization if default. Please enter appropriate lambda value for regularization.

    Return:
    db (scalar)
    dw (n by 1 vector)
    '''
    
    ### the following section convert 1d array to 2d array for ease of computation ###
    if X.ndim == 1:
        X = X.reshape(-1,1)
    
    if y.ndim == 1:
        y = y.reshape(-1,1)
        
    if np.isscalar(coefficient) == True:
        coefficient = np.array([coefficient])
        
    if coefficient.ndim == 1:
        coefficient = coefficient.reshape(-1,1)

    ### The following will check if the size match
    if X.shape[0] != y.shape[0]:
        print('Error, the size of X and y does not match!')
        return 
    
    if coefficient.shape[0] != X.shape[1]:
        print('Error, the size of X features does not match with size of w (coefficient)!')
        print(coefficient.shape)
        print(X.shape[1])
        return
    
    # total number of features and training examples
    m,n = X.shape

    # Initialization of Variables 
    db = 0
    dw = 0
    temp_db = 0
    temp_dw = 0

    # Compute Sigmoid 
    fx = ((X@coefficient) + intercept)
    sig = sigmoid(fx)

    temp_dw = (sig - y) * X
    temp_db = (sig - y)
    
    db = temp_db.mean() 
    dw = temp_dw.mean(axis=0).reshape(-1,1) #.mean() function return a 1 d array

    # Compute Regularization Term
    regu = coefficient * (reguLambda/m) 

    dwRegu = dw + regu

    return db, dwRegu

def predict_logistic(X, coefficient, intercept, prob=False):
    '''
    Make Logistic prediction

    INPUT:
    X = matrix of training data, each training examples in rows and features in column, 
        Single feature data must be in m by 1 vector, where m is total number of training examples.
    coefficient = The trained coefficient/weights 
    intercept = The intercept
    prob = return probability instead of True/False (default: False)

    RETURN:
    (If proba = False ) y_predict = True or False if prediction over 0.5
    (If proba = True) proba = the probability of teh prediction.
    '''
    
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    if X.ndim == 1:
        X = X.reshape(-1,1)
        
    if np.isscalar(coefficient) == True:
        coefficient = np.array([coefficient])
        
    if coefficient.ndim == 1:
        coefficient = coefficient.reshape(-1,1)
    
    z = np.matmul(X,coefficient) + intercept
    #print(z)
    
    proba = sigmoid(z)
    #print(proba)
    
    y_predict = proba>=0.5
    y_predict = y_predict.astype(int)
    if prob == False:
        return y_predict
    elif prob == True:
        return proba


#######################################################################################################
## One-vs-Rest Logistic Regression
#######################################################################################################

def ovrLogisticRegression(X, y, gradient_function = compute_logistic_gradient, cost_function = compute_logistic_cost, init_coefficient = None, 
                             init_intercept=0., iterations=10000, alpha=0.01, reguLambda = 0., 
                             printProgress=False, printProgressRate = 1000, storeHistory=False ):

    m,n = X.shape
    number_class = np.unique(y)   
    GD_result = {}
    
    for c in number_class:
        y_class = np.where(y == c,1,0)
        print('Classifying Class:',c)
        coef, intercept, _, _, _ = compute_gradient_descent(X, y_class, gradient_function = gradient_function, cost_function = cost_function, 
                                                               init_coefficient = init_coefficient, init_intercept = init_intercept, 
                                                               iterations = iterations, alpha = alpha, reguLambda = reguLambda, 
                                                               printProgress = printProgress, printProgressRate = printProgressRate, 
                                                               storeHistory = storeHistory) 
        GD_result[c] = {'coef':coef, 'intercept':intercept}

    result = GD_result
    return result

def ovrMultiClassPredict(X, GD_result, prob=True):

    number_class = GD_result.keys()
    first = True

    for c in number_class: 
        #print(c)
        proba = predict_logistic(X, GD_result[c]['coef'], GD_result[c]['intercept'], prob=prob)
        if first:
            y_predict_all = np.array(proba)
            first = False
            #print('shape',y_predict_all.shape)
        else:
            #print(c)
            y_predict_all = np.hstack((y_predict_all, proba))
            #print(y_predict_all.shape)

    #print('shape_all',y_predict_all.shape)
    y_predict = np.argmax(y_predict_all, axis = 1)
    #print('yredit',y_predict.shape)
    y_proba = np.max(y_predict_all, axis = 1)

    return y_predict, y_proba

#######################################################################################################
## Gradient Descent
#######################################################################################################

# Run Gradient Descent  
def compute_gradient_descent(X, y, gradient_function = compute_gradient, cost_function = cost_function, init_coefficient = None, 
                             init_intercept=0., iterations=10000, alpha=0.01, reguLambda = 0., 
                             printProgress=False, printProgressRate = 1000, storeHistory=False):
    '''
    Runs Gradient Descent

    Compulsory Input
    X = matrix of training data, each training examples in rows and features in column, 
        Single feature data must be in m by 1 vector, where m is total number of training examples.
    y = m by 1 vector, where m is total number of training examples.
    
    Optional Input with Defauls
    gradient_function = enter name of function that compute Logistic gradient (default is compute_gradient which is meant for Linear Regression)
    cost_function = enter name of function that compute Logistic cost (default is cost_function which is meant for Linear Regression)
    init_coefficient = initial w value (type: n by 1 vector, where n is total number of features) (default:None)
    init_intercept = initial b value (type: scalar) (default:0.)
    iterations = total number of runs for the gradient descent (default: 10,000)
    alpha = learning rate / step size (default:0.01)
    reguLambda = regularization control parameter, lambda (default:0.)

    Print Progress Options
    printProgress = To print the details while running gradient descent (type:Boolean) (default: False)
    printProgressRate = To print the details every n iterations (default:1000)
    storeHistory = To record all coefficient and intercept history (default:False) [Please note that if turn on, this may slow down the process] 

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


    ### Check for init_coefficient separately as the dimension of w depends on X
    ### Also perform copy to avoid changing the original array

    if init_coefficient is None:
        init_coefficient = np.zeros([X.shape[1],1])

    if isinstance(init_coefficient, pd.Series):
        init_coefficient = init_coefficient.copy().to_frame()

    if isinstance(init_coefficient, pd.DataFrame):
        init_coefficient = init_coefficient.copy().to_numpy()

    if init_coefficient.ndim == 1:
        init_coefficient = init_coefficient.copy()
        init_coefficient = init_coefficient.reshape(-1,1)
    
    ### The following will check if the size match
    if X.shape[0] != y.shape[0]:
        print('Error, the size of X and y does not match!')
        return 
    
    if init_coefficient.shape[0] != X.shape[1]:
        print('Error, the size of X features does not match with size of w (coefficient)!')
        print('coefficient:',init_coefficient.shape)
        print('X Features:',X.shape[1])
        return

    # Initialization of variables
    m,n = X.shape
    
    db = 0
    dw = 0

    b = init_intercept
    w = init_coefficient
    
    cost_history = []
    w_history = np.zeros((1,n))
    b_history = []

    
    for j in range(iterations):

        # Compute Partial Derivatives
        db, dw = gradient_function(X,y,w,b,reguLambda)

        b = b - (alpha * db)
        w = w - (alpha * dw)
        
        cost = cost_function(X,y,w,b,reguLambda)

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


    print(f"iteration {j}: cost = {cost:.4e}: intercept = {b:.4e}: weights = {w_convert.flatten()}")    
    print('best w', np.round(w.flatten(),4))
    print('best b', np.round(b,4)) 

    return w, b, cost_history, w_history[1:], b_history


#######################################################################################################
## Feature Scaling
#######################################################################################################

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


