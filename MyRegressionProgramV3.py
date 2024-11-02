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


#######################################################################################################
## Softmax Logistic Regression
#######################################################################################################

def softmax(fx):
    '''
    This Softmax function use the formula np.exp(X)/np.sum(np.exp(X))
    For matrix this function only accepts columns as features.
    The Softmax will work on per data observation in rows

    INPUT:
    fx = fx should be in (m,n) or (m,) 

    OUTPUT:
    softmax (matrix) if there is more than one feature. All the columns in a specific row added up to 1.
    '''
    # Taking care of infinity number
    fx = np.where(np.isinf(fx),  9.9e+307, fx)
    #print(fx)

    # if fx is 1d array, then use the following direct computation
    if fx.ndim == 1:
        softmax = np.exp(fx - np.max(fx))/(np.sum(np.exp(fx - np.max(fx))))
    else:
        # process following if it is 2D array
        m,n = fx.shape
        # If it is only one column, use the following computation
        if n == 1:
            softmax = np.exp(fx - np.max(fx))/(np.sum(np.exp(fx - np.max(fx)), keepdims=True))
        else:
            # If there is more than one column use axis = 1
            softmax = np.exp(fx - np.max(fx, axis=1, keepdims=True))/  (np.sum(np.exp(fx - np.max(fx, axis=1, keepdims=True)), axis=1, keepdims=True))
    return softmax


#######################################################################################################
## Softmax Cost Function with Regularization
#######################################################################################################

def categoricalCrossEntropyCost(X,y,W,b,reguLambda=0):
    '''
    Categorical Cross Entropy Loss for One Hot Encoded Label

    INPUT:
    X = features data in matrix form in (m, n) matrix where m is the total number of examples and n is number of features
    y = also known as y or truth label. Accepts a matrix (m,k) of one hot ecoding to each class. m is total number of examples 
        and k is total number of classes
    W = estimate W in matrix form in (n,k) where n is the number of features and k is number of classes
    b = estimate b in row vectors (1, k) where k is number of classes

    OUTPUT:
    loss = loss of each observation/example
    cost = the mean of the loss function
    '''
    
    ## Dimension checking
    if y.ndim == 1:
        print('Error, the y label must be one-hot encoded')
        return

    if X.ndim == 1:
        X = X.reshape(-1,1)

    m,n = X.shape
    k = y.shape[1]

    ## Check the shape of data input to see if it is matched
    if y.shape != (m,k):
        print('Error, y must be in m by k matrix where m = total examples as X, k is total number of class')
        return

    if W.shape != (n,k):
        print(W.shape)
        print((n,k))
        print('Error, type mismatch for W. W must be n by k matrix where n = number of features, k = number of classes')
        return

    if b.shape != (1,k):
        print('Error, type mismatch for b. b must be 1 by k matrix where k = number of classes')
        return

    if X.shape[0] != y.shape[0]:
        print('Error, the total number of observation does not match between X and y')
        return
    
    ## Make prediction
    fx = X@W + b

    ## Applied Softmax
    y_predict = softmax(fx)
    #print('softmax',softmax)

    
    ## Compute Loss Function
    loss = np.negative(np.sum((y * np.log(y_predict + 1e-20)), axis = 1))

    ## Compute Cost
    cost = loss.mean()
    
    # Compute Regularization Term
    # lambda/2m * sum(w^2)
    regu = (reguLambda/ (2*m)) * sum(W ** 2)

    totalCost = cost + regu[0]

    return totalCost


def sparseCategoricalCrossEntropyCost(X,y,W,b,reguLambda=0):
    '''
    Sparse Categorical Cross Entropy Loss for Class Index Label

    INPUT:
    X = features data in matrix form in (m, n) matrix where m is the total number of examples and n is number of features
    y = also known as y or truth label. Accepts a matrix of one hot ecoding to each class.
    W = estimate W in matrix form in (n,k) where n is the number of features and k is number of classes
    b = estimate b in row vectors (1, k) where k is number of classes

    OUTPUT:
    loss = loss each observation
    cost = the mean of the loss function
    '''

    ## Dimension checking
    if y.ndim == 1:
        y = y.reshape(-1,1)

    if X.ndim == 1:
        X = X.reshape(-1,1)

    m,n = X.shape
    k = len(np.unique(y))

    ## Check the shape of data input to see if it is matched
    if W.shape != (n,k):
        print('Error, type mismatch for W. W must be n by k matrix where n = number of features, k = number of classes')
        return

    if b.shape != (1,k):
        print('Error, type mismatch for b. b must be 1 by k matrix where k = number of classes')
        return

    if X.shape[0] != y.shape[0]:
        print('Error, the total number of observation does not match between X and y')
        return
        

    ## Make prediction
    fx = X@W + b

    ## Applied Softmax
    y_predict = softmax(fx)
    #print('softmax',softmax)

     ## Compute Loss Function
    loss = np.negative(np.log(y_predict[np.arange(m).reshape(-1,1), y] + 1e-20))

    ## Compute Cost
    cost = loss.mean()

    # Compute Regularization Term
    # lambda/2m * sum(w^2)
    regu = (reguLambda/ (2*m)) * sum(W ** 2)

    totalCost = cost + regu[0]

    return totalCost


#######################################################################################################
## Categorical Cross Entropy Logistic Regression with Gradient Descent with L2
#######################################################################################################

def categoricalSoftmaxGradient(X,y,W,b,reguLambda=0.):
    '''
    Compute gradient for each step size
    X = m by n matrix of training data, each training examples (m) in rows and features (n) in column, 
        Single feature data must be in m by 1 vector, where m is total number of training examples.
    y = m by k matrix, one-hot encoded, where m is total number of examples and k is number of classes.
    W = n by k matrix, where n is total number of features and k is number of classes.
    b = 1 by k row vector, where k is number of classes.

    Return:
    db (1 by k)
    dw (n by k matrix)
    '''
    
    ## Dimension checking
    if y.ndim == 1:
        print('Error, the y label must be one-hot encoded')
        return

    if X.ndim == 1:
        X = X.reshape(-1,1)

    m,n = X.shape
    k = y.shape[1]

    ## Check the shape of data input to see if it is matched
    if y.shape != (m,k):
        print('Error, y must be in m by k matrix where m = total examples as X, k is total number of class')
        return

    if W.shape != (n,k):
        print(W.shape)
        print((n,k))
        print('Error, type mismatch for W. W must be n by k matrix where n = number of features, k = number of classes')
        return

    if b.shape != (1,k):
        print('Error, type mismatch for b. b must be 1 by k matrix where k = number of classes')
        return

    if X.shape[0] != y.shape[0]:
        print('Error, the total number of observation does not match between X and y')
        return
    
    
    # total number of features and training examples
    m,n = X.shape

    ## Make prediction
    fx = X@W + b

    ## Applied Softmax
    y_pred = softmax(fx)

    ## Compute Gradient
    temp_dW = X.T @ (y_pred - y)
    temp_db = y_pred - y

    dW = temp_dW /m
    db = np.mean(temp_db, axis=0, keepdims=True)
    #print(db.shape)

    # Compute Regularization Term
    regu = W * (reguLambda/m) 

    dwRegu = dW + regu
    
    return db, dwRegu



def categorical_softmax_gradient_descent(X, y, k, init_coefficient = None, init_intercept= None, iterations=10000, alpha=0.01, reguLambda = 0., 
                             printProgress=False, printProgressRate = 1000, storeHistory=False):
    '''
    Runs Gradient Descent

    Compulsory Input
    X = m by n matrix of training data, each training examples (m) in rows and features (n) in column, 
        Single feature data must be in m by 1 vector, where m is total number of training examples.
    y = m by k matrix, one-hot encoded, where m is total number of examples and k is number of classes.
    k = number of class (Use as a double check)
    
    Optional Input with Defaults
    init_coefficient = n by k matrix, where n is total number of features and k is number of classes. (default=None)
    init_intercept = 1 by k row vector, where k is number of classes. (default=None)
    iterations = total number of runs for the gradient descent (default: 10,000)
    alpha = learning rate / step size (default:0.01)
    reguLambda = regularization control parameter, lambda (default:0.)

    Print Progress Options
    printProgress = To print the details while running gradient descent (type:Boolean) (default: False)
    printProgressRate = To print the details every n iterations (default:1000)
    storeHistory = To record all coefficient and intercept history (default:False) [Please note that if turn on, this may slow down the process] 

    Return:
    w = best w (n by k matrix, where n is total number of features and k is number of classes)
    b = best b (1 by k vector)
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
        X = X.reshape(-1,1)
    if y.ndim == 1:
        print('Error, y must be one hot encoded!')
        return 


    ### The following will check if the size match
    if X.shape[0] != y.shape[0]:
        print('Error, the size of X and y does not match!')
        return 

    m,n = X.shape

    ### check if the number of class is correct
    if y.shape[1] != k:
        print('Error, the number of class you entered is different from the actual one-hot encoded y label!')
        return

    m,n = X.shape
    ### Initialize W should by n by num_class
    if init_coefficient is None:
        init_coefficient = np.zeros((n,k))

    if isinstance(init_coefficient, pd.Series):
        init_coefficient = init_coefficient.copy().to_frame()

    if isinstance(init_coefficient, pd.DataFrame):
        init_coefficient = init_coefficient.copy().to_numpy()

    if init_coefficient.ndim == 1 & n == 1:
        init_coefficient = init_coefficient.copy()
        init_coefficient = init_coefficient.reshape(1,-1)

    if init_coefficient.ndim == 1 & k == 1:
        init_coefficient = init_coefficient.copy()
        init_coefficient = init_coefficient.reshape(-1,1)

    if init_coefficient.shape[1] != k:
        print('Error, the number of class k is different from the coefficient!')
        print('coefficient:',init_coefficient.shape)
        return
        
    if init_coefficient.shape[0] != X.shape[1]:
        print('Error, the size of X features does not match with size of w (coefficient)!')
        print('coefficient:',init_coefficient.shape)
        print('X Features:',X.shape[1])
        return

    ### Initialize b should by n by num_class
    if init_intercept is None:
        init_intercept = np.zeros((1,k))

    if isinstance(init_intercept, pd.Series):
        init_intercept = init_intercept.copy().to_frame()

    if isinstance(init_intercept, pd.DataFrame):
        init_intercept = init_intercept.copy().to_numpy()

    if init_intercept.ndim == 1:
        init_intercept = init_intercept.copy()
        init_intercept = init_intercept.reshape(1,-1)

    if init_intercept.shape[1] != k:
        print('Error, the number of class k is different from the intercept!')
        print('intercept:',init_intercept.shape)
        return

    # Initialization of variables
    m,n = X.shape
    
    db = 0
    dW = 0

    b = init_intercept
    W = init_coefficient
    
    cost_history = []
    W_history = []
    b_history = []

    
    for j in range(iterations):

        # Compute Partial Derivatives
        db, dW = categoricalSoftmaxGradient(X,y,W,b,reguLambda=reguLambda)
        #print(db.shape)

        b = b - (alpha * db)
        W = W - (alpha * dW)

        cost = categoricalCrossEntropyCost(X,y,W,b,reguLambda=reguLambda)

        # Reshape w for printing and storing history
        W_convert = W.copy()
        W_convert = np.transpose(W_convert)
        
        if storeHistory == True: 
            cost_history.append(cost)
            b_history.append(b)
            W_history = np.vstack((W_history,W_convert))

        if printProgress == True:
            if j % printProgressRate == 0:
                print(f"iteration {j}: cost = {cost:.6e}: intercept = {b:}: weights = {W_convert}")


    print(f"iteration {j}: cost = {cost:.6e}: intercept = {b}: weights = {W_convert}")    
    print('best w', np.round(W,6))
    print('best b', np.round(b,6)) 

    return W, b, cost_history, W_history[1:], b_history



#######################################################################################################
## Sparse Categorical Cross Entropy Logistic Regression with Gradient Descent with L2
#######################################################################################################

def sparseCategoricalSoftmaxGradient(X,y,W,b,reguLambda=0.):
    '''
    Compute gradient for each step size
    X = m by n matrix of training data, each training examples (m) in rows and features (n) in column, 
        Single feature data must be in m by 1 vector, where m is total number of training examples.
    y = m by k matrix, one-hot encoded, where m is total number of examples and k is number of classes.
    W = n by k matrix, where n is total number of features and k is number of classes.
    b = 1 by k row vector, where k is number of classes.

    Return:
    db (1 by k)
    dw (n by k matrix)
    '''
    
    ## Dimension checking
    if y.ndim == 1:
        y = y.reshape(-1,1)

    if X.ndim == 1:
        X = X.reshape(-1,1)

    m,n = X.shape
    k = len(np.unique(y))

    ## Check the shape of data input to see if it is matched
    if W.shape != (n,k):
        print(W.shape)
        print((n,k))
        print('Error, type mismatch for W. W must be n by k matrix where n = number of features, k = number of classes')
        return

    if b.shape != (1,k):
        print('Error, type mismatch for b. b must be 1 by k matrix where k = number of classes')
        return

    if X.shape[0] != y.shape[0]:
        print('Error, the total number of observation does not match between X and y')
        return
    
    
    # total number of features and training examples
    m,n = X.shape

    ## Make prediction
    fx = X@W + b

    ## Applied Softmax
    y_pred = softmax(fx)

    ## perform selection that matches the truth
    y_pred[np.arange(m).reshape(-1,1), y] -= 1


    ## Compute Gradient
    temp_dW = X.T@y_pred
    temp_db = y_pred

    dW = temp_dW /m
    db = np.mean(temp_db, axis=0, keepdims=True)
    #print(db.shape)

    # Compute Regularization Term
    regu = W * (reguLambda/m) 

    dwRegu = dW + regu

    return db, dwRegu


def sparse_categorical_softmax_gradient_descent(X, y, k, init_coefficient = None, init_intercept= None, iterations=10000, alpha=0.01, reguLambda = 0., 
                             printProgress=False, printProgressRate = 1000, storeHistory=False):
    '''
    Runs Gradient Descent Using Class Index in Label

    Compulsory Input
    X = m by n matrix of training data, each training examples (m) in rows and features (n) in column, 
        Single feature data must be in m by 1 vector, where m is total number of training examples.
    y = m by k matrix, one-hot encoded, where m is total number of examples and k is number of classes.
    k = number of class (Use as a double check)
    
    Optional Input with Defaults
    init_coefficient = n by k matrix, where n is total number of features and k is number of classes. (default=None)
    init_intercept = 1 by k row vector, where k is number of classes. (default=None)
    iterations = total number of runs for the gradient descent (default: 10,000)
    alpha = learning rate / step size (default:0.01)
    reguLambda = regularization control parameter, lambda (default:0.)

    Print Progress Options
    printProgress = To print the details while running gradient descent (type:Boolean) (default: False)
    printProgressRate = To print the details every n iterations (default:1000)
    storeHistory = To record all coefficient and intercept history (default:False) [Please note that if turn on, this may slow down the process] 

    Return:
    w = best w (n by k matrix, where n is total number of features and k is number of classes)
    b = best b (1 by k vector)
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
        X = X.reshape(-1,1)
    if y.ndim == 1:
        y = y.reshape(-1,1)


    ### The following will check if the size match
    if X.shape[0] != y.shape[0]:
        print('Error, the size of X and y does not match!')
        return 

    m,n = X.shape

    ### Initialize W should by n by num_class
    if init_coefficient is None:
        init_coefficient = np.zeros((n,k))

    if isinstance(init_coefficient, pd.Series):
        init_coefficient = init_coefficient.copy().to_frame()

    if isinstance(init_coefficient, pd.DataFrame):
        init_coefficient = init_coefficient.copy().to_numpy()

    if init_coefficient.ndim == 1 & n == 1:
        init_coefficient = init_coefficient.copy()
        init_coefficient = init_coefficient.reshape(1,-1)

    if init_coefficient.ndim == 1 & k == 1:
        init_coefficient = init_coefficient.copy()
        init_coefficient = init_coefficient.reshape(-1,1)

    if init_coefficient.shape[1] != k:
        print('Error, the number of class k is different from the coefficient!')
        print('coefficient:',init_coefficient.shape)
        return
        
    if init_coefficient.shape[0] != X.shape[1]:
        print('Error, the size of X features does not match with size of w (coefficient)!')
        print('coefficient:',init_coefficient.shape)
        print('X Features:',X.shape[1])
        return

    ### Initialize b should by n by num_class
    if init_intercept is None:
        init_intercept = np.zeros((1,k))

    if isinstance(init_intercept, pd.Series):
        init_intercept = init_intercept.copy().to_frame()

    if isinstance(init_intercept, pd.DataFrame):
        init_intercept = init_intercept.copy().to_numpy()

    if init_intercept.ndim == 1:
        init_intercept = init_intercept.copy()
        init_intercept = init_intercept.reshape(1,-1)

    if init_intercept.shape[1] != k:
        print('Error, the number of class k is different from the intercept!')
        print('intercept:',init_intercept.shape)
        return

    # Initialization of variables
    m,n = X.shape
    
    db = 0
    dW = 0

    b = init_intercept
    W = init_coefficient
    
    cost_history = []
    W_history = []
    b_history = []

    
    for j in range(iterations):

        # Compute Partial Derivatives
        db, dW = sparseCategoricalSoftmaxGradient(X,y,W,b,reguLambda=reguLambda)
        #print(db.shape)

        b = b - (alpha * db)
        W = W - (alpha * dW)

        cost = sparseCategoricalCrossEntropyCost(X,y,W,b,reguLambda=reguLambda)

        # Reshape w for printing and storing history
        W_convert = W.copy()
        W_convert = np.transpose(W_convert)
        
        if storeHistory == True: 
            cost_history.append(cost)
            b_history.append(b)
            W_history = np.vstack((W_history,W_convert))

        if printProgress == True:
            if j % printProgressRate == 0:
                print(f"iteration {j}: cost = {cost:.6e}: intercept = {b:}: weights = {W_convert}")


    print(f"iteration {j}: cost = {cost:.6e}: intercept = {b}: weights = {W_convert}")    
    print('best w', np.round(W,6))
    print('best b', np.round(b,6)) 

    return W, b, cost_history, W_history[1:], b_history


#######################################################################################################
## Softmax Prediction
#######################################################################################################

def softmax_predict(X, coefficient, intercept):

    '''
    Make Softmax Prediction

    INPUT:
    X = (m by n) matrix of training data, each training examples (m) in rows and features (n) in column, 
        Single feature data must be in m by 1 vector, where m is total number of training examples.
    coefficient = the trained coefficient/weights in (n by k) matrix 
    intercept = the trained intercept in (1 by k) row vector 

    RETURN:
    y_predict, proba
    
    where
    y_predict = True or False if prediction over 0.5
    proba = the probability of the prediction.
    '''
    
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    if X.ndim == 1:
        X = X.reshape(-1,1)
        
    ## Make prediction    
    fx = X@coefficient + intercept

    ## Applied Softmax
    soft_fx = softmax(fx)

    ## Compute Prediction and Probabilities
    y_predict = np.argmax(soft_fx, axis = 1)
    y_proba = np.max(soft_fx, axis = 1)
    
    return y_predict, y_proba