# Test Cost Function Script
import math
import numpy as np
import pandas as pd

## The following is a direct copy of code from DeepLearning.AI
## This code is used to double check the result on our own function
## Thus this code is only use in testing
def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """

    m  = X.shape[0]
    n  = len(w)
    cost = 0.
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b                                   #(n,)(n,)=scalar, see np.dot
        cost = cost + (f_wb_i - y[i])**2                               #scalar             
    cost = cost / (2 * m)                                              #scalar  
 
    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar
    
    total_cost = cost + reg_cost                                       #scalar
    return total_cost                                                  #scalar

## The following is a direct copy of code from DeepLearning.AI
## This code is used to double check the result on our own function
## Thus this code is only use in testing
def compute_gradient_linear_reg(X, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]                 
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]               
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m   
    
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw

def test_cost_function(myCostFunction):

        print("Testing Cost Function", myCostFunction)
        print("\n")
        # Test Case 1
        print("Test Case 1: Testing....")
        X = np.array([1.,2.,3.],dtype=np.float64)
        y = np.array([1.,2.,3.],dtype=np.float64)
        w = 0
        b = 0

        myCost = myCostFunction(X,y,w,b)
        stdCost = compute_cost_linear_reg(X, y, np.array(w).reshape(-1,1), b, lambda_ = 0)
        print("Test Result: Test Case 1")
        print("X = ", X)
        print("y = ", y)
        print("w = ", w)
        print("b = ", b)
        print("Test Cost Function:", myCost)
        print("Expected Cost Function:", stdCost)
        print("\n")

        
        # Test Case 2
        print("Test Case 2: Testing....")
        x2 = np.array([1.0, 2.0])   #features
        y2 = np.array([300.0, 500.0])   #target value
        w = 0
        b = 0

        myCost = myCostFunction(x2,y2,w,b)
        stdCost = compute_cost_linear_reg(x2, y2, np.array(w).reshape(-1,1), b, lambda_ = 0)
        print("Test Result: Test Case 2")
        print("X = ", x2)
        print("y = ", y2)
        print("w = ", w)
        print("b = ", b)
        print("Test Cost Function:", myCost)
        print("Expected Cost Function:", stdCost)
        print("\n")

        
        # Test Case 3
        print("Test Case 3: Testing....")
        x3 = np.array([1,2,3]).reshape((3,1))
        y3 = np.array([3,4,5]).reshape((3,1))
        w = 0
        b = 0

        myCost = myCostFunction(x3,y3,w,b)
        stdCost = compute_cost_linear_reg(x3, y3, np.array(w).reshape(-1,1), b, lambda_ = 0)
        print("Test Result: Test Case 3")
        print("X = ", x3)
        print("y = ", y3)
        print("w = ", w)
        print("b = ", b)
        print("Test Cost Function:", myCost)
        print("Expected Cost Function:", stdCost)
        print("\n")

        

        # Test Case 4
        print("Test Case 4: Testing....")
        X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]]).reshape([3,4])
        y_train = np.array([460, 232, 178]).reshape([3,1])
        w = np.zeros((4,1))
        b = 0

        myCost = myCostFunction(X_train,y_train,w,b)
        stdCost = compute_cost_linear_reg(X_train, y_train, np.array(w).reshape(-1,1), b, lambda_ = 0)
        print("Test Result: Test Case 4")
        print("X = ", X_train)
        print("y = ", y_train)
        print("w = ", w)
        print("b = ", b)
        print("Test Cost Function:", myCost)
        print("Expected Cost Function:", stdCost)
        print("\n")


        
        # Test Case 5
        print("Test Case 5: Testing....")
        X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
        y_train = np.array([460, 232, 178])
        b_init = 785.1811367994083
        w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])


        myCost = myCostFunction(X_train,y_train,coefficient=w_init,intercept=b_init)
        stdCost = compute_cost_linear_reg(X_train, y_train, w_init, b_init, lambda_ = 0)
        print("Test Result: Test Case 5")
        print("X = ", X_train)
        print("y = ", y_train)
        print("w = ", w_init)
        print("b = ", b_init)
        print("Test Cost Function:", myCost)
        print("Expected Cost Function:", stdCost)
        print("\n")


        
        # Test Case 6
        print("Test Case 6: Testing....")
        np.random.seed(1)
        X_tmp = np.random.rand(5,6)
        y_tmp = np.array([0,1,0,1,0])
        w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
        b_tmp = 0.5
        lambda_tmp = 0.7

        myCost = myCostFunction(X_tmp, y_tmp, coefficient=w_tmp, intercept=b_tmp, reguLambda=lambda_tmp)
        stdCost = compute_cost_linear_reg(X_tmp, y_tmp, w=w_tmp, b=b_tmp, lambda_=lambda_tmp)
        print("Test Result: Test Case 6")
        print("X = ", X_tmp)
        print("y = ", y_tmp)
        print("w = ", w_tmp)
        print("b = ", b_tmp)
        print("Test Cost Function:", myCost)
        print("Expected Cost Function:", stdCost)
        print("\n")
        print("\n")

        
        print("Test Completed.")






def test_compute_gradient_function(myComputeGradientFunction):

        # Test Case 1
        print("Test Case 1: Testing....")
        np.random.seed(1)
        X_tmp = np.random.rand(5,3)
        y_tmp = np.array([0,1,0,1,0])
        w_tmp = np.random.rand(X_tmp.shape[1])
        b_tmp = 0.5
        lambda_tmp = 0.7

        my_db, my_dw = myComputeGradientFunction(X_tmp, y_tmp, coefficient=w_tmp, intercept=b_tmp, reguLambda=lambda_tmp)
        std_db, std_dw = compute_gradient_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_ = lambda_tmp)
        print("Test Result: Test Case 1")
        print("Test Gradients:", my_db,my_dw )
        print("Expected Gradients:", std_db, std_dw)


        
        # Test Case 2
        print("Test Case 2: Testing....")
        X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
        y_train = np.array([460, 232, 178])
        b_init = 785.1811367994083
        w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

        my_db, my_dw = myComputeGradientFunction(X_train, y_train, coefficient=w_init, intercept=b_init)
        std_db, std_dw = compute_gradient_linear_reg(X_train, y_train, w_init, b_init, lambda_=0)
        print("Test Result: Test Case 2")
        print("Test Gradients:", my_db,my_dw )
        print("Expected Gradients:", std_db, std_dw)


        
        
        print("Test Completed.")