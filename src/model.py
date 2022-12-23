import numpy as np


def gradient_descent(X_train, X_val, y_train, y_val, n_alphas, lambda_, eta, epochs):
    '''Implement stochastic gradient descent to find optimal model weights
    for every alpha for a given lambda and return the validation average log loss.

    Inputs: 
        X_train (Numpy Matrix): X for training
        X_val (Numpy Matrix): X for validation
        y_train (Numpy Matrix): true labels for training data
        y_val (Numpy Matrix): true labels for test data
        n_alphas (int): number of alphas to test 
        lambda_ (float): regulation parameter
    
    Returns: 
        loss_alpha (Numpy Array): average log loss for each alpha value.
    '''

    alphas = np.linspace(0, 1, n_alphas+1)
    errors = []
    loss_alpha = np.zeros(n_alphas+1)

    for i, alpha in enumerate(alphas):
        w = sgd(X_train, y_train, alpha, lambda_, eta, epochs)
        pred = (1 / (1 + np.exp(-X_val.dot(w)))).A1
        errors = (-y_val*np.log(pred)) - ((1-y_val)*np.log(pred))
        loss_alpha[i]  = np.mean(errors)

    return loss_alpha


def sgd(X, y, alpha, lambda_, eta, epochs): 
    '''Implement stochastic gradient descent and returns optimal weights.

    Inputs: 
        X (Numpy Matrix): predictors
        y (Numpy Array): true label (0 or 1)
        lambda (float): regularization parameter
        eta (float): gradient descent step size
        epochs (int): number of steps in gradient descent
    
    Returns:
        w (Numpy Array): optimal weights
    '''
    n, p = X.shape
    l1 = alpha * lambda_
    l2 = (1 - alpha) * lambda_

    w = np.random.rand(p)
    dw = np.zeros(p)
    for _ in range(epochs):
        i = np.random.randint(0, n)
        x_i = X[i]
        y_i = y[i]
        pred_i = 1 / (1 + np.exp(-x_i.dot(w)))
        dw = -1*(y_i - pred_i)*x_i + l1*np.sign(w) + l2*w
        w = w - eta*dw
        w = w.A1

    return w


def predict(X_train, y_train, X_test, alpha, lambda_, eta, epochs):
    '''Train model and make predictions for test data. 

    Inputs: 
        X_train (Numpy Matrix): training data
        y_train (Numpy Matrix): true labels of training data
        X_test (Numpy Matrix): test data to make predictions for
        alpha (float): ratio of l1 v l2 regularization)
        lambda_ (float): regularization parameter
        eta (float): gradient descent step size
        epochs (int): number of steps in gradient descent

    Returns: 
        pred (Numpy Array): predictions for test data
    '''
    w = sgd(X_train, y_train, alpha, lambda_, eta, epochs)
    pred = (1 / (1 + np.exp(-X_test.dot(w))))

    return pred


def pred_to_accuracy(y_pred, y_test, threshold):
    '''Calculate prediction accuracy. 

    Input: 
        y_pred (Numpy Array): predicted y
        y_test (Numpy Array): true labels
        threshold (Numpy Array): threshold to turn the predicted y into predicted labels
    
    Returns: 
        accuracy (float)
    '''
    pred_label = np.where(y_pred > threshold, 1, 0)
    errors = abs(y_test - pred_label)
    accuracy = 1 - np.mean(errors)

    return accuracy


def get_best_vals(avg_loss_array, lambdas):
    '''Select the best performing lambda and alpha for regularization. 

    Inputs: 
        avg_loss_array (Numpy Matrix): 
        lambdas (list of float)
    
    Returns:
        ridge_lambda (float): best performing lambda for ridge regularization
        lasso_lambda (float): best performing lambda for lasso regularization
        en_lambda (float): best performing lambda for elastic net regularization
        en_alpha (float): best performing alpha for elastic net regularization
    '''
    n = len(lambdas)
    ind_ridge = (np.where(avg_loss_array[0] == avg_loss_array[0].min())[0][0])
    ind_lasso = (np.where(avg_loss_array[10] == avg_loss_array[-1].min())[0][0])

    enl_matrix = avg_loss_array[1:n-1]
    alpha_idx, en_lambda_idx = np.where(enl_matrix == np.min(enl_matrix))
    alpha_idx += 1

    ridge_lambda = lambdas[ind_ridge]
    lasso_lambda = lambdas[ind_lasso]
    en_lambda = lambdas[en_lambda_idx[0]]
    en_alpha = alpha_idx[0] * 0.1

    return  ridge_lambda, lasso_lambda, en_lambda, en_alpha
