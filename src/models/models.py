import numpy as np


def gradient_descent(X_train, X_val, y_train, y_val, n_alphas, lambda_):
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
        w = sgd(X_train, y_train, alpha, lambda_, 1e-3)
        pred = (1 / (1 + np.exp(-X_val.dot(w)))).A1
        errors = (-y_val*np.log(pred)) - ((1-y_val)*np.log(pred))
        loss_alpha[i]  = np.mean(errors)

    return loss_alpha


def sgd(X, y, alpha, lambda_, eta): 
    '''Implement stochastic gradient descent and returns optimal weights.

    Inputs: 
        X (Numpy Matrix): predictors
        y (Numpy Array): true label (0 or 1)
        lambda (float): regularization parameter
        eta (float): gradient descent step size
    
    Returns:
        w (Numpy Array): optimal weights
    '''
    n, p = X.shape
    l1 = alpha * lambda_
    l2 = (1 - alpha) * lambda_

    w = np.ones(p)
    dw = np.zeros(p)
    for _ in range(10000):
        i = np.random.randint(0, n)
        x_i = X[i]
        y_i = y[i]
        pred_i = 1 / (1 + np.exp(-x_i.dot(w)))
        dw = -1*(y_i - pred_i)*x_i + l1*np.sign(w) + 2*l2*w
        w = w - eta*dw
        w = w.A1

    return w
