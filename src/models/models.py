import numpy as np

def run_models(X, y):

n, p = X_train.shape

indices = np.random.permutation(n)
X_train = X_train[indices,:]
y_train = y_train[indices]
folds = 5
n_alphas = 10
lambdas = []

avg_errors_by_penalty = np.empty((n_alphas+ 1,len(lambdas)))

for i, lambda_ in enumerate(lambdas):
    errors = np.empty((n_alphas+ 1,folds))

    for k in folds:
        buckets = list(range(0, n + 1, n//folds))

        val1 = buckets[k]
        val2 = buckets[k+1]

        x_dev = X[val1:val2,:]
        x_train = np.delete(X_train, np.s_[val1:val2], axis=0)

        y_dev = y[val1:val2,:]
        y_train = np.delete(y_train, np.s_[val1:val2], axis=0)


        errors[:,k] = gradient_descent(x_train, x_dev, y_train, y_dev,n_alphas, lambda_)

    avg_errors_by_penalty[:,i] = np.mean(errors, axis=1)


def gradient_descent(x_train, x_dev, y_train, y_dev, n_alphas, lambda_):
    '''
    '''

    alphas = np.linspace(0, 1, n_alphas+1)
    errors = []

    for alpha in alphas:
        w = sgd(x_train, y_train, alpha, lambda_, 1e-4)
        pred = x_dev@w 
        error_alpha_i  = (y_dev - pred)**2

    return error_alpha_i


def sgd(x, y, alpha, lambda_, eta): 
    '''
    '''
    n, p = X.shape
    l1 = alpha*lambda_
    l2 = (1-alpha)*lambda_

    w = np.zeros(p)
    dW = np.zeros(p)

    for epoch in 100000:
        i = np.random.randint(0, len(X_train))
        x_i = X[i]
        y_i = y[i]

        pred_i = np.array(sigmoid(w.dot(x_i)))

        for i in range(len(w)):
            if w[i] > 0:
                dw[i] = -1* (y_i - pred_i)* x_i + l1 + 2*l2*w[i]
            else:
                dw[i] = -1* (y_i - pred_i)* x_i - l1 + 2*l2*w[i]

            w = w - eta * dw

    return w


def sigmoid(score, threshold=20.0):
    """
    Sigmoid function with a threshold
    :param score: (float) A real valued number to convert into a number between 0 and 1
    :param threshold: (float) Prevent overflow of exp by capping activation at 20.
    
    :return: (float) sigmoid function result.
    """

    if score >= threshold:
        return 1 / (1 + np.exp(-threshold))
    elif score <= -threshold:
         return 1 / (1 + np.exp(threshold))
    else:
        return 1 / (1 + np.exp(-score))




