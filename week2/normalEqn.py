import numpy as np

def normalEqn(X_train, y_train):
    XT = X_train.T
    XTX = np.matmul(XT,X_train)
    XTy = np.matmul(XT,y_train)
    return np.matmul(np.linalg.pinv(XTX), XTy)