import numpy as np
# returns θ
def normalEqn(X_train, y_train):
    XT = X_train.T
    XTX = np.matmul(XT,X_train)
    XTy = np.matmul(XT,y_train)
    return np.matmul(np.linalg.pinv(XTX), XTy)