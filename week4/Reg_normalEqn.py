import numpy as np

def Reg_normalEqn(X, y, λ):
    XTX = np.matmul(X.T, X)

    # make identity matrix excluding the bias feature in normalizing
    n = np.size(X,1)
    id_mtx = np.eye(n)
    id_mtx[0,0] = 0

    λI = λ * id_mtx

    return np.linalg.pinv(XTX + λI) @ X.T @ y