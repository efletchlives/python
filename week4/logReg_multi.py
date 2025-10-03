from sklearn.linear_model import LogisticRegression
import numpy as np

def logReg_multi(X_train, y_train, X_test):
#             m x (n + 1)  m x 1  d x (n + 1)
    # create 3 separate y_c vectors from y_train
    y_1 = y_train
    y_1[y_train == 1] = 1
    y_1[y_train == 2] = 0
    y_1[y_train == 3] = 0

    y_2 = y_train
    y_2[y_train == 1] = 0
    y_2[y_train == 2] = 1
    y_2[y_train == 3] = 0

    y_3 = y_train
    y_3[y_train == 1] = 0
    y_3[y_train == 2] = 0
    y_3[y_train == 3] = 1

    mdl1 = LogisticRegression(random_state=0).fit(X_train,y_1)
    mdl2 = LogisticRegression(random_state=0).fit(X_train,y_2)
    mdl3 = LogisticRegression(random_state=0).fit(X_train,y_3)

    proba_1 = mdl1.predict_proba(X_test)[:,1]
    proba_2 = mdl2.predict_proba(X_test)[:,1]
    proba_3 = mdl3.predict_proba(X_test)[:,1]

    y_pred = np.array([[proba_1],[proba_2],[proba_3]])
    return y_pred