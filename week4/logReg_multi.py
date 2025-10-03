from sklearn.linear_model import LogisticRegression
import numpy as np

def logReg_multi(X_train, y_train, X_test):
#             m x (n + 1)  m x 1  d x (n + 1)
    # create 3 separate y_c vectors from y_train
    y_1 = (y_train == 1).astype(int)
    y_2 = (y_train == 2).astype(int)
    y_3 = (y_train == 3).astype(int)

    mdl1 = LogisticRegression(random_state=0).fit(X_train,y_1.ravel())
    mdl2 = LogisticRegression(random_state=0).fit(X_train,y_2.ravel())
    mdl3 = LogisticRegression(random_state=0).fit(X_train,y_3.ravel())

    train_proba_1 = mdl1.predict_proba(X_train)[:,1]
    train_proba_2 = mdl2.predict_proba(X_train)[:,1]
    train_proba_3 = mdl3.predict_proba(X_train)[:,1]

    train_proba_1 = np.mean(train_proba_1, axis=0)
    train_proba_2 = np.mean(train_proba_2, axis=0)
    train_proba_3 = np.mean(train_proba_3, axis=0)

    test_proba_1 = mdl1.predict_proba(X_test)[:,1]
    test_proba_2 = mdl2.predict_proba(X_test)[:,1]
    test_proba_3 = mdl3.predict_proba(X_test)[:,1]

    test_proba_1 = np.mean(test_proba_1, axis=0)
    test_proba_2 = np.mean(test_proba_2, axis=0)
    test_proba_3 = np.mean(test_proba_3, axis=0)

    y_pred = np.hstack([[train_proba_1],[train_proba_2],[train_proba_3],[test_proba_1],[test_proba_2],[test_proba_3]])
    return y_pred