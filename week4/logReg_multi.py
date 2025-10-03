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

    train_proba = np.vstack([mdl1.predict_proba(X_train)[:,1], 
                             mdl2.predict_proba(X_train)[:,1],
                             mdl3.predict_proba(X_train)[:,1]]).T

    test_proba = np.vstack([mdl1.predict_proba(X_test)[:,1],
                            mdl2.predict_proba(X_test)[:,1],
                            mdl3.predict_proba(X_test)[:,1]]).T

    train_pred = np.argmax(train_proba, axis=1) + 1 # 1 for bias
    test_pred  = np.argmax(test_proba, axis=1) + 1 

    # put in a dictionary for easy access
    y_pred = {
        'train_pred': train_pred,
        'test_pred': test_pred
    }

    return y_pred