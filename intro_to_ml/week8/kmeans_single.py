import numpy as np
from scipy.spatial.distance import cdist

def kmeans_single(X, K, iters):
    m, n = X.shape

    # initialize means randomly
    means = np.zeros((K,n))
    for i in range(n):
        feat_min = np.min(X[:,i])
        feat_max = np.max(X[:,i])
        means[:,i] = np.random.uniform(feat_min, feat_max, K)
    
    ids = np.zeros(m)
    
    # perform kmeans
    for iter in range(iters):
        # assign each pt to closest cluster
        dist = cdist(X, means, metric='euclidean')
        ids = np.argmax(dist, axis=1)+1 # to move from 0-K-1 to 1-K

        # update cluster means
        for k in range(K):
            cluster_pts = X[ids == (k+1)]
            if len(cluster_pts) > 0:
                means[k] = np.mean(cluster_pts, axis=0)
    
    # compute final ssd
    ssd = 0
    for k in range(K):
        cluster_pts = X[ids == (k+1)]
        if len(cluster_pts) > 0:
            dist_to_mean = np.sum((cluster_pts - means[k])**2) # mean squared error of pts to means
            ssd += dist_to_mean

    return ids, means, ssd