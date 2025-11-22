import kmeans_single
import numpy as np

def kmeans_multiple(X, K, iters, R):
    best_ssd = np.inf
    best_ids = None
    best_means = None

    for r in range(R):
        ids, means, ssd = kmeans_single.kmeans_single(X, K, iters)

        if ssd < best_ssd:
            best_ssd = ssd
            best_ids = ids
            best_means = means
    
    return best_ids, best_means, best_ssd