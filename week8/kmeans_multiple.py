import kmeans_single

def kmeans_multiple(X, K, iters, R):
    for r in range(R):
        ids, means, ssd = kmeans_single.kmeans_single(X, K, iters)

        if ssd < best_ssd:
            best_ssd = ssd
            best_ids = ids
            best_means = means
    
    return best_ids, best_means, best_ssd