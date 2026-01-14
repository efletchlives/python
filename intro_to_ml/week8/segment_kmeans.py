import numpy as np
import kmeans_multiple

def kmeans(im_in, K, iters, R):

    H, W, _ = im_in.shape

    # reshape the image
    X = im_in.reshape(H*W, 3)

    # perform clustering on img
    ids, means, ssd = kmeans_multiple.kmeans_multiple(X, K, iters, R)

    im_out = np.zeros_like(X) # initialize empty output image
    for k in range(K):
        im_out[ids == (k+1)] = means[k]
    
    im_out = im_out.reshape(H, W, 3)
    return im_out