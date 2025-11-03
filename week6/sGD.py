import numpy as np
import sigmoid
import sigmoidGradient

def sGD(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, λ, alpha, MaxEpochs):
    # randomly initialize θ1, θ2 w/o bias
    θ1 = np.random.uniform(-0.15,0.15,(hidden_layer_size,input_layer_size+1))
    θ2 = np.random.uniform(-0.15,0.15,(num_labels,hidden_layer_size+1))   

    m = X_train.shape[0]

    # turn y_labels to y vector
    y_vec = np.zeros((m, num_labels))
    for i in range(m):
        y_vec[i,y_train[i]-1] = 1 # puts 1 at the corresponding spot in the vector
    y = y_vec

    for epoch in range(MaxEpochs):
        rand_sample_idx = np.random.permutation(m)
        # perform forward pass to compute all activations 
        for i in rand_sample_idx:
            x = X_train[i]
            y = y_train[i]
            a1 = np.hstack(([1],x)) # add bias  
            z2 = np.dot(θ1,a1)
            a2 = sigmoid.sigmoid(z2)
            a2 = np.hstack(([1], a2)) # add bias

            z3 = np.dot(θ2,a2)
            a3 = sigmoid.sigmoid(z3)

            # perform backpropagation + gradient descent for each epoch
            δ3 = a3 - y
            δ2 = (θ2[:,1:].T @ δ3)  * (sigmoidGradient.sigmoidGradient(z2))

            # get gradients
            Δ2 = np.outer(δ3, a2)
            Δ1 = np.outer(δ2, a1)

            # regularization w|o bias
            D2 = Δ2
            D2[:,1:] += λ * θ2[:,1:] 
            D1 = Δ1 
            D1[:,1:] += λ * θ1[:,1:] 

            # update θ values
            θ1 -= alpha * D1
            θ2 -= alpha * D2

    print(f'epoch = {epoch}/{MaxEpochs}')

    return θ1, θ2
    