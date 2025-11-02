import sigmoid

def sigmoidGradient(z):
    return (sigmoid.sigmoid(z)*(1-sigmoid.sigmoid(z)))