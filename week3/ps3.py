import numpy as np
import matplotlib.pyplot as plot
import scipy as scipy

# functions
import sigmoid
import costFunction
import gradFunction

# load data
print('1)')
print('a.')
data = np.loadtxt('/workspaces/python/week3/hw3_data1.txt',delimiter=',')
cols = np.size(data, axis=1)
X = data[:,[0,cols-2]]
y = data[:,[cols-1]]

# plot training data
print('b.')
not_admit = np.where(y == 0)[0]
admit = np.where(y == 1)[0]

plot.figure()
plot.scatter(X[admit,0], X[admit,1],marker='+',color='black',label='admitted')
plot.scatter(X[not_admit,0], X[not_admit,1],marker='o',color='yellow',label='not admitted')
plot.xlabel('exam 1 scores')
plot.ylabel('exam 2 scores')
plot.legend()
plot.savefig('/workspaces/python/week3/ps3-1-b.png')
plot.close()

# add col of ones for bias feature X0
m = np.size(X,0) # returns m - # of features
ones = np.ones((m,1))
X = np.hstack((ones, X)) # merge X0 and rest of X


# randomly divide data into training and testing data
print('c.')
indices = np.random.permutation(m) # randomize m indices
size = round(m*0.9)
idx_train = indices[:size]
idx_test = indices[size:]

# create X + y train & test data
X_train = X[idx_train]
y_train = y[idx_train]
X_test = X[idx_test]
y_test = y[idx_test]

# computes sigmoid function
print('d.')
z = np.arange(-15,15,0.01)
gz = sigmoid.sigmoid(z)

plot.figure()
plot.plot(z,gz,'o',color='blue')
plot.title('sigmoid function plot')
plot.xlabel('z')
plot.ylabel('gz')
plot.savefig('/workspaces/python/week3/ps3-1-c.png')
plot.close()

# compute cost function and gradient descent
print('e.')
X_toy = np.array([[1,1,0],[1,1,3],[1,3,1],[1,3,4]]) # m x n
y_toy = np.array([[0],[1],[0],[1]]) # m x 1
θ = np.array([1,0.9,1.2]) # 1 x n

J = costFunction.costFunction(θ, X_toy, y_toy)
grad = gradFunction.gradFunction(θ, X_toy, y_toy)
print(J)
#print(grad)

# optimize cost function with parameters θ
print('f.')
θ = np.array([0,0,0])
θopt = scipy.optimize.fmin_bfgs(costFunction.costFunction, θ, fprime=gradFunction.gradFunction, args=(X_train, y_train))
print(θopt)
θ = θopt

print('g.')
# get the x and y intercepts for the decision boundary line
x_pts = [-θ[0]/θ[1],0]
y_pts = [0,-θ[0]/θ[2]]

plot.figure()
# plot decision boundary

plot.scatter(X[admit,1], X[admit,2],marker='+',color='black',label='admitted')
plot.scatter(X[not_admit,1], X[not_admit,2],marker='o',color='yellow',label='not admitted')
plot.plot(x_pts,y_pts,'b-')
plot.xlabel('exam 1 scores')
plot.ylabel('exam 2 scores')
plot.xlim(25,105)
plot.ylim(25,105)
plot.legend()
plot.savefig('/workspaces/python/week3/ps3-1-g.png')
plot.close()

print('h.')
# compare each point to the expected point from the line
# if lower and 1, then wrong
# if higher and 0, then wrong
size = np.size(X_test,0)
wrong = 0
for i in range(size):
    if(X_test[i,2] <= ((-θ[1]*X_test[i,1]-θ[0])/θ[2]) and y_test[i,0] == 1):
        wrong = wrong + 1
    elif(X_test[i,2] >= ((-θ[1]*X_test[i,1]-θ[0])/θ[2]) and y_test[i,0] == 0):
        wrong = wrong + 1
    else:
        continue
accuracy = (size - wrong)/size # it has a 90% accuracy

print('i.')
θTx = θ[0] + θ[1]*55 + θ[2]*70
adm_prob = sigmoid.sigmoid(θTx)
print(adm_prob) # >0.5 => admitted

print('2)')
print('a.')

data2 = np.loadtxt('/workspaces/python/week3/hw3_data2.csv',delimiter=',')

n_p_og = data2[:,[0]]
n_p_sqr = n_p_og**2
profit = data2[:,1]

n_p = np.hstack((n_p_og,n_p_sqr))

# add col of ones for bias feature X0
m = np.size(data2,0) # returns m - # of features
ones = np.ones((m,1))
n_p = np.hstack((ones, n_p)) # merge X0 and rest of X

# use normal eqn to find θ parameters
n_pTprofit = np.matmul(n_p.T,profit)
XTXinv = np.linalg.pinv(np.matmul(n_p.T, n_p))
θ = np.matmul(XTXinv,n_pTprofit)
print(θ)

x_pts = np.linspace(500,1000,500)
y_pts = θ[2] * x_pts**2 + θ[1] * x_pts + θ[0]

plot.figure()
plot.plot(x_pts,y_pts,'-',color='green',label='fitted model')
plot.plot(n_p_og,profit,'o',color='orange',markerfacecolor='none',markeredgecolor='orange',label='training data')
plot.ylabel('profit')
plot.xlabel('population in thousands, n')
plot.legend()
plot.savefig('/workspaces/python/week3/ps3-2-b.png')
plot.close()
