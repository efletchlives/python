# 1. Power Consumption Pricing
# a) X would be the power used on a said day in a month in a year for many years. 

# b) Y would be the price of electric per kwh on that day

# c) Using previous discrete points several years in the past, 
# you can train the model to predict power consumption in the future.

# d) Choosing the level of regression (linear, polynomial, logarithmic) is a challenging part
# because you want it to fit to the curve as best as possible especially for power consumption.
# This typically decides what the power company will charge you for electric in the future.

# 2. Identifying different types of cars in photos
# a) X would be the photos of certain types of cars in them

# b) Y would be a string of what brand car is in the photo

# c) I would take photos from the internet of the cars on the road and then create the output 
# by looking at the photos and writing down which brand it is (Toyota, Honda, Mercedes, BMW)

# d) Since there is a lot of variation within car brands, I would need an emormous dataset.



# 3) Basic Operations
import numpy as np

# a) done
mean = 2.4
std_dev = 0.75
x = mean + std_dev * np.random.randn(1000000,1)

# b) done
z = -2 + (1 - -2) * np.random.rand(1000000,1)

# c) done
import matplotlib.pyplot as plot

# density=True means the histogram is normalized
plot.hist(x, bins=100, density=True, color="blue")
plot.title("Normalized Gaussian Distribution of x")
plot.savefig("ps1-3-c-1.png")
plot.close() # to prevent plotting this graph again

# it does look like a gaussian distribution

plot.hist(z, bins=100, density=True, color="red")
plot.title("Normalized Uniform Distribution of z")
plot.savefig("ps1-3-c-2.png")
plot.close() # to prevent plotting this graph again

# it does look like a uniform distribution


# d) done
import time as t

x_1 = x

# get start time
t_start = t.time()

for i in range(np.size(x)):
    x_1[i] = x[i] + 2

# get end time
t_end = t.time()

print("execution time of d:", t_end-t_start, "seconds")

# e) done

x_2 = x

# get start time
t_start = t.time()

add_vector = np.ones((1000000,1)) * 2
x_2 = x + add_vector # using vector adding instead

# get end time
t_end = t.time()

print("execution time of e:", t_end-t_start, "seconds")

# f) done
y_list = []
for i in range(np.size(z)):
    if(z[i] > 0 and z[i] < 0.8):
        y_list.append(z[i])

y = np.array(y_list)

print("f) elements retrieved:",np.size(y)) 
# first time: 266869 
# second time: 266818



# 4) Linear Algebra
# a)
A = np.array([[2,10,8],[3,5,2],[6,4,4]])

# find min value in each column
np.min(A, axis=0) # axis=0 means work down columns

# find max value in each row
np.max(A, axis=1) # axis=1 means work across rows

# find smallest value in A
A.min()

# vector that contains sum of each row in A
A_vector = A.sum(axis=1) # axis=1 means work across rows

# sum of matrix
A_sum = A.sum()

# compute matrix B whose size is the same as A but elements are the square of the corresponding elements in A
B = A
B = B**2

# b)
# put in Ax = b format
A = np.array([[2,5,-2],[2,6,4],[6,8,18]])
b = np.array([[12],[6],[15]])

x = np.linalg.solve(A,b)
print("x =", x)

# c)
# L1-norm and L2-norm for vectors x1 and x2
x1 = np.array([-4, 0, 1])
l1 = 0
for i in x1:
    l1 += abs(i)

print("x1 l1 found:",l1)
print("x1 norm l1:",np.linalg.norm(x1, 1))

l2 = 0
for i in x1:
    l2 += abs(i)**2

l2 = l2**0.5

print("x1 l2 found:",l2)
print("x1 norm l2:",np.linalg.norm(x1,2))

x2 = np.array([-2, -2, 0])
l1 = 0
for i in x2:
    l1 += abs(i)

print("x2 l1 found:",l1)
print("x2 norm l1:",np.linalg.norm(x2, 1))

l2 = 0
for i in x2:
    l2 += abs(i)**2

l2 = l2**0.5

print("x2 l2 found:",l2)
print("x2 norm l2:",np.linalg.norm(x2,2))



# 5) Splitting Data
X = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7],[8,8,8],[9,9,9],[10,10,10]])
print("X = \n",X)

y = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
print("\n")

# since repeat 3 times
for i in range(1,4):
    index = np.arange(10)
    np.random.shuffle(index)
    for ind in range(10):
        if(0 <= ind < 8):
            X_train_idx = index[ind]
        else:
            X_test_idx = index[ind]
    
    X_train = X[X_train_idx]
    X_test = X[X_test_idx]

    y_train = y[X_train_idx]
    y_test = y[X_test_idx]

    print("Iteration",i,"\n")
    print("X_train:",X_train)
    print("X_test:",X_test)
    print("y_train:",y_train)
    print("y_test:",y_test)
    print("\n")