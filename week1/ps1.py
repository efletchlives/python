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

# a)
mean = 2.4
std_dev = 0.75
x = mean + std_dev * np.random.randn(1000000,1)

# b)
z = -2 + (1 + 2) * np.random.rand(1000000,1)

# c)
x_hist = np.histogram(x) # since x is already normalized


# d)
x_1 = x
for i in range(np.size(x)):
    x_1[i] = x[i] + 2
    # figure out how to time operations

# e)
x_2 = x
add_vector = np.ones(1000000) * 2
x_2 = x + add_vector # using vector adding instead


# 4) Linear Algebra

# a)
A = np.array([2,10,8],[3,5,2],[6,4,4])
# find min value in each column, max value in each row, smallest value in A,
# vector that contains sum of each row in A, sum of all elements in A,
# compute matrix B whose size is the same as A but elements are the square of the corresponding elements in A

# b)
A = np.array([2,5,-2],[2,6,4],[6,8,18])
b = np.array([12],[6],[15])
# solve the system of equations

# c)
# L1-norm and L2-norm for vectors x1 and x2



# 5) Splitting Data








