import os
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


######################################
# Distribution fitting               #
######################################

# Import the data set

print(os.getcwd())  # print the working directory
#os.chdir('..') # change working directory if necessary

# Import a tab delimited text file
data = pd.read_csv('data/skulls.txt', header=0)  # header row is row 0 (the first row). Use header=None if there is no header
skulls = data['skulls']  # put the column values in an array
print(skulls)

# Import the Excel file instead
# You need the Python library openpyxl:
#    pip install openpyxl

data = pd.read_excel('data/skulls.xlsx', header=0)  # header row is row 0 (the first row). Use header=None if there is no header
skulls = data['skulls']  # put the column values in an array
print(skulls)


# Make a plot
plt.figure()
plt.plot(skulls, 'ob')
# Make a histogram
plt.figure()   # create a new plot window
plt.hist(skulls, bins=10, rwidth=0.8, density=True)
# explicit definition of the bins:
plt.figure()
plt.hist(skulls, bins=np.arange(110, 144, 3), rwidth=0.8, density=True)

# Fit a normal distribution

# First and second moment
M1 = np.mean(skulls)      # first moment
M2 = np.mean(skulls**2)   # second moment

# Estimates for mu and sigma^2
mu = M1                
sigma2 = M2 - M1**2
fitNormDist = stats.norm(mu, np.sqrt(sigma2))

# Add theoretical density
xs = np.arange(np.min(skulls), np.max(skulls), 0.1)
ys = fitNormDist.pdf(xs)
plt.plot(xs, ys, color='red')

# Plot of the empirical distribution function and
# the theoretical distribution function.

# Method 1: write it yourself
ecdfx = np.sort(skulls)
ecdfy = np.arange(1, len(skulls)+1) / len(skulls)
plt.figure()
plt.step(ecdfx, ecdfy, color='blue', where='post')
plt.plot(xs, fitNormDist.cdf(xs), color='red')

# Method 2: using Python function ECDF
ecdf = ECDF(skulls)
plt.figure()
plt.step(ecdf.x, ecdf.y, color='black', where='post')
plt.plot(xs, fitNormDist.cdf(xs), color='b')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(skulls, fitNormDist.cdf)
print('KS Test Normal distribution: ' + str(tst1))
# Test statistic: 0.104, P-value: 0.866

# Shapiro-Wilk test for normality
tst2 = stats.shapiro(skulls)
print('Shapiro-Wilk Test: ' + str(tst2))
# Test statistic: 0.981, P-value: 0.860


# fit a gamma distribution
alpha = M1**2 / (M2 - M1**2)
beta = M1 / (M2 - M1**2)
fitGammaDist = stats.gamma(alpha, scale=1/beta)
plt.plot(xs, fitGammaDist.cdf(xs), color='r')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(skulls, fitGammaDist.cdf)
print('KS Test Gamma distribution: ' + str(tst1))

# fit an exponential distribution
lam = 1/M1
fitExpDist = stats.expon(scale=1/lam)
plt.plot(xs, fitExpDist.cdf(xs), color='g')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(skulls, fitExpDist.cdf)
print('KS Test Exponential distribution: ' + str(tst1))

# fit a uniform distribution 
# (Careful: we are NOT using the method of moments here!!!!!)
a = min(skulls)
b = max(skulls)
fitUnifDist = stats.uniform(loc=a, scale=b - a) # Careful! look at definition!
plt.plot(xs, fitUnifDist.cdf(xs), color='orange')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(skulls, fitUnifDist.cdf)
print('KS Test Uniform distribution: ' + str(tst1))

# Exercise: use the method of moments to estimate a and b.
# Solution:
a = M1 - np.sqrt(3*(M2-M1**2))
b = M1 + np.sqrt(3*(M2-M1**2))

plt.show()

tst1 = stats.kstest(skulls, fitExpDist.cdf)
print('KS Test Exponential distribution: ' + str(tst1))

