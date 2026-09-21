from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


######################################
# The Poisson process                #
######################################

# simulate a Poisson process {N(t), t >= 0} with rate lambda,

def simPoissonProcess(lam, t):    # lambda is a reserved word
    expDist = stats.expon(scale=1/lam)
    time = expDist.rvs()
    nT = 0
    while time < t:
        nT += 1
        time += expDist.rvs()
    return nT

# simulate N(t) for t=10 and lambda=2
n = 10000  # number of runs
results = [simPoissonProcess(2, 10) for _ in range(n)]
print(np.mean(results))
print(np.var(results))
plt.figure()
plt.hist(results, bins=np.arange(-0.5, max(results)+0.51, 1), rwidth=0.8, density=True)
# Note the 0.51 instead of 0.5, to include the last bin in arange

# theory: this is Poisson(2*10) distributed
x = np.arange(0, np.max(results)+1)  # +1 to include the last number in arange
plt.plot(x, stats.poisson(2*10).pmf(x), 'bo')
plt.show()

# Visualise a sample path of a Poisson process with lambda=2 up to time T

# This function does not just return N(T), but it returns ALL arrival times up to time T
def simulatePoissonProcessSamplePath(lam, T) :
    arrivalTimes = np.array([])  
    expDist = stats.expon(scale=1/lam)
    t = expDist.rvs()
    while(t < T) :
        arrivalTimes = np.append(arrivalTimes, t)
        t += expDist.rvs()
    return arrivalTimes


# Create a sample path of one run
lam = 2
T = 10
# arrival times
arrivals = simulatePoissonProcessSamplePath(lam, T)
# Total number of arrivals at time T
NT = len(arrivals)

# create a plot: note that we add the point (0, 0)
ys = range(0, len(arrivals) + 1)
ys = np.append(ys, [len(arrivals)]) # point at t=10
xs = np.append(np.append([0], arrivals), [T])  # t = 100
plt.figure()
plt.step(xs,ys, 'b', where='post')
plt.show()

# Note that 'diff' converts arrival times to interarrival times:
interArrivalTimes = np.diff(arrivals) 
print(interArrivalTimes)
# cumsum does the opposite:
arrivalTimes = np.cumsum(interArrivalTimes) 
print(arrivalTimes)

