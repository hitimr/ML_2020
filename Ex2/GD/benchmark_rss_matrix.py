import numpy as np
from time import time

repeats = 1
n = int(10**4)
m = int(10)
X = np.random.rand(m,n)
y = np.random.rand(n)
w0 = np.random.rand(m)
w1 = np.random.rand(m)

def RSS_vec(x,y,w0,w1):
    return np.dot(y,y) + w1*w1*np.dot(x,x)  + len(x)*w0*w0 - 2.0*(w1*np.dot(x,y) + w0*np.sum(y) - w0*w1*sum(x))



def RSS1(X, y, w0, w1):
    sums = []
    for j in range(m):
        sum = 0
        for i in range(n):
            sum += (y[i] - (w0[j] + w1[j]*X[j][i]))**2
        sums.append(sum)
    return sums

def RSS2(X, y, w0, w1):
    return [RSS_vec(X[j].T, y, w0[j], w1[j]) for j in range(m)]




def benchmark(f):
    start_time = time()
    for n in range(repeats):
        f(X, y, w0, w1)
    end_time = time()
    return end_time - start_time




print("Python Loop 1:   ", benchmark(RSS1))
print("Python Loop 1:   ", benchmark(RSS2))

print("\nCheck if results match")
print(RSS1(X, y, w0, w1))
print(RSS2(X, y, w0, w1))