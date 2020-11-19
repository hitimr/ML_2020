import numpy as np
from time import time


def RSS1(x, y, w0, w1):
    sum = 0
    for i in range(len(x)):
        sum += (y[i] - (w0 + w1*x[i]))**2
    return sum

def RSS2(x, y, w0, w1):
    sum = 0
    for i in range(len(x)):
        sum += (y[i] - (w0 + w1*x[i]))*(y[i] - (w0 + w1*x[i]))
    return sum

def RSS3(x, y, w0, w1):
    sum = 0
    for i in range(len(x)):
        sum += np.power((y[i] - (w0 + w1*x[i])),2)
    return sum

def RSS4(x, y, w0, w1):
    sum = 0
    for i in range(len(x)):
        sum += np.square((y[i] - (w0 + w1*x[i])))
    return sum

def RSS5(x,y,w0,w1):
    n = len(x)
    return np.dot(y,y) + w1*w1*np.dot(x,x) - 2*w1*np.dot(x,y) + n*w0*w0 - 2*w0*np.sum(y) + 2*w0*w1*sum(x)

# Best runtime
def RSS6(x,y,w0,w1):
    return np.dot(y,y) + w1*w1*np.dot(x,x) - 2*w1*np.dot(x,y) + len(x)*w0*w0 - 2*w0*np.sum(y) + 2*w0*w1*sum(x)


def RSS7(x,y,w0,w1):
    return np.dot(y,y) + w1*w1*np.dot(x,x)  + len(x)*w0*w0 - 2.0*(w1*np.dot(x,y) + w0*np.sum(y) - w0*w1*sum(x))


repeats = 20
n = int(10**6)
x = np.random.rand(n)
y = np.random.rand(n)
w0 = np.random.rand()
w1 = np.random.rand()

def benchmark(f):
    start_time = time()
    for n in range(repeats):
        f(x, y, w0, w1)
    end_time = time()
    return end_time - start_time

 
#print("Python Loop 1:   ", benchmark(RSS1))
#print("Python Loop 2:   ", benchmark(RSS2))
#print("Python Loop 3:   ", benchmark(RSS3))
#print("Python Loop 3:   ", benchmark(RSS4))
print("Unwrapped sum 1: ", benchmark(RSS5))
print("Unwrapped sum 2: ", benchmark(RSS6))
print("Unwrapped sum 3: ", benchmark(RSS7))

print("\nCheck if results match")
print(RSS1(x, y, w0, w1))
print(RSS2(x, y, w0, w1))
print(RSS5(x, y, w0, w1))
print(RSS6(x, y, w0, w1))
print(RSS7(x, y, w0, w1))

