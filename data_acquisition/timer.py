import numpy as np
import time

timeLS = []
t0 = time.time()
s = np.array([])
s = np.append(s, [1]*100000000)
timeLS += [time.time() - t0]
print("0", timeLS[-1])
#print(s)

t0 = time.time()
s = []
s += [1]*100000000
timeLS += [time.time() - t0]
print("1", timeLS[-1])
#print(s)


#t0 = time.time()
#s = []
#s.append([1]*10)
#timeLS += [time.time() - t0]
#print("2", timeLS[-1])
#print(s)

t0 = time.time()
s = []
s.extend([1]*100000000)
timeLS += [time.time() - t0]
print("2", timeLS[-1])
#print(s)

print("MIN", np.argmin(timeLS))

