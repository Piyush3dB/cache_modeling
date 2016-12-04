import matplotlib.pyplot as plt
import numpy as np

n = 1200

rdd = np.zeros(n)
hit = np.zeros(n)
evict = np.zeros(n)

f = open("rdd.dump")
for i,line in enumerate(f):
    if i < n:
        rdd[i] = line
f.close()

f = open("hit.out")
for i,line in enumerate(f):
    if i < n:
        hit[i] = line
f.close()

f = open("evict.out")
for i,line in enumerate(f):
    if i < n:
        evict[i] = line
f.close()

plt.subplot(3,1,1)
plt.plot(rdd)
plt.title('rdd')
plt.subplot(3,1,2)
plt.plot(np.cumsum(hit))
plt.title('hit')
plt.subplot(3,1,3)
plt.plot(np.cumsum(evict))
plt.title('evict')
plt.show()
