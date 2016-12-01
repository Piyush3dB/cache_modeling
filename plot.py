import matplotlib.pyplot as plt
import numpy as np

n = 12000

hit = np.zeros(n)
evict = np.zeros(n)

f = open("hit.log")
for i,line in enumerate(f):
    if i < n:
        hit[i] = line
f.close()

f = open("evict.log")
for i,line in enumerate(f):
    if i < n:
        evict[i] = line
f.close()

plt.subplot(2,1,1)
plt.plot(np.cumsum(hit))
plt.title('hit')
plt.subplot(2,1,2)
plt.plot(np.cumsum(evict))
plt.title('evict')
plt.show()
