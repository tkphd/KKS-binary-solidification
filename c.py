import matplotlib.pylab as plt
import numpy as np

y = np.loadtxt('c.log')
n = len(y)
x = np.linspace(0,n-1,n)

plt.plot(x, y, linewidth=5,color='k')
plt.xlabel('$t$',fontsize=28)
plt.ylabel('$c$',fontsize=28)

plt.savefig("c.png", dpi=600, bbox_inches='tight')

