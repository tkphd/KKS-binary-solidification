import matplotlib.pylab as plt
import numpy as np

c,f,Co,u = np.loadtxt('c.log',delimiter='\t',unpack=True)
n = len(f)
x = np.linspace(0,n-1,n)

plt.plot(x, c, linewidth=5,color='k')
plt.xlabel('$t$',fontsize=28)
plt.ylabel('$c$',fontsize=28)
plt.savefig("c.png", dpi=600, bbox_inches='tight')
plt.close()

f0 = 1.01*np.min(f)
plt.semilogy(x, f-f0, linewidth=5,color='k')
plt.xlabel('$t$',fontsize=28)
plt.ylabel('$\Delta\mathcal{F}$',fontsize=28)
plt.savefig("f.png", dpi=600, bbox_inches='tight')
plt.close()

C0 = 1.01*np.min(Co)
plt.semilogy(x, Co-C0, linewidth=5,color='k')
plt.xlabel('$t$',fontsize=28)
plt.ylabel('Co',fontsize=28)
plt.savefig("Co.png", dpi=600, bbox_inches='tight')
plt.close()

plt.plot(x, u, linewidth=5,color='k')
plt.xlabel('$t$',fontsize=28)
plt.ylabel('$\Delta\mu$',fontsize=28)
plt.savefig("u.png", dpi=600, bbox_inches='tight')
plt.close()
