# -*- coding: utf-8 -*-
# Plot the bulk contributions (no gradients) for a selection of multiphase-field models.

import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Define regex to parse c from column headers
numeric_pattern = r"""
    [-+]? # optional sign
    (?:
        (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
        |
        (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
    )
    """
rx = re.compile(numeric_pattern, re.VERBOSE)

# Read phi as x, and c as y, and f as z, from disk
x = np.loadtxt('energy.csv',usecols=(0,),skiprows=1,delimiter=',')
y = 0
ncol = 0
with open("energy.csv") as f:
    names = f.readline().split(',')
    ncol = len(names)
    y = np.zeros(ncol-2,dtype=float)
    for i in range(1,len(names)-1):
        matches = rx.findall(names[i])
        y[i-1] = matches[0]
z = np.loadtxt('energy.csv',skiprows=1,usecols=range(1,ncol-1),delimiter=',')

# Configure plot
#plt.rcParams['figure.figsize'] = (8.5, 8.0)
plt.axis('equal')
plt.xlabel(r'$\phi$',fontsize=14)
plt.ylabel(r'$c$',fontsize=14)

# Plot contours
ncontours=128
plt.contourf(x,y,z.transpose(), ncontours, cmap=plt.cm.get_cmap('coolwarm'))

# Plot bounding box surrounding phi=[0,1] and c=[0,1]
sqx = np.array([0,1,1,0,0])
sqy = np.array([0,0,1,1,0])
plt.plot(sqx,sqy, linestyle=':', color='w')

# Add points from simulation runtime, if available
#for fname in glob.glob("data/planar/*.xy"):
fnames = glob.glob("data/planar/*.xy")
n = len(fnames)
for i in (1,n/2,n-1):
	p,c = np.loadtxt(fnames[i], delimiter=',', unpack=True)
	plt.plot(p,c,'.-',markersize=1.5,linewidth=1)

# Save image and close
plt.savefig('energy.png', dpi=600, bbox_inches='tight')
plt.close()
