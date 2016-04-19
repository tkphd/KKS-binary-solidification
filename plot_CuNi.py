
# coding: utf-8

# # Calculating Isobaric Binary Phase Diagrams

# These are a few examples of how to use CALPHAD TDB files to calculate isobaric binary phase diagrams.
# 
# Note that, at the time of writing, invariant reactions (three-phase 'regions' on binary diagrams) are not yet automatically detected so they
# are not drawn on the diagram.

import matplotlib.pyplot as plt
from pycalphad import Database, binplot, calculate
import pycalphad.variables as v
from pycalphad.plot.utils import phase_legend
import numpy as np
from scipy.optimize import curve_fit, newton

fig = plt.figure(figsize=(9,6))

db_cuni = Database('CuNi_RWTH.tdb')
my_phases_cuni = ['LIQUID', 'FCC_A1', 'BCC_A2', 'HCP_A3']
binplot(db_cuni, ['CU', 'NI', 'VA'] , my_phases_cuni, {v.X('NI'):(0,1,0.01),  v.T: (1300, 1800, 5), v.P:101325},  ax=fig.gca())

plt.savefig('CuNi.png', dpi=400, bbox_inches='tight')
plt.close()


def func(x, a, b, c, d, e, f, g, h, i, j, k):
	return a*x**10 + b*x**9 + c*x**8 + d*x**7 + e*x**6 + f*x**5 + g*x**4 + h*x**3 + i*x**2 + j*x + k
def fprime(x, a, b, c, d, e, f, g, h, i, j):
	return 10*a*x**9 + 9*b*x**8 + 8*c*x**7 + 7*d*x**6 + 6*e*x**5 + 5*f*x**4 + 4*g*x**3 + 3*h*x**2 + 2*i*x + j
xref = np.linspace(-0.1,1.1,150)
# Cu-Ni (S. an Mey 1999)

db_cuni = Database('CuNi_RWTH.tdb')
my_phases_cuni = ['LIQUID', 'FCC_A1', 'BCC_A2', 'HCP_A3']

# # Calculating Energy Surfaces of Binary Systems

# It is very common in CALPHAD modeling to directly examine the Gibbs energy surface of all the constituent phases in a system.
# 
# Below we show how the Gibbs energy of all phases may be calculated as a function of composition at a given temperature (1550 K).
# 

legend_handles, colorlist = phase_legend(my_phases_cuni)
fig = plt.figure(figsize=(9,6))
ax = fig.gca()
for name in my_phases_cuni:
    result = calculate(db_cuni, ['CU', 'NI', 'VA'], name, T=1550, output='GM')
    x = np.ravel(result.X.sel(component='NI'))
    y = np.ravel(result.GM)
    ax.scatter(x, y, marker='.', s=5, color=colorlist[name.upper()])
    popt, pcov = curve_fit(func, x, y)
    print name, popt
    ax.plot(xref, func(xref,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],popt[8],popt[9],popt[10]), '-', color=colorlist[name.upper()])
    roo = newton(fprime, 0.5, args=(popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],popt[8],popt[9]),maxiter=1000)
    print name, "root~", roo
ax.set_xlim((-0.1, 1.1))
ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.6))
plt.savefig('CuNi_energy.png',dpi=400,bbox_inches='tight')
plt.close()
