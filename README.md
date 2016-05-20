# KKS Binary Solidification with FiPy
This notebook contains code for binary solidification using the
Kim-Kim-Suzuki model [1] for interfacial energy. This allows easy specification
of gamma, but requires constant chemical potential through the interface.
The implementation involves iteratively solving for chemical composition in
pure phases such that the chemical potential constraint is satisfied [2].

Questions/comments to trevor.keller@nist.gov (Trevor Keller).

References:
1. Kim, Kim, and Suzuki. "Phase-field model for binary alloys."
    _Physical Review E_ 60:6;7186-7197 (1999). 
2. Provatas and Elder. _Phase-Field Methods in Materials Science and Engineering_,
    Chapter 6, Section 9. Wiley VCH: Weinheim, Germany. 2010.

## Model Description (paraphrased after Provatas & Elder)
We are setting out to simulate solidification of two-component alloy with a lenticular phase diagram, or "binary isomorphous" system, such as Cu-Ni. The free energy curves for pure phases, $f_S$ and $f_L$, are generated from a CALPHAD database, rather than a regular solution model.

In addition to this thermodynamic description, we are adopting the KKS treatment of diffuse interfaces. This simply means that at equilibrium, chemical potential is constant through the interface, and composition varies to make it so. More concretely, composition is defined by the phase fraction $\phi$ and two fictitious concentration fields, $C_S$ and $C_L$, representing composition of the pure phase, as

$$c = h(\phi)C_S + (1-h(\phi))C_L,$$

where the interpolation function $h(\phi)=\phi^3(6\phi^2-15\phi+10)$ takes the values in solid $h(\phi=1)=1$ and liquid $h(\phi=0)=0$. At equilibrium,

$$\mu = \left.\frac{\partial f_S}{\partial c}\right|_{c=C_S} = \left.\frac{\partial f_L}{\partial c}\right|_{c=C_L}.$$

Taken together, and introducing a double-well function $g(\phi)=\phi^2(1-\phi)^2$, the thermodynamic and interfacial treatments provide the bulk free energy,

$$f(\phi,c,T) = \omega g(\phi) + h(\phi)f_S(C_S,T) + (1-h(\phi))f_L(C_L,T).$$

Now, assuming nonconserved (Allen-Cahn) dynamics for $\phi$ and conserved (Cahn-Hilliard) dynamics for $c$, we can write the equations of motion

$$\frac{\partial\phi}{\partial t} = -M_\phi\frac{\delta\mathcal{F}}{\delta\phi} \rightarrow \tau\frac{\partial\phi}{\partial t} = \epsilon_\phi^2\nabla^2\phi -\omega g'(\phi) + h'(\phi)\left[f_L(C_L) - f_S(C_S) - \frac{\partial f_L(C_L)}{\partial c}(C_L - C_S)\right]$$

$$\frac{\partial c}{\partial t} = \nabla\cdot M_c\nabla\frac{\delta\mathcal{F}}{\delta c} = M_c\nabla\cdot Q(\phi)\left[h(\phi)\nabla C_S + (1-h(\phi))\nabla C_L\right]$$

with phase-dependent mobility $Q(\phi)=\frac{1-\phi}{(1+k) - (1-k)\phi}$, partition coefficient $k=\frac{C_S^e}{C_L^e}$, and time constant $\tau = M_\phi^{-1}$.

### The Wrinkle
$C_S$ and $C_L$ are not constants, they are field variables whose values depend on $\phi$ and $c$. Determining their values requires solving for the common tangent, or the coupled roots

$$f_1(C_S,C_L) = h(\phi)C_S + (1-h(\phi))C_L -c = 0$$

$$f_2(C_S,C_L) = \frac{\partial f_S(C_S)}{\partial c} - \frac{\partial f_L(C_L)}{\partial c} = 0.$$

While these equations can be solved using Newton's Method (cf. Provatas & Elder Appendix C.3), it's better to invoke a library, which we'll do a little later on. Even with a highly optimized root solver, determining $(C_S,C_L)$ at every grid point gets prohibitively expensive. The standard approach is to construct a lookup table for $C_S$ and $C_L$ covering $\phi=[-\delta,1+\delta]$ and $c=[-\delta,1+\delta]$ with a reasonably high number of points, then interpolating from the LUT at runtime. For best results, use the interpolated values as initial guesses for a touch-up iteration or two.
