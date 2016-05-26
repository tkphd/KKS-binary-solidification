// KKS.cpp
// Algorithms for 2D and 3D isotropic binary alloy solidification
// Questions/comments to trevor.keller@nist.gov (Trevor Keller)

#ifndef KKS_UPDATE
#define KKS_UPDATE
#include<cmath>
#include"MMSP.hpp"
#include"KKS.hpp"

// Note: KKS.hpp contains important declarations and comments. Have a look.

// To use a synthetic phase diagram that works poorly, comment out the next line.
#define CALPHAD

#ifdef CALPHAD
// 10-th order polynomial fitting coefficients from PyCALPHAD
double calCs[11] = { 6.19383857e+03,-3.09926825e+04, 6.69261368e+04,-8.16668934e+04,
                     6.19902973e+04,-3.04134700e+04, 9.74968659e+03,-2.04529002e+03,
                     2.95622845e+02,-3.70962613e+01,-6.12900561e+01};
double calCl[11] = { 6.18692878e+03,-3.09579439e+04, 6.68516329e+04,-8.15779791e+04,
                     6.19257214e+04,-3.03841489e+04, 9.74145735e+03,-2.04379606e+03,
                     2.94796431e+02,-3.39127135e+01,-6.26373908e+01};
const double  Cse = 0.48300,  Cle = 0.33886;    // equilibrium concentration
#else
// Parabolic model parameters
const double  As = 150.0, Al = 150.0;   // 2*curvature of parabola
const double dCs = 10.0, dCl  = 10.0;   // y-axis offset
const double  Cse = 0.3,  Cle = 0.7;    // equilibrium concentration
#endif

// Numerical stability (Courant-Friedrich-Lewy) parameters
const double CFL = 1.0/300.0; // controls timestep

// Kinetic and model parameters
const double meshres = 0.075; // dx=dy
const double eps_sq = 1.25;
const double a_int = 2.5; // alpha, prefactor of interface width
const double halfwidth = 2.25*meshres; // half the interface width
const double omega = 2.0*eps_sq*pow(a_int/halfwidth,2.0);
const double dt = 2.0*CFL*pow(meshres,2.0)/eps_sq;
const double ps0 = 1.0, pl0 = 0.0; // initial phase fractions
const double cBs = (Cse+Cle)/2.0 /*+ 0.01*/;  // initial solid concentration
const double cBl = (Cse+Cle)/2.0 /*- 0.001*/;  // initial liquid concentration

// Resolution of the constant chem. pot. composition lookup table
const int LUTnc = 125; // number of points along c-axis
const int LUTnp = 125; // number of points along p-axis
const double dp = 1.0/LUTnp;
const double dc = 1.0/LUTnc;

// Newton-Raphson root finding parameters
const unsigned int refloop = 1e7;// ceiling to kill infinite loops in iterative scheme: reference table threshold
const unsigned int fasloop = 5e6;// ceiling to kill infinite loops in iterative scheme: fast update() threshold
const double reftol = 1.0e-8;    // tolerance for iterative scheme to satisfy equal chemical potential: reference table threshold
const double fastol = 1.0e-7;    // tolerance for iterative scheme to satisfy equal chemical potential: fast update() threshold
const double epsilon = 1.0e-10;  // what to consider zero to avoid log(c) explosions

namespace MMSP{

void generate(int dim, const char* filename)
{
	int rank=0;
	#ifdef MPI_VERSION
	rank=MPI::COMM_WORLD.Get_rank();
	#endif
	srand(time(NULL)+rank);

	/* ======================================================================== *
	 * Construct look-up table for fast enforcement of equal chemical potential *
	 * ======================================================================== */

	// Consider generating a free energy plot and lookup table.
	bool nrg_not_found=true; // set False to disable energy plot, which may save a few minutes of work
	bool lut_not_found=true; // LUT must exist -- do not disable!
	if (rank==0) {
		if (1) {
			std::ifstream fnrg("energy.csv");
			if (fnrg) {
				nrg_not_found=false;
				fnrg.close();
			}
			std::ifstream flut("consistentC.lut");
			if (flut) {
				lut_not_found=false;
				flut.close();
			}
		}
	}
	#ifdef MPI_VERSION
	MPI::COMM_WORLD.Bcast(&nrg_not_found,1,MPI_BOOL,0);
	MPI::COMM_WORLD.Bcast(&lut_not_found,1,MPI_BOOL,0);
	MPI::COMM_WORLD.Barrier();
	#endif

	if (nrg_not_found) {
		// Print out the free energy for phi in [-0.75,1.75] with slices of c in [-0.75,1.75] for inspection.
		// This is time consuming!
		if (rank==0)
			std::cout<<"Sampling free energy landscape, exporting to energy.csv."<<std::endl;

		bool silent=true;
		#ifdef MPI_VERSION
		MPI::COMM_WORLD.Barrier();
		#endif
		if (rank==0) export_energy(silent);
		#ifdef MPI_VERSION
		MPI::COMM_WORLD.Barrier();
		#endif
	}

	if (lut_not_found) {
		/* Generate Cs,Cl look-up table (LUT) using Newton-Raphson method, outlined in Provatas' Appendix C3
		 * Store results in pureconc, which contains two fields:
		 * 0. Cs, fictitious composition of pure liquid
		 * 1. Cl, fictitious composition of pure solid
		 *
		 * The grid is discretized over phi (axis 0) and c (axis 1).
		*/
		if (rank==0)
			std::cout<<"Writing look-up table of Cs, Cl to consistentC.lut. Please be patient..."<<std::endl;
		bool silent=true, randomize=true;
		LUTGRID pureconc(3,-1,2+LUTnp,-1,2+LUTnc);
		dx(pureconc,0) = dp; // different resolution in phi
		dx(pureconc,1) = dc; // and c is not unreasonable

		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(pureconc); n++) {
			simple_progress(n,nodes(pureconc));
			vector<int> x = position(pureconc,n);
			pureconc(n)[0] = 1.0 - dc*x[1]; // guess Cs
			pureconc(n)[1] = 1.0 - dc*x[1]; // guess Cl
			pureconc(n)[2] = iterateConc(reftol, refloop, randomize, dp*x[0], dc*x[1], pureconc(n)[0], pureconc(n)[1], silent);
		}

		output(pureconc,"consistentC.lut");
	}

	// Read concentration look-up table from disk, in its entirety, even in parallel. Should be relatively small.
	#ifndef MPI_VERSION
	const int ghost=0;
	LUTGRID pureconc("consistentC.lut",ghost);
	#else
	MPI::COMM_WORLD.Barrier();
	LUTGRID pureconc(3,-1,2+LUTnp,-1,2+LUTnc);
	const bool serial=true; // Please do not change this :-)
	const int ghost=1;
	pureconc.input("consistentC.lut",ghost,serial);
	#endif

	/* ====================================================================== *
	 * Generate initial conditions using phase diagram and freshly minted LUT *
	 * ====================================================================== */

	/* Grid contains four fields:
	   0. phi, phase fraction solid. Phi=1 means Solid.
	   1. c, concentration of component A
	   2. Cs, fictitious composition of solid
	   3. Cl, fictitious composition of liquid
	   4. Residual associated with Cs,Cl computation
	 */
	if (dim==1) {
		int L=512;
		GRID1D initGrid(5,0,L);
		for (int d=0; d<dim; d++) {
			dx(initGrid,d) = meshres;
			if (x0(initGrid,d)==g0(initGrid,d))
				b0(initGrid,d) = Neumann;
			else if (x1(initGrid,d)==g1(initGrid,d))
				b1(initGrid,d) = Neumann;
		}

		double ctot = 0.0, ftot = 0.0;
		double radius=(g1(initGrid,0)-g0(initGrid,0))/4;
		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid,n);
			double r = std::abs(x[0] - (g1(initGrid,0)-g0(initGrid,0))/2);
			if (r < radius) { // Solid
				initGrid(n)[0] = ps0;
				initGrid(n)[1] = cBs;
			} else {
				initGrid(n)[0] = pl0;
				initGrid(n)[1] = cBl;
			}
			initGrid(n)[4] = interpolateConc(pureconc, initGrid(n)[0], initGrid(n)[1], initGrid(n)[2], initGrid(n)[3]);
			//initGrid(n)[4] = iterateConc(reftol, refloop, 0, initGrid(n)[0], initGrid(n)[1], initGrid(n)[2], initGrid(n)[3], 1);
			ctot += initGrid(n)[1]*dx(initGrid);
			ftot += f(initGrid(n)[0], initGrid(n)[1], initGrid(n)[2], initGrid(n)[3])*dx(initGrid);
		}
		// Add gradient to ftot
		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> s = position(initGrid,n);
			double gradPsq = 0.0;
			for (int d=0; d<1; d++) {
				// Get low value
				s[d]--;
				double pl = initGrid(s)[0];
				// Get high value
				s[d]+=2;
				double ph = initGrid(s)[0];
				// Back to the middle
				s[d]--;

				// Put 'em together
				double weight = 1.0/(2.0*dx(initGrid,d));
				gradPsq += pow(weight*(ph - pl),2.0);
			}
			ftot += gradPsq*dx(initGrid);
		}

		print_values(initGrid, rank);

		output(initGrid,filename);

		#ifdef MPI_VERSION
		double myct(ctot);
		double myft(ftot);
		MPI::COMM_WORLD.Allreduce(&myct, &ctot, 1, MPI_DOUBLE, MPI_SUM);
		MPI::COMM_WORLD.Allreduce(&myft, &ftot, 1, MPI_DOUBLE, MPI_SUM);
		#endif
		if (rank==0) {
			std::ofstream cfile("c.log");
			cfile<<ctot<<'\t'<<ftot<<std::endl;
			cfile.close();
		}

	} else if (dim==2) {
		int L = 64;
		GRID2D initGrid(5,0,L,0,L/2);
		for (int d=0; d<dim; d++) {
			dx(initGrid,d) = meshres;
			if (x0(initGrid,d)==g0(initGrid,d))
				b0(initGrid,d) = Neumann;
			else if (x1(initGrid,d)==g1(initGrid,d))
				b1(initGrid,d) = Neumann;
		}

		double ctot = 0.0, ftot = 0.0;
		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid,n);
			/**/
			double ra = 15.0, rb = 8.0, rc = 8.0;
			// Circular interfaces
			if ( (pow(x[0] - (ra+1   ),2) + pow(x[1] - (L/2-ra-1  ),2) < ra*ra) ||
			     (pow(x[0] - 0.625*(L),2) + pow(x[1] - (L/2-rb-1  ),2) < rb*rb) ||
			     (pow(x[0] - (L-rc-1 ),2) + pow(x[1] - (rc+1),2) < rc*rc)
			) {
				// Solid
				initGrid(n)[0] = ps0;
				initGrid(n)[1] = cBs;
			} else {
				// Liquid
				initGrid(n)[0] = pl0;
				initGrid(n)[1] = cBl;
			}
			/*
			// Planar interface
			if (x[0] < L/2) {
				// Solid
				initGrid(n)[0] = ps0;
				initGrid(n)[1] = cBs;
			} else {
				// Liquid
				initGrid(n)[0] = pl0;
				initGrid(n)[1] = cBl;
			}
			*/
			initGrid(n)[4] = interpolateConc(pureconc, initGrid(n)[0], initGrid(n)[1], initGrid(n)[2], initGrid(n)[3]);
			//initGrid(n)[4] = iterateConc(reftol, refloop, 0, initGrid(n)[0], initGrid(n)[1], initGrid(n)[2], initGrid(n)[3], 1);
			ctot += initGrid(n)[1]*dx(initGrid)*dy(initGrid);
			ftot += f(initGrid(n)[0], initGrid(n)[1], initGrid(n)[2], initGrid(n)[3])*dx(initGrid)*dy(initGrid);
		}
		// Add gradient to ftot
		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> s = position(initGrid,n);
			double gradPsq = 0.0;
			for (int d=0; d<2; d++) {
				// Get low value
				s[d]--;
				double pl = initGrid(s)[0];
				// Get high value
				s[d]+=2;
				double ph = initGrid(s)[0];
				// Back to the middle
				s[d]--;

				// Put 'em together
				double weight = 1.0/(2.0*dx(initGrid,d));
				gradPsq += pow(weight*(ph - pl),2.0);
			}
			ftot += gradPsq*dx(initGrid);
		}

		print_values(initGrid, rank);

		output(initGrid,filename);

		#ifdef MPI_VERSION
		double myct(ctot);
		double myft(ftot);
		MPI::COMM_WORLD.Allreduce(&myct, &ctot, 1, MPI_DOUBLE, MPI_SUM);
		MPI::COMM_WORLD.Allreduce(&myft, &ftot, 1, MPI_DOUBLE, MPI_SUM);
		#endif
		if (rank==0) {
			std::ofstream cfile("c.log");
			cfile<<ctot<<'\t'<<ftot<<std::endl;
			cfile.close();
		}
	} else if (dim==3) {
		int L=64;
		double radius=10.0;
		GRID3D initGrid(5,0,L,0,L,0,L);
		for (int d=0; d<dim; d++) {
			dx(initGrid,d) = meshres;
			if (x0(initGrid,d)==g0(initGrid,d))
				b0(initGrid,d) = Neumann;
			else if (x1(initGrid,d)==g1(initGrid,d))
				b1(initGrid,d) = Neumann;
		}

		double ctot = 0.0, ftot = 0.0;
		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid,n);
			double r = sqrt(pow(radius-x[0]%64,2)+pow(radius-x[1]%64,2));
			if (r<radius) { // Solid
				initGrid(n)[0] = ps0;
				initGrid(n)[1] = cBs;
			} else {
				initGrid(n)[0] = pl0;
				initGrid(n)[1] = cBl;
			}
			initGrid(n)[4] = interpolateConc(pureconc, initGrid(n)[0], initGrid(n)[1], initGrid(n)[2], initGrid(n)[3]);
			//initGrid(n)[4] = iterateConc(reftol, refloop, 0, initGrid(n)[0], initGrid(n)[1], initGrid(n)[2], initGrid(n)[3], 1);
			ctot += initGrid(n)[1]*dx(initGrid)*dy(initGrid)*dz(initGrid);
			ftot += f(initGrid(n)[0], initGrid(n)[1], initGrid(n)[2], initGrid(n)[3])*dx(initGrid)*dy(initGrid)*dz(initGrid);
		}
		// Add gradient to ftot
		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> s = position(initGrid,n);
			double gradPsq = 0.0;
			for (int d=0; d<3; d++) {
				// Get low value
				s[d]--;
				double pl = initGrid(s)[0];
				// Get high value
				s[d]+=2;
				double ph = initGrid(s)[0];
				// Back to the middle
				s[d]--;

				// Put 'em together
				double weight = 1.0/(2.0*dx(initGrid,d));
				gradPsq += pow(weight*(ph - pl),2.0);
			}
			ftot += gradPsq*dx(initGrid);
		}

		print_values(initGrid, rank);

		output(initGrid,filename);

		#ifdef MPI_VERSION
		double myct(ctot);
		double myft(ftot);
		MPI::COMM_WORLD.Allreduce(&myct, &ctot, 1, MPI_DOUBLE, MPI_SUM);
		MPI::COMM_WORLD.Allreduce(&myft, &ftot, 1, MPI_DOUBLE, MPI_SUM);
		#endif
		if (rank==0) {
			std::ofstream cfile("c.log");
			cfile<<ctot<<'\t'<<ftot<<std::endl;
			cfile.close();
		}
	} else {
		std::cerr<<"ERROR: "<<dim<<"-dimensional domains not supported."<<std::endl;
		exit(-1);
	}

	if (rank==0)
		printf("Equilibrium Cs=%.2f, Cl=%.2f\n", Cs_e(), Cl_e());

}

template <int dim, typename T> void update(grid<dim,vector<T> >& oldGrid, int steps)
{
	int rank=0;
    #ifdef MPI_VERSION
    rank = MPI::COMM_WORLD.Get_rank();
    #endif

	// Read concentration look-up table from disk, in its entirety, even in parallel. Should be relatively small.
	#ifndef MPI_VERSION
	const int ghost=0;
	LUTGRID pureconc("consistentC.lut",ghost);
	#else
	LUTGRID pureconc(3,0,1+LUTnp,0,1+LUTnc);
	const bool serial=true; // Please do not change this :-)
	const int ghost=1;
	pureconc.input("consistentC.lut",ghost,serial);
	#endif

	ghostswap(oldGrid);
   	grid<dim,vector<T> > newGrid(oldGrid);
	double dV=1.0;
	for (int d=0; d<dim; d++) {
		dx(oldGrid,d) = meshres;
		dx(newGrid,d) = meshres;
		dV *= dx(oldGrid,d);
		if (x0(oldGrid,d) == g0(oldGrid,d)) {
			b0(oldGrid,d) = Neumann;
			b0(newGrid,d) = Neumann;
		} else if (x1(oldGrid,d) == g1(oldGrid,d)) {
			b1(oldGrid,d) = Neumann;
			b1(newGrid,d) = Neumann;
		}
	}

	std::ofstream cfile;
	if (rank==0)
		cfile.open("c.log",std::ofstream::out | std::ofstream::app);

	for (int step=0; step<steps; step++) {
		if (rank==0)
			print_progress(step, steps);

		double ctot=0.0, ftot=0.0;
		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(oldGrid); n++) {
			vector<int> x = position(oldGrid,n);

			// Cache some frequently-used reference values
			const T phi_old = oldGrid(n)[0];
			const T c_old   = oldGrid(n)[1];
			const T Cs_old  = oldGrid(n)[2];
			const T Cl_old  = oldGrid(n)[3];

			// Compute divergence of c, phi using half-steps in space for second-order accuracy,
			// laplacian of phi (alone, since built-in Laplacian returns all fields),
			// and grad(phi)^2 for free energy computation

			double divGradP = 0.0;
			double divGradC = 0.0;
			double lapPhi = 0.0;
			double gradPsq = 0.0;
			vector<int> s(x);
			for (int d=0; d<dim; d++) {
				// Second-order differencing requires consistent schemes at the boundaries.
				// Implemented after Strikwerda 2004 (p. 152) and KTH CFD coursenotes,
				//     http://www.mech.kth.se/~ardeshir/courses/literature/fd.pdf
				// with guessed handling of variable coefficients
				double weight = 1.0/pow(dx(oldGrid,d), 2.0);

				if (x[d]==x0(oldGrid,d) && b0(oldGrid,d) == Neumann) {
					// Based on the "Right-sided second order difference"
					// Get central values
					const T& p0 = oldGrid(s)[0];
					const T& c0 = oldGrid(s)[1];
					const T& S0 = oldGrid(s)[2];
					const T& L0 = oldGrid(s)[3];
					const T Mp0 = Q(p0,S0,L0)*hprime(p0)*(L0-S0);
					const T Mc0 = Q(p0,S0,L0);
					// Get middle values
					s[d] += 1;
					const T& p1 = oldGrid(s)[0];
					const T& c1 = oldGrid(s)[1];
					const T& S1 = oldGrid(s)[2];
					const T& L1 = oldGrid(s)[3];
					const T Mp1 = Q(p1,S1,L1)*hprime(p1)*(L1-S1);
					const T Mc1 = Q(p1,S1,L1);
					// Get high values
					s[d] += 1;
					const T& p2 = oldGrid(s)[0];
					const T& c2 = oldGrid(s)[1];
					const T& S2 = oldGrid(s)[2];
					const T& L2 = oldGrid(s)[3];
					const T Mp2 = Q(p2,S2,L2)*hprime(p2)*(L2-S2);
					const T Mc2 = Q(p2,S2,L2);
					// Get very high value
					s[d] += 1;
					const T& p3 = oldGrid(s)[0];

					// Re-center and put 'em all together
					s[d] -= 3;

					divGradP += 0.25*weight*( 3.0*(Mp1+Mp0)*(p1-p0) - (Mp2+Mp1)*(p2-p1) );
					divGradC += 0.25*weight*( 3.0*(Mc1+Mc0)*(c1-c0) - (Mc2+Mc1)*(c2-c1) );
					lapPhi   += weight*(2.0*p0 - 5.0*p1 + 4.0*p2 - p3);
				} else if (x[d]==x1(oldGrid,d)-1 && b1(oldGrid,d)==Neumann) {
					// Based on the "Left-sided second order difference"
					// Get central values
					const T& p3 = oldGrid(s)[0];
					const T& c3 = oldGrid(s)[1];
					const T& S3 = oldGrid(s)[2];
					const T& L3 = oldGrid(s)[3];
					const T Mp3 = Q(p3,S3,L3)*hprime(p3)*(L3-S3);
					const T Mc3 = Q(p3,S3,L3);
					// Get middle values
					s[d] -= 1;
					const T& p2 = oldGrid(s)[0];
					const T& c2 = oldGrid(s)[1];
					const T& S2 = oldGrid(s)[2];
					const T& L2 = oldGrid(s)[3];
					const T Mp2 = Q(p2,S2,L2)*hprime(p2)*(L2-S2);
					const T Mc2 = Q(p2,S2,L2);
					// Get low values
					s[d] -= 1;
					const T& p1 = oldGrid(s)[0];
					const T& c1 = oldGrid(s)[1];
					const T& S1 = oldGrid(s)[2];
					const T& L1 = oldGrid(s)[3];
					const T Mp1 = Q(p1,S1,L1)*hprime(p1)*(L1-S1);
					const T Mc1 = Q(p1,S1,L1);
					// Get very low value
					s[d] -= 1;
					const T& p0 = oldGrid(s)[0];

					// Re-center and put 'em all together
					s[d] += 3;

					divGradP += 0.25*weight*( 3.0*(Mp3+Mp2)*(p3-p2) - (Mp2+Mp1)*(p2-p1) );
					divGradC += 0.25*weight*( 3.0*(Mc3+Mc2)*(c3-c2) - (Mc2+Mc1)*(c2-c1) );
					lapPhi   += weight*(2.0*p3 - 5.0*p2 + 4.0*p1 - p0);
				} else {
					// Central second-order difference
					// Get low values
					s[d] -= 1;
					const T& pl = oldGrid(s)[0];
					const T& cl = oldGrid(s)[1];
					const T& Sl = oldGrid(s)[2];
					const T& Ll = oldGrid(s)[3];
					const T Mpl = Q(pl,Sl,Ll)*hprime(pl)*(Ll-Sl);
					const T Mcl = Q(pl,Sl,Ll);
					// Get high values
					s[d] += 2;
					const T& ph = oldGrid(s)[0];
					const T& ch = oldGrid(s)[1];
					const T& Sh = oldGrid(s)[2];
					const T& Lh = oldGrid(s)[3];
					const T Mph = Q(ph,Sh,Lh)*hprime(ph)*(Lh-Sh);
					const T Mch = Q(ph,Sh,Lh);
					// Get central values
					s[d] -= 1;
					const T& pc = oldGrid(s)[0];
					const T& cc = oldGrid(s)[1];
					const T& Sc = oldGrid(s)[2];
					const T& Lc = oldGrid(s)[3];
					const T Mpc = Q(pc,Sc,Lc)*hprime(pc)*(Lc-Sc);
					const T Mcc = Q(pc,Sc,Lc);

					// Put 'em all together
					divGradP += 0.5*weight*( (Mph+Mpc)*(ph-pc) - (Mpc+Mpl)*(pc-pl) );
					divGradC += 0.5*weight*( (Mch+Mcc)*(ch-cc) - (Mcc+Mcl)*(cc-cl) );
					lapPhi   += weight*(ph - 2.*pc + pl);
					gradPsq  += pow(0.5*(ph - pl)/dx(oldGrid,d), 2.0);
				}
			}

			/* ==================================== *
			 * Solve the Equation of Motion for phi *
			 * ==================================== */

			// Provatas & Elder: Eqn. 6.97
			newGrid(n)[0] = phi_old + dt*( eps_sq*lapPhi - omega*gprime(phi_old)
			                               + hprime(phi_old)*( fl(Cl_old)-fs(Cs_old)-(Cl_old-Cs_old)*dfl_dc(Cl_old) ));


			/* ================================== *
			 * Solve the Equation of Motion for c *
			 * ================================== */

			// Kim, Kim, & Suzuki: Eqn. 33
			newGrid(n)[1] = c_old + dt*(divGradC + divGradP);


			/* ============================== *
			 * Determine consistent Cs and Cl *
			 * ============================== */


			bool silent=true, randomize=false;
			newGrid(n)[2] = Cs_old;
			newGrid(n)[3] = Cl_old;
			newGrid(n)[4] = interpolateConc(pureconc, newGrid(n)[0], newGrid(n)[1], newGrid(n)[2], newGrid(n)[3]);
			//newGrid(n)[4] = iterateConc(reftol, refloop, randomize, newGrid(n)[0], newGrid(n)[1], newGrid(n)[2], newGrid(n)[3], silent);

			// Update total mass and energy, using critical block containing as little arithmetic as possible, in OpenMP- and MPI-compatible manner
			double myc = dV*newGrid(n)[1];
			double myf = dV*(0.5*eps_sq*gradPsq + f(newGrid(n)[0], newGrid(n)[1], newGrid(n)[2], newGrid(n)[3]));
			#ifndef MPI_VERSION
			#pragma omp critical
			{
			#endif
			ctot += myc;
			ftot += myf;
			#ifndef MPI_VERSION
			}
			#endif

			// ~ fin ~
		}
		swap(oldGrid,newGrid);
		ghostswap(oldGrid);

		// Compute total mass
		#ifdef MPI_VERSION
		double myct(ctot);
		double myft(ftot);
		MPI::COMM_WORLD.Allreduce(&myct, &ctot, 1, MPI_DOUBLE, MPI_SUM);
		MPI::COMM_WORLD.Allreduce(&myft, &ftot, 1, MPI_DOUBLE, MPI_SUM);
		#endif
		if (rank==0)
			cfile<<ctot<<'\t'<<ftot<<std::endl;
	}
	if (rank==0)
		cfile.close();

	print_values(oldGrid, rank);
}


} // namespace MMSP

template<int dim, typename T>
void print_values(const MMSP::grid<dim,MMSP::vector<T> >& oldGrid, const int rank) {
	double pTot=0.0;
	double cTot=0.0;
	unsigned int nTot = nodes(oldGrid);
	for (int n=0; n<nodes(oldGrid); n++) {
		pTot += oldGrid(n)[0];
		cTot += oldGrid(n)[1];
	}

	#ifdef MPI_VERSION
	double myP(pTot), myC(cTot);
	unsigned int myN(nTot);
	MPI::COMM_WORLD.Allreduce(&myP, &pTot, 1, MPI_DOUBLE, MPI_SUM);
	MPI::COMM_WORLD.Allreduce(&myC, &cTot, 1, MPI_DOUBLE, MPI_SUM);
	MPI::COMM_WORLD.Allreduce(&myN, &nTot, 1, MPI_UNSIGNED, MPI_SUM);
	#endif
	cTot /= nTot;
	double wps = (100.0*pTot)/nTot;
	double wpl = (100.0*(nTot-pTot))/nTot;
	double fs = 100.0*(cTot - Cl_e())/(Cs_e()-Cl_e());
	double fl = 100.0*(Cs_e() - cTot)/(Cs_e()-Cl_e());
	if (rank==0)
		printf("System has %.2f%% solid, %.2f%% liquid, and composition %.2f%% B. Equilibrium is %.2f%% solid, %.2f%% liquid.\n",
		       wps, wpl, 100.0*cTot, fs, fl);
}

double fl(const double c)
{
	#ifdef CALPHAD
	// 10-th order polynomial fit to S. an Mey Cu-Ni CALPHAD database
	return  calCl[0]*pow(c,10)
	       +calCl[1]*pow(c,9)
	       +calCl[2]*pow(c,8)
	       +calCl[3]*pow(c,7)
	       +calCl[4]*pow(c,6)
	       +calCl[5]*pow(c,5)
	       +calCl[6]*pow(c,4)
	       +calCl[7]*pow(c,3)
	       +calCl[8]*pow(c,2)
	       +calCl[9]*c
	       +calCl[10];
	#else
	return Al*pow(c-Cle,2.0)+dCl;
	#endif
}

double fs(const double c)
{
	#ifdef CALPHAD
	// 10-th order polynomial fit to S. an Mey Cu-Ni CALPHAD database
	return  calCs[0]*pow(c,10)
	       +calCs[1]*pow(c,9)
	       +calCs[2]*pow(c,8)
	       +calCs[3]*pow(c,7)
	       +calCs[4]*pow(c,6)
	       +calCs[5]*pow(c,5)
	       +calCs[6]*pow(c,4)
	       +calCs[7]*pow(c,3)
	       +calCs[8]*pow(c,2)
	       +calCs[9]*c
	       +calCs[10];
	#else
	return As*pow(c-Cse,2.0)+dCs;
	#endif
}


double dfl_dc(const double c)
{
	#ifdef CALPHAD
	return  10.0*calCl[0]*pow(c,9)
	       +9.0*calCl[1]*pow(c,8)
	       +8.0*calCl[2]*pow(c,7)
	       +7.0*calCl[3]*pow(c,6)
	       +6.0*calCl[4]*pow(c,5)
	       +5.0*calCl[5]*pow(c,4)
	       +4.0*calCl[6]*pow(c,3)
	       +3.0*calCl[7]*pow(c,2)
	       +2.0*calCl[8]*c
	       +calCl[9];
	#else
	return 2.0*Al*(c-Cle);
	#endif
}

double dfs_dc(const double c)
{
	#ifdef CALPHAD
	return  10.0*calCs[0]*pow(c,9)
	       +9.0*calCs[1]*pow(c,8)
	       +8.0*calCs[2]*pow(c,7)
	       +7.0*calCs[3]*pow(c,6)
	       +6.0*calCs[4]*pow(c,5)
	       +5.0*calCs[5]*pow(c,4)
	       +4.0*calCs[6]*pow(c,3)
	       +3.0*calCs[7]*pow(c,2)
	       +2.0*calCs[8]*c
	       +calCs[9];
	#else
	return 2.0*As*(c-Cse);
	#endif
}

double d2fl_dc2(const double c)
{
	#ifdef CALPHAD
	return  90.0*calCl[0]*pow(c,8)
	       +72.0*calCl[1]*pow(c,7)
	       +56.0*calCl[2]*pow(c,6)
	       +42.0*calCl[3]*pow(c,5)
	       +30.0*calCl[4]*pow(c,4)
	       +20.0*calCl[5]*pow(c,3)
	       +12.0*calCl[6]*pow(c,2)
	       +6.0*calCl[7]*c
	       +2.0*calCl[8];
	#else
	return 2.0*Al;
	#endif
}

double d2fs_dc2(const double c)
{
	#ifdef CALPHAD
	return  90.0*calCs[0]*pow(c,8)
	       +72.0*calCs[1]*pow(c,7)
	       +56.0*calCs[2]*pow(c,6)
	       +42.0*calCs[3]*pow(c,5)
	       +30.0*calCs[4]*pow(c,4)
	       +20.0*calCs[5]*pow(c,3)
	       +12.0*calCs[6]*pow(c,2)
	       +6.0*calCs[7]*c
	       +2.0*calCs[8];
	#else
	return 2.0*As;
	#endif
}

double R(const double p, const double Cs, const double Cl)
{
	// denominator for dCs, dCl, df
	return h(p)*d2fl_dc2(Cl) + (1.0-h(p))*d2fs_dc2(Cs);
}

double dCl_dc(const double p, const double Cs, const double Cl)
{
	double invR = R(p, Cs, Cl);
	if (fabs(invR)>epsilon) invR = 1.0/invR;
	return d2fl_dc2(Cl)*invR;
}

double dCs_dc(const double p, const double Cs, const double Cl)
{
	double invR = R(p, Cs, Cl);
	if (fabs(invR)>epsilon) invR = 1.0/invR;
	return d2fs_dc2(Cs)*invR;
}

double Cl_e() {return Cle;}

double Cs_e() {return Cse;}

double f(const double p, const double c, const double Cs, const double Cl)
{
	//const double w = 1.0; // well barrier height
	return omega*g(p) + h(p)*fs(Cs) + (1.0-h(p))*fl(Cl);
}

double d2f_dc2(const double p, const double c, const double Cs, const double Cl)
{
	double invR = R(p, Cs, Cl);
	if (fabs(invR)>epsilon) invR = 1.0/invR;
	return d2fl_dc2(Cl)*d2fs_dc2(Cs)*invR;
}

void simple_progress(int step, int steps) {
	if (step==0)
		std::cout<<" ["<<std::flush;
	else if (step==steps-1)
		std::cout<<"•] "<<std::endl;
	else if (step % (steps/20) == 0)
		std::cout<<"• "<<std::flush;
}

void export_energy(bool silent)
{
	const int np=100;
	const int nc=100;
	const double dp = (1.0/np);
	const double dc = (1.0/nc);
	const double pmin=-dp, pmax=1.0+dp;
	const double cmin=-dc, cmax=1.0+dc;


	bool randomize=true;

	std::ofstream ef("energy.csv");
	ef<<"p";
	for (int i=0; i<nc+1; i++) {
		double c = cmin+(cmax-cmin)*dc*i;
		ef<<",f(c="<<c<<')';
	}
	ef<<'\n';
	for (int i=0; i<np+1; i++) {
		if (silent)
			simple_progress(i, np+1);
		double p = pmin+(pmax-pmin)*dp*i;
		ef << p;
		for (int j=0; j<nc+1; j++) {
			double c = cmin+(cmax-cmin)*dc*j;
			double cs(0.0), cl(1.0);
			double res=iterateConc(1.0e-6,5e6,randomize,p,c,cs,cl,silent);
			ef << ',' << f(p, c, cs, cl);
		}
		ef << '\n';
	}
	ef.close();
}

template <class T>
double F1(const T& p, const T& c, const T& Cs, const T& Cl){return h(p)*Cs + (1.0-h(p))*Cl - c;}

template <class T>
double F2(const T& Cs, const T& Cl){return dfs_dc(Cs) - dfl_dc(Cl);}

/* Given const phase fraction (p) and concentration (c), iteratively determine
 * the solid (Cs) and liquid (Cl) fictitious concentrations that satisfy the
 * equal chemical potential constraint. Pass p and c by const value,
 * Cs and Cl by non-const reference to update in place. This allows use of this
 * single function to both populate the LUT and interpolate values based thereupon.
 */
template<class T> double iterateConc(const double tol, const unsigned int maxloops, bool randomize, const T& p, const T& c, T& Cs, T& Cl, bool silent)
{
	int rank=0;
	#ifdef MPI_VERSION
	rank=MPI::COMM_WORLD.Get_rank();
	#endif

	double res = std::sqrt(pow(F1(p,c,Cs,Cl),2.0) + pow(F2(Cs,Cl),2.0)); // initial residual

	double bestCs = Cs;
	double bestCl = Cl;
	double bestRes = res;

	const double cmin(-5.0), cmax(6.0); // min, max values for Cs, Cl before triggering random re-initialization

	// Iterate until either the matrix is solved (residual<tolerance)
	// or patience wears out (loop>maxloops, likely due to infinite loop).
	unsigned int l=0;
	unsigned int resets=0;
	while (l<maxloops && res>tol) {
		// copy current values as "old guesses"
		T Cso = Cs;
		T Clo = Cl;
		T detJ = ( h(p)*d2fl_dc2(Clo) + (1.0-h(p))*d2fs_dc2(Cso) ); // determinant of the Jacobian matrix
		T weight = (detJ>epsilon) ? 1.0/detJ : 0.0;
		T ds = d2fl_dc2(Clo)*F1(p,c,Cso,Clo) + (1.0-h(p))*F2(Cso,Clo);
		T dl = d2fs_dc2(Cso)*F1(p,c,Cso,Clo) -      h(p) *F2(Cso,Clo);

		Cs = Cso + weight * ds;
		Cl = Clo + weight * dl;
		if (randomize && (Cs<cmin || Cs>cmax || Cl<cmin || Cl>cmax)) {
			// If Newton falls out of bounds, shake everything up.
			// Helps the numerics, but won't fix fundamental problems.
			Cs = double(rand())/RAND_MAX;
			Cl = double(rand())/RAND_MAX;
			resets++;
		}

		res = std::sqrt(pow(F1(p,c,Cs,Cl),2.0) + pow(F2(Cs,Cl),2.0));

		if (res < bestRes) {
			bestCs = Cs;
			bestCl = Cl;
			bestRes = res;
		}

		l++;
	}
	Cs = bestCs;
	Cl = bestCl;
	res = bestRes;
	if (!silent && rank==0) {
		if (l>=maxloops)
			printf("p=%.4f, c=%.4f, iter=%-8u:\tCs=%.4f, Cl=%.4f, res=%.2e, %7u resets (failed to converge)\n", p, c, l, Cs, Cl, res, resets);
		else
			printf("p=%.4f, c=%.4f, iter=%-8u:\tCs=%.4f, Cl=%.4f, res=%.2e, %7u resets\n",                      p, c, l, Cs, Cl, res, resets);
	}
	return res;
}

template<class T> double interpolateConc(const LUTGRID& lut, const T& p, const T& c, T& Cs, T& Cl)
{
	// Determine indices in (p,c) space for LUT access
	int idp_lo = int( double(p+dp)*double(LUTnp+2)/(1.0+2.0*dp) );
	int idc_lo = int( double(c+dc)*double(LUTnc+2)/(1.0+2.0*dc) );
	if (idp_lo>LUTnp)
		idp_lo = LUTnp;
	if (idp_lo<-1)
		idp_lo = -1;
	if (idc_lo>LUTnc)
		idc_lo = LUTnc;
	if (idc_lo<-1)
		idc_lo = -1;
	int idp_hi = idp_lo+1;
	int idc_hi = idc_lo+1;

	// Bound p,c in LUT neighborhood
	const double p_lo = dp*(idp_lo-1);
	const double p_hi = dp*(idp_hi-1);
	const double c_lo = dc*(idc_lo-1);
	const double c_hi = dc*(idc_hi-1);

	if (1) { // Interpolate Cs
		// Determine limiting values of Cs at corners
		const double C00 = lut[idp_lo][idc_lo][0]; // lower left
		const double C01 = lut[idp_lo][idc_hi][0]; // upper left
		const double C10 = lut[idp_hi][idc_lo][0]; // lower right
		const double C11 = lut[idp_hi][idc_hi][0]; // upper right

		// Linear interpolation to estimate Cs: if the LUT mesh is sufficiently dense, no further work is required. (Big If.)
		if (idp_lo==idp_hi) {
			// Linear interpolation in c, only
			Cs = C00 + ((C01-C00)/(c_hi-c_lo))*(c-c_lo);
		} else if (idc_lo==idc_hi) {
			// Linear interpolation in phi, only
			Cs = C00 + ((C10-C00)/(p_hi-p_lo))*(p-p_lo);
		} else {
			// Bilinear interpolation to estimate Cs
			Cs = (  (p_hi-p   )*(c_hi-c   )*C00
	               +(p   -p_lo)*(c_hi-c   )*C10
			       +(p_hi-p   )*(c   -c_lo)*C01
			       +(p   -p_lo)*(c   -c_lo)*C11
			     )/((p_hi-p_lo)*(c_hi-c_lo));
		}
	}

	if (1) { // Interpolate Cl
		// Determine limiting values of Cl at corners
		const double C00 = lut[idp_lo][idc_lo][1]; // lower left
		const double C01 = lut[idp_lo][idc_hi][1]; // upper left
		const double C10 = lut[idp_hi][idc_lo][1]; // lower right
		const double C11 = lut[idp_hi][idc_hi][1]; // upper right

		// Linear interpolation to estimate Cl: if the LUT mesh is sufficiently dense, no further work is required. (Big If.)
		if (idp_lo==idp_hi) {
			// Linear interpolation in c, only
			Cl = C00 + ((C01-C00)/(c_hi-c_lo))*(c-c_lo);
		} else if (idc_lo==idc_hi) {
			// Linear interpolation in phi, only
			Cl = C00 + ((C10-C00)/(p_hi-p_lo))*(p-p_lo);
		} else {
			// Bilinear interpolation to estimate Cl
			Cl = (  (p_hi-p   )*(c_hi-c   )*C00
	               +(p   -p_lo)*(c_hi-c   )*C10
			       +(p_hi-p   )*(c   -c_lo)*C01
			       +(p   -p_lo)*(c   -c_lo)*C11
			     )/((p_hi-p_lo)*(c_hi-c_lo));
		}
	}
	return std::max(std::max(lut[idp_lo][idc_lo][2], lut[idp_lo][idc_hi][2]),
	                std::max(lut[idp_hi][idc_lo][2], lut[idp_hi][idc_hi][2]));
}
#endif

#include"MMSP.main.hpp"
