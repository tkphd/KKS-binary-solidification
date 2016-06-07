// KKS.cpp
// Algorithms for 2D and 3D isotropic binary alloy solidification
// using RWTH database for Cu-Ni system, CuNi_RWTH.tdb,
// extracted from the COST507 Light Alloys Database.
// Questions/comments to trevor.keller@nist.gov (Trevor Keller)

// This implementation depends on the GNU Scientific Library
// for multivariate root finding and interpolation algorithms


#ifndef KKS_UPDATE
#define KKS_UPDATE
#include<cmath>
#include<gsl/gsl_blas.h>
#include<gsl/gsl_vector.h>
#include<gsl/gsl_multiroots.h>
#include<gsl/gsl_interp2d.h>
#include<gsl/gsl_spline2d.h>

#include"MMSP.hpp"
#include"KKS.hpp"

// Note: KKS.hpp contains important declarations and comments. Have a look.

// 10-th order polynomial fitting coefficients from CALPHAD
double calCs[11] = { 6.19383857e+03,-3.09926825e+04, 6.69261368e+04,-8.16668934e+04,
                     6.19902973e+04,-3.04134700e+04, 9.74968659e+03,-2.04529002e+03,
                     2.95622845e+02,-3.70962613e+01,-6.12900561e+01};
double calCl[11] = { 6.18692878e+03,-3.09579439e+04, 6.68516329e+04,-8.15779791e+04,
                     6.19257214e+04,-3.03841489e+04, 9.74145735e+03,-2.04379606e+03,
                     2.94796431e+02,-3.39127135e+01,-6.26373908e+01};

// Solidus and Liquidus compositions from CALPHAD
const double Cse = 0.5413,  Cle = 0.3940;

// Numerical stability (Courant-Friedrich-Lewy) parameters
const double epsilon = 1.0e-10;  // what to consider zero to avoid log(c) explosions
const double CFL = 0.2; // controls timestep


const bool useNeumann = true;

const bool planarTest = false;


// Kinetic and model parameters
const double meshres = 0.075; // dx=dy
const double eps_sq = 1.25;
const double a_int = 2.5; // alpha, prefactor of interface width
const double halfwidth = 5.0*meshres; // half the interface width
const double omega = 2.0*eps_sq*pow(a_int/halfwidth,2.0);
const double dt_plimit = CFL*meshres/eps_sq;          // speed limit based on numerical viscosity
const double dt_climit = CFL*pow(meshres,2.0)/eps_sq; // speed limit based on diffusion timescale
const double dt = std::min(dt_plimit, dt_climit);
const double ps0 = 1.0, pl0 = 0.0; // initial phase fractions
const double cBs = Cle + 0.50*(Cse-Cle); // initial solid concentration
const double cBl = Cle + 0.50*(Cse-Cle); // initial solid concentration

// Resolution of the constant chem. pot. composition lookup table
const int LUTnc = 1000;        // number of points along c-axis
const int LUTnp = 1000;        // number of points along p-axis
const int LUTmargin = 1; // number of points below zero and above one
const double LUTdp = 1.0/LUTnp;
const double LUTdc = 1.0/LUTnc;
const double LUTpmin = -LUTdp*LUTmargin;
const double LUTpmax =  LUTdp*(LUTnp+LUTmargin);
const double LUTcmin = -LUTdc*LUTmargin;
const double LUTcmax =  LUTdc*(LUTnc+LUTmargin);


/* =============================================== *
 * Implement MMSP kernels, generate() and update() *
 * =============================================== */

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

	// Construct solver for LUT generation
	rootsolver LUTsolver;

	if (nrg_not_found) {
		// Print out the free energy for phi in [-0.01,1.01] with slices of c in [-0.01,1.01] for inspection.
		if (rank==0)
			std::cout<<"Sampling free energy landscape, exporting to energy.csv."<<std::endl;

		#ifdef MPI_VERSION
		MPI::COMM_WORLD.Barrier();
		#endif
		if (rank==0) export_energy(LUTsolver);
		#ifdef MPI_VERSION
		MPI::COMM_WORLD.Barrier();
		#endif
	}

	if (lut_not_found) {
		/* Generate Cs,Cl look-up table (LUT) using Newton-Raphson method, as
		 * outlined in Provatas' Appendix C3 but using libgsl for speed and accuracy
		 * Store results in pureconc, which contains two fields:
		 * 0. Cs, fictitious composition of pure liquid
		 * 1. Cl, fictitious composition of pure solid
		 *
		 * The grid is discretized over phi (axis 0) and c (axis 1).
		*/
		if (rank==0)
			std::cout<<"Writing look-up table of Cs, Cl to consistentC.lut. Please be patient..."<<std::endl;
		LUTGRID pureconc(3, -LUTmargin,LUTnp+LUTmargin+1, -LUTmargin,LUTnc+LUTmargin+1);
		dx(pureconc,0) = LUTdp; // different resolution in phi
		dx(pureconc,1) = LUTdc; // and c is not unreasonable

		#ifndef MPI_VERSION
		#pragma omp parallel for schedule(dynamic) private(LUTsolver)
		#endif
		for (int n=0; n<nodes(pureconc); n++) {
			simple_progress(n,nodes(pureconc));
			vector<int> x = position(pureconc,n);
			pureconc(n)[0] = 0.5; // guess Cs
			pureconc(n)[1] = 0.5; // guess Cl
			pureconc(n)[2] = LUTsolver.solve(LUTdp*x[0], LUTdc*x[1], pureconc(n)[0], pureconc(n)[1]);
		}

		output(pureconc,"consistentC.lut");
	}

	// Read concentration look-up table from disk, in its entirety, even in parallel. Should be relatively small.
	#ifndef MPI_VERSION
	const int ghost=0;
	LUTGRID pureconc("consistentC.lut",ghost);
	#else
	MPI::COMM_WORLD.Barrier();
	LUTGRID pureconc(3, -LUTmargin,LUTnp+LUTmargin+1, -LUTmargin,LUTnc+LUTmargin+1);
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
	   4. deviation from equilibrium (chemical potential difference)
	 */
	vector<double> solidValue(4, 0.0);
	solidValue[0] = ps0;
	solidValue[1] = cBs;
	solidValue[2] = 0.5;
	solidValue[3] = 0.5;
	LUTsolver.solve(solidValue[0], solidValue[1], solidValue[2], solidValue[3]);

	vector<double> liquidValue(4, 0.0);
	liquidValue[0] = pl0;
	liquidValue[1] = cBl;
	liquidValue[2] = 0.5;
	liquidValue[3] = 0.5;
	LUTsolver.solve(liquidValue[0], liquidValue[1], liquidValue[2], liquidValue[3]);

	if (dim==1) {
		int L=512;
		GRID1D initGrid(4,0,L);
		for (int d=0; d<dim; d++) {
			dx(initGrid,d) = meshres;
			if (useNeumann && x0(initGrid,d)==g0(initGrid,d))
				b0(initGrid,d) = Neumann;
			else if (useNeumann && x1(initGrid,d)==g1(initGrid,d))
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
			if (r < radius)
				initGrid(n).copy(solidValue);
			else
				initGrid(n).copy(liquidValue);
			ctot += initGrid(n)[1]*dx(initGrid);
			ftot += f(initGrid(n)[0], initGrid(n)[1], initGrid(n)[2], initGrid(n)[3])*dx(initGrid);
		}
		// Add gradient to ftot
		ghostswap(initGrid);
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
			cfile<<ctot<<'\t'<<ftot<<'\t'<<CFL<<'\t'<<1.0<<std::endl;
			cfile.close();
		}

	} else if (dim==2) {
		int L = planarTest ? 1000 : 256;
		int Ly = 32;
		GRID2D initGrid(4,0,L,0,Ly);
		for (int d=0; d<dim; d++) {
			dx(initGrid,d) = meshres;
			if (useNeumann && x0(initGrid,d)==g0(initGrid,d))
				b0(initGrid,d) = Neumann;
			else if (useNeumann && x1(initGrid,d)==g1(initGrid,d))
				b1(initGrid,d) = Neumann;
		}

		double ctot = 0.0, ftot = 0.0;
		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid,n);
			if (planarTest) {
				// Planar interface
				if (x[0] < L/2)
					initGrid(n).copy(solidValue);
				else
					initGrid(n).copy(liquidValue);
			} else {
				double ra = 15.0, rb = 8.0, rc = 8.0;
				double offset = 64.0;
				// Circular interfaces
				if ( (pow(x[0] - (ra+1   ),2) + pow(x[1] - (offset/2-ra-1  ),2) < ra*ra) ||
				     (pow(x[0] - 0.625*(offset),2) + pow(x[1] - (offset/2-rb-1  ),2) < rb*rb) ||
				     (pow(x[0] - (offset-rc-1 ),2) + pow(x[1] - (rc+1),2) < rc*rc)
				)
					initGrid(n).copy(solidValue);
				else
					initGrid(n).copy(liquidValue);
			}
			ctot += initGrid(n)[1]*dx(initGrid)*dy(initGrid);
			ftot += f(initGrid(n)[0], initGrid(n)[1], initGrid(n)[2], initGrid(n)[3])*dx(initGrid)*dy(initGrid);
		}
		// Add gradient to ftot
		ghostswap(initGrid);
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
			cfile<<ctot<<'\t'<<ftot<<'\t'<<CFL<<'\t'<<1.0<<std::endl;
			cfile.close();
		}
	} else if (dim==3) {
		int L=64;
		double radius=10.0;
		GRID3D initGrid(4,0,L,0,L,0,L);
		for (int d=0; d<dim; d++) {
			dx(initGrid,d) = meshres;
			if (useNeumann && x0(initGrid,d)==g0(initGrid,d))
				b0(initGrid,d) = Neumann;
			else if (useNeumann && x1(initGrid,d)==g1(initGrid,d))
				b1(initGrid,d) = Neumann;
		}

		double ctot = 0.0, ftot = 0.0;
		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid,n);
			double r = sqrt(pow(radius-x[0]%64,2)+pow(radius-x[1]%64,2));
			if (r<radius)
				initGrid(n).copy(solidValue);
			else
				initGrid(n).copy(liquidValue);
			ctot += initGrid(n)[1]*dx(initGrid)*dy(initGrid)*dz(initGrid);
			ftot += f(initGrid(n)[0], initGrid(n)[1], initGrid(n)[2], initGrid(n)[3])*dx(initGrid)*dy(initGrid)*dz(initGrid);
		}
		// Add gradient to ftot
		ghostswap(initGrid);
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
			cfile<<ctot<<'\t'<<ftot<<'\t'<<CFL<<'\t'<<1.0<<std::endl;
			cfile.close();
		}
	} else {
		std::cerr<<"ERROR: "<<dim<<"-dimensional domains not supported."<<std::endl;
		exit(-1);
	}

	if (rank==0)
		printf("\nEquilibrium Cs=%.2f, Cl=%.2f. Timestep dt=%.2e\n", Cse, Cle, dt);

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
	LUTGRID pureconc(3, -LUTmargin,LUTnp+LUTmargin+1, -LUTmargin,LUTnc+LUTmargin+1);
	const bool serial=true; // Please do not change this :-)
	const int ghost=1;
	pureconc.input("consistentC.lut",ghost,serial);
	#endif

	// Construct the LUT solver and interpolator
	rootsolver LUTsolver;
	interpolator LUTinterp(pureconc);

	ghostswap(oldGrid);
   	grid<dim,vector<T> > newGrid(oldGrid);
	double dV=1.0;
	for (int d=0; d<dim; d++) {
		dx(oldGrid,d) = meshres;
		dx(newGrid,d) = meshres;
		dV *= dx(oldGrid,d);
		if (useNeumann && x0(oldGrid,d) == g0(oldGrid,d)) {
			b0(oldGrid,d) = Neumann;
			b0(newGrid,d) = Neumann;
		} else if (useNeumann && x1(oldGrid,d) == g1(oldGrid,d)) {
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

		double ctot=0.0, ftot=0.0, utot=0.0, vmax=0.0;
		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(oldGrid); n++) {
			/* ============================================== *
			 * Point-wise kernel for parallel PDE integration *
			 * ============================================== */

			vector<int> x = position(oldGrid,n);

			// Cache some frequently-used reference values
			const T& phi_old = oldGrid(n)[0];
			const T& c_old   = oldGrid(n)[1];
			const T& Cs_old  = oldGrid(n)[2];
			const T& Cl_old  = oldGrid(n)[3];


			/* ======================================= *
			 * Compute Second-Order Finite Differences *
			 * ======================================= */

			double divGradP = 0.0;
			double divGradC = 0.0;
			double lapPhi = 0.0;
			double gradPsq = 0.0;
			vector<int> s(x);
			for (int d=0; d<dim; d++) {
				double weight = 1.0/pow(dx(oldGrid,d), 2.0);

				if (x[d] == x0(oldGrid,d) &&
				    x0(oldGrid,d) == g0(oldGrid,d) &&
				    useNeumann)
				{
					// Central second-order difference at lower boundary:
					// Flux_lo = grad(phi)_(i-1/2) is defined to be 0
					// Get high values
					s[d] += 1;
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
					divGradP += 0.5*weight*( (Mph+Mpc)*(ph-pc) );
					divGradC += 0.5*weight*( (Mch+Mcc)*(ch-cc) );
					lapPhi   += weight*(ph-pc);
				} else if (x[d] == x1(oldGrid,d)-1 &&
				           x1(oldGrid,d) == g1(oldGrid,d) &&
				           useNeumann)
				{
					// Central second-order difference at upper boundary:
					// Flux_hi = grad(phi)_(i+1/2) is defined to be 0
					// Get low values
					s[d] -= 1;
					const T& pl = oldGrid(s)[0];
					const T& cl = oldGrid(s)[1];
					const T& Sl = oldGrid(s)[2];
					const T& Ll = oldGrid(s)[3];
					const T Mpl = Q(pl,Sl,Ll)*hprime(pl)*(Ll-Sl);
					const T Mcl = Q(pl,Sl,Ll);
					// Get central values
					s[d] += 1;
					const T& pc = oldGrid(s)[0];
					const T& cc = oldGrid(s)[1];
					const T& Sc = oldGrid(s)[2];
					const T& Lc = oldGrid(s)[3];
					const T Mpc = Q(pc,Sc,Lc)*hprime(pc)*(Lc-Sc);
					const T Mcc = Q(pc,Sc,Lc);

					// Put 'em all together
					divGradP += 0.5*weight*( (Mpc+Mpl)*(pl-pc) );
					divGradC += 0.5*weight*( (Mcc+Mcl)*(cl-cc) );
					lapPhi   += weight*(pl-pc);
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
					lapPhi   += weight*( (ph-pc) - (pc-pl) );
					gradPsq  += weight * pow(0.5*(ph-pl), 2.0);
				}
			}

			/* ==================================================================== *
			 * Solve the Equation of Motion for phi: Kim, Kim, & Suzuki Equation 31 *
			 * ==================================================================== */

			newGrid(n)[0] = phi_old + dt*( eps_sq*lapPhi - omega*gprime(phi_old)
			                               + hprime(phi_old)*( fl(Cl_old)-fs(Cs_old)-(Cl_old-Cs_old)*dfl_dc(Cl_old) ));


			/* ================================================================== *
			 * Solve the Equation of Motion for c: Kim, Kim, & Suzuki Equation 33 *
			 * ================================================================== */

			newGrid(n)[1] = c_old + dt*(divGradC + divGradP);


			/* ================================================== *
			 * Interpolate consistent Cs and Cl from lookup table *
			 * ================================================== */

			LUTinterp.interpolate(newGrid(n)[0], newGrid(n)[1], newGrid(n)[2], newGrid(n)[3]);


			/* ====================================================================== *
			 * Collate summary & diagnostic data in OpenMP- and MPI-compatible manner *
			 * ====================================================================== */

			double myc = dV*newGrid(n)[1];
			double myf = dV*(0.5*eps_sq*gradPsq + f(newGrid(n)[0], newGrid(n)[1], newGrid(n)[2], newGrid(n)[3]));
			double myv = 0.0;
			if (newGrid(n)[0]>0.3 && newGrid(n)[0]<0.7) {
				gradPsq = 0.0;
				for (int d=0; d<dim; d++) {
					double weight = 1.0/pow(dx(newGrid,d), 2.0);
					s[d] -= 1;
					const T& pl = newGrid(s)[0];
					s[d] += 2;
					const T& ph = newGrid(s)[0];
					s[d] -= 1;
					gradPsq  += weight * pow(0.5*(ph-pl), 2.0);
				}
				myv = (newGrid(n)[0] - phi_old) / (dt * std::sqrt(gradPsq));
			}
			double myu = (fl(newGrid(n)[3])-fs(newGrid(n)[2]))/(Cle - Cse);

			#ifndef MPI_VERSION
			#pragma omp critical
			{
			#endif
			vmax = std::max(vmax,myv); // maximum velocity
			ctot += myc;               // total mass
			ftot += myf;               // total free energy
			utot += myu*myu;           // deviation from equilibrium
			#ifndef MPI_VERSION
			}
			#endif

			/* ======= *
			 * ~ fin ~ *
			 * ======= */
		}
		swap(oldGrid,newGrid);
		ghostswap(oldGrid);

		double ntot(nodes(oldGrid));
		#ifdef MPI_VERSION
		double myvm(vmax);
		double myct(ctot);
		double myft(ftot);
		double myut(utot);
		double myn(ntot);
		MPI::COMM_WORLD.Allreduce(&myct, &ctot, 1, MPI_DOUBLE, MPI_SUM);
		MPI::COMM_WORLD.Allreduce(&myft, &ftot, 1, MPI_DOUBLE, MPI_SUM);
		MPI::COMM_WORLD.Allreduce(&myvm, &vmax, 1, MPI_DOUBLE, MPI_MAX);
		MPI::COMM_WORLD.Allreduce(&myut, &utot, 1, MPI_DOUBLE, MPI_SUM);
		MPI::COMM_WORLD.Allreduce(&myn,  &ntot, 1, MPI_DOUBLE, MPI_SUM);
		#endif
		double CFLmax = (vmax * dt) / meshres;
		utot = std::sqrt(utot/ntot);
		if (rank==0)
			cfile<<ctot<<'\t'<<ftot<<'\t'<<CFLmax<<'\t'<<utot<<std::endl;
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
	double fs = 100.0*(cTot - Cle)/(Cse-Cle);
	double fl = 100.0*(Cse - cTot)/(Cse-Cle);
	if (rank==0)
		printf("System has %.2f%% solid, %.2f%% liquid, and composition %.2f%% B. Equilibrium is %.2f%% solid, %.2f%% liquid.\n",
		       wps, wpl, 100.0*cTot, fs, fl);
}


double h(const double p)
{
	return p;
}

double hprime(const double p)
{
	return 1.0;
}

double g(const double p)
{
	return pow(p,2.0) * pow(1.0-p,2.0);
}

double gprime(const double p)
{
	return 2.0*p * (1.0-p)*(1.0-2.0*p);
}

double k()
{
	// Partition coefficient, from equilibrium phase diagram
	return Cse / Cle;
}

double Q(const double p, const double Cs, const double Cl)
{
	const double Qmin = 0.01;
    return Qmin + (1.0 - Qmin) * (1.0-p)/(1.0 + k() - (1.0-k())*p);
}

double fl(const double c)
{
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
}

double fs(const double c)
{
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
}


double dfl_dc(const double c)
{
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
}

double dfs_dc(const double c)
{
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
}

double d2fl_dc2(const double c)
{
	return  90.0*calCl[0]*pow(c,8)
	       +72.0*calCl[1]*pow(c,7)
	       +56.0*calCl[2]*pow(c,6)
	       +42.0*calCl[3]*pow(c,5)
	       +30.0*calCl[4]*pow(c,4)
	       +20.0*calCl[5]*pow(c,3)
	       +12.0*calCl[6]*pow(c,2)
	       +6.0*calCl[7]*c
	       +2.0*calCl[8];
}

double d2fs_dc2(const double c)
{
	return  90.0*calCs[0]*pow(c,8)
	       +72.0*calCs[1]*pow(c,7)
	       +56.0*calCs[2]*pow(c,6)
	       +42.0*calCs[3]*pow(c,5)
	       +30.0*calCs[4]*pow(c,4)
	       +20.0*calCs[5]*pow(c,3)
	       +12.0*calCs[6]*pow(c,2)
	       +6.0*calCs[7]*c
	       +2.0*calCs[8];
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

double f(const double p, const double c, const double Cs, const double Cl)
{
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

void export_energy(rootsolver& NRGsolver)
{
	const int np=100;
	const int nc=100;
	const double dp = 1.0/np;
	const double dc = 1.0/nc;
	const double pmin=-dp, pmax=1.0+dp;
	const double cmin=-dc, cmax=1.0+dc;

	std::ofstream ef("energy.csv");
	ef<<"p";
	for (int i=0; i<nc+1; i++) {
		double c = cmin+(cmax-cmin)*dc*i;
		ef<<",f(c="<<c<<')';
	}
	ef<<'\n';
	for (int i=0; i<np+1; i++) {
		simple_progress(i, np+1);
		double p = pmin+(pmax-pmin)*dp*i;
		ef << p;
		for (int j=0; j<nc+1; j++) {
			double c = cmin+(cmax-cmin)*dc*j;
			double cs(0.5), cl(0.5);
			double res=NRGsolver.solve(p,c,cs,cl);
			ef << ',' << f(p, c, cs, cl);
		}
		ef << '\n';
	}
	ef.close();
}


/* ================================= *
 * Invoke GSL to solve for Cs and Cl *
 * ================================= */

/* Given const phase fraction (p) and concentration (c), iteratively determine
 * the solid (Cs) and liquid (Cl) fictitious concentrations that satisfy the
 * equal chemical potential constraint. Pass p and c by const value,
 * Cs and Cl by non-const reference to update in place. This allows use of this
 * single function to both populate the LUT and interpolate values based thereupon.
 */
int commonTangent_f(const gsl_vector* x, void* params, gsl_vector* f)
{
	const double p = ((struct rparams *) params)->p;
	const double c = ((struct rparams *) params)->c;

	const double Cs = gsl_vector_get(x, 0);
	const double Cl = gsl_vector_get(x, 1);

	const double f1 = h(p)*Cs + (1.0-h(p))*Cl - c;
	const double f2 = dfs_dc(Cs) - dfl_dc(Cl);

	gsl_vector_set(f, 0, f1);
	gsl_vector_set(f, 1, f2);

	return GSL_SUCCESS;
}

int commonTangent_df(const gsl_vector* x, void* params, gsl_matrix* J)
{
	const double p = ((struct rparams *) params)->p;

	const double Cs = gsl_vector_get(x, 0);
	const double Cl = gsl_vector_get(x, 1);

	// Jacobian matrix
	const double df11 = h(p);
	const double df12 = 1.0-h(p);
	const double df21 =  d2fs_dc2(Cs);
	const double df22 = -d2fl_dc2(Cl);

	gsl_matrix_set(J, 0, 0, df11);
	gsl_matrix_set(J, 0, 1, df12);
	gsl_matrix_set(J, 1, 0, df21);
	gsl_matrix_set(J, 1, 1, df22);

	return GSL_SUCCESS;
}

int commonTangent_fdf(const gsl_vector* x, void* params, gsl_vector* f, gsl_matrix* J)
{
	commonTangent_f(x, params, f);
	commonTangent_df(x, params, J);

	return GSL_SUCCESS;
}


rootsolver::rootsolver() :
	n(2), // two equations
	maxiter(5000),
	tolerance(1.0e-12)
{
	x = gsl_vector_alloc(n);

	// configure algorithm
	algorithm = gsl_multiroot_fdfsolver_gnewton; // gnewton, hybridj, hybridsj, newton
	solver = gsl_multiroot_fdfsolver_alloc(algorithm, n);

	mrf = {&commonTangent_f, &commonTangent_df, &commonTangent_fdf, n, &par};
}

template <typename T> double rootsolver::solve(const T& p, const T& c, T& Cs, T& Cl)
{
	int status;
	size_t iter = 0;

	// initial guesses
	par.p = p;
	par.c = c;
	gsl_vector_set(x, 0, Cs);
	gsl_vector_set(x, 1, Cl);

	gsl_multiroot_fdfsolver_set(solver, &mrf, x);

	do {
		iter++;
		status = gsl_multiroot_fdfsolver_iterate(solver);
		if (status) // extra points for finishing early!
			break;
		status = gsl_multiroot_test_residual(solver->f, tolerance);
	} while (status==GSL_CONTINUE && iter<maxiter);

	Cs = static_cast<T>(gsl_vector_get(solver->x, 0));
	Cl = static_cast<T>(gsl_vector_get(solver->x, 1));

	double residual = gsl_blas_dnrm2(solver->f);

	return residual;
}

rootsolver::~rootsolver()
{
	gsl_multiroot_fdfsolver_free(solver);
	gsl_vector_free(x);
}



interpolator::interpolator(const LUTGRID& lut)
{
	// System size
	const int x0 = MMSP::g0(lut, 0);
	const int y0 = MMSP::g0(lut, 1);
	nx = MMSP::g1(lut,0) - x0;
	ny = MMSP::g1(lut,1) - y0;
	const double dx = MMSP::dx(lut,0);
	const double dy = MMSP::dx(lut,1);

	// Data arrays
	xa = new double[nx];
	ya = new double[ny];
	CSa = new double[nx*ny];
	CLa = new double[nx*ny];
	Ra = new double[nx*ny];

	for (int i=0; i<nx; i++)
		xa[i] = dx*(i+x0);
	for (int j=0; j<ny; j++)
		ya[j] = dy*(j+y0);

	// GSL interpolation function
	algorithm = gsl_interp2d_bicubic; // options: gsl_interp2d_bilinear or gsl_interp2d_bicubic

	CSspline = gsl_spline2d_alloc(algorithm, nx, ny);
	CLspline = gsl_spline2d_alloc(algorithm, nx, ny);
	Rspline = gsl_spline2d_alloc(algorithm, nx, ny);

	xacc1 = gsl_interp_accel_alloc();
	xacc2 = gsl_interp_accel_alloc();
	xacc3 = gsl_interp_accel_alloc();

	yacc1 = gsl_interp_accel_alloc();
	yacc2 = gsl_interp_accel_alloc();
	yacc3 = gsl_interp_accel_alloc();

	// Initialize interpolator
	for (int n=0; n<MMSP::nodes(lut); n++) {
		MMSP::vector<int> x = MMSP::position(lut, n);
		gsl_spline2d_set(CSspline, CSa, x[0]-x0, x[1]-y0, lut(n)[0]);
		gsl_spline2d_set(CLspline, CLa, x[0]-x0, x[1]-y0, lut(n)[1]);
		gsl_spline2d_set(Rspline,  Ra,  x[0]-x0, x[1]-y0, lut(n)[2]);
	}
	gsl_spline2d_init(CSspline, xa, ya, CSa, nx, ny);
	gsl_spline2d_init(CLspline, xa, ya, CLa, nx, ny);
	gsl_spline2d_init(Rspline,  xa, ya, Ra,  nx, ny);
}

interpolator::~interpolator()
{
	gsl_spline2d_free(CSspline);
	gsl_spline2d_free(CLspline);
	gsl_spline2d_free(Rspline);

	gsl_interp_accel_free(xacc1);
	gsl_interp_accel_free(xacc2);
	gsl_interp_accel_free(xacc3);

	gsl_interp_accel_free(yacc1);
	gsl_interp_accel_free(yacc2);
	gsl_interp_accel_free(yacc3);

	delete [] xa; xa=NULL;
	delete [] ya; ya=NULL;
	delete [] CSa; CSa=NULL;
	delete [] CLa; CLa=NULL;
	delete [] Ra; Ra=NULL;
}

template <typename T> void interpolator::interpolate(const T& p, const T& c, T& Cs, T& Cl)
{
	if (p<LUTpmin || c<LUTcmin ||
	    p>LUTpmax || c>LUTcmax)
	{
		printf("GSL interp2d error: phi=%.4f, c=%.4f\n", p, c);
		std::exit(-1);
	}
	Cs = static_cast<T>(gsl_spline2d_eval(CSspline, p, c, xacc1, yacc1));
	Cl = static_cast<T>(gsl_spline2d_eval(CLspline, p, c, xacc2, yacc2));
}


#endif

#include"MMSP.main.hpp"
