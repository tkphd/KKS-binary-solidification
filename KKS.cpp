// KKS.cpp
// Algorithms for 2D and 3D isotropic binary alloy solidification
// Questions/comments to trevor.keller@nist.gov (Trevor Keller)

#ifndef KKS_UPDATE
#define KKS_UPDATE
#include<cmath>
#include"MMSP.hpp"
#include"KKS.hpp"

// Note: KKS.hpp contains important declarations and comments. Have a look.

// Ideal solution model parameters
//#define CALPHAD            // uncomment this line to switch ON the ideal solution model
const double fA = 1.0;     // equilibrium free energy of pure liquid A
const double fB = 2.5;     // equilibrium free energy of pure liquid B
const double Theta = 0.4;      // homologous, isothermal temperature
const double RT = 8.314*Theta; // units?

// Kinetic and model parameters
const double meshres = 3.75e-2; // dx=dy
const double dt = 6.0e-2;
const double Ds = 0.0, Dl = 3.0e-3; // diffusion constants
const double eps_sq = 0.25;
const double ps0 = 1.0, pl0 = 0.0; // initial phase fractions
const double cBs = 0.4, cBl = 0.4; // initial concentrations
const double randCamp = 0.0; // amplitude for initial composition fluctuations
const double randPamp = 0.0; // amplitude for initial phase fluctuations

// Parabolic model parameters

// WARNING: If you change the following 4 lines, you must regenerate the lookup table!


const double  Cse = 0.3,  Cle = 0.7;    // equilibrium concentration
const double  As = 150.0, Al = 150.0;   // 2*curvature of parabola
const double dCs = 10.0, dCl  = 10.0;   // y-axis offset
const double omega = 1.125*As;          // double well height

// Resolution of the constant chem. pot. composition lookup table
const int LUTnc = 50; // number of points along c-axis
const int LUTnp = 50; // number of points along p-axis
const double dp = 1.0/LUTnp;
const double dc = 1.0/LUTnc;


// Newton-Raphson root finding parameters
const unsigned int refloop = 1e7;// ceiling to kill infinite loops in iterative scheme: reference table threshold
const unsigned int fasloop = 1e7;// ceiling to kill infinite loops in iterative scheme: fast update() threshold
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
			pureconc(n)[0] = dc*x[1];       // guess Cs
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
	unsigned int nSol=0, nLiq=0;
	if (dim==1) {
		int L=512;
		GRID1D initGrid(5,0,L);
		for (int d=0; d<dim; d++)
			dx(initGrid,d) = meshres;

		double ctot = 0.0;
		double radius=(g1(initGrid,0)-g0(initGrid,0))/4;
		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid,n);
			double r = std::abs(x[0] - (g1(initGrid,0)-g0(initGrid,0))/2);
			if (r < radius) { // Solid
				nSol++;
				initGrid(n)[0] = ps0;
				initGrid(n)[1] = cBs;
			} else {
				nLiq++;
				initGrid(n)[0] = pl0;
				initGrid(n)[1] = cBl;
			}
			initGrid(n)[4] = interpolateConc(pureconc, initGrid(n)[0], initGrid(n)[1], initGrid(n)[2], initGrid(n)[3]);
			ctot += initGrid(n)[1]*dx(initGrid);
		}
		unsigned int nTot = nSol+nLiq;
		#ifdef MPI_VERSION
		unsigned int mySol(nSol), myLiq(nLiq), myTot(nTot);
		MPI::COMM_WORLD.Allreduce(&mySol, &nSol, 1, MPI_UNSIGNED, MPI_SUM);
		MPI::COMM_WORLD.Allreduce(&myLiq, &nLiq, 1, MPI_UNSIGNED, MPI_SUM);
		MPI::COMM_WORLD.Allreduce(&myTot, &nTot, 1, MPI_UNSIGNED, MPI_SUM);
		#endif
		double C0 = (double(nLiq)*cBs + double(nSol)*cBl) / double(nSol+nLiq); // weighted average of solid + liquid
		assert(C0>0.);
		if (rank==0)
			std::cout<<"System is "<<(100*nSol)/nTot<<"% solid, "<<(100*nLiq)/nTot<<"% liquid."<<std::endl;

		output(initGrid,filename);

		#ifdef MPI_VERSION
		double myct(ctot);
		MPI::COMM_WORLD.Allreduce(&myct, &ctot, 1, MPI_DOUBLE, MPI_SUM);
		#endif
		if (rank==0) {
			std::ofstream cfile("c.log");
			cfile<<ctot<<std::endl;
			cfile.close();
		}

	} else if (dim==2) {
		int L=100; //64;
		GRID2D initGrid(5,0,2*L,0,L/4);
		//double radius = 20.0;
		double radius = (g1(initGrid,0)-g0(initGrid,0))/4;
		for (int d=0; d<dim; d++)
			dx(initGrid,d) = meshres;

		double ctot = 0.0;
		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid,n);
			//double r = sqrt(pow(radius-x[0]%64,2)+pow(radius-x[1]%64,2));
			double r = std::abs(x[0] - (g1(initGrid,0)-g0(initGrid,0))/2);
			if (r<radius) { // Solid
				nSol++;
				initGrid(n)[0] = std::max(0.0,std::min(1.0,  ps0  +randPamp*double(rand()-RAND_MAX/2)/RAND_MAX));
				initGrid(n)[1] = std::max(0.0,std::min(1.0,  cBs  +randCamp*double(rand()-RAND_MAX/2)/RAND_MAX));
			} else {
				nLiq++;
				initGrid(n)[0] = std::max(0.0,std::min(1.0,  pl0  +randPamp*double(rand()-RAND_MAX/2)/RAND_MAX));
				initGrid(n)[1] = std::max(0.0,std::min(1.0,  cBl  +randCamp*double(rand()-RAND_MAX/2)/RAND_MAX));
			}
			initGrid(n)[4] = interpolateConc(pureconc, initGrid(n)[0], initGrid(n)[1], initGrid(n)[2], initGrid(n)[3]);
			ctot += initGrid(n)[1]*dx(initGrid)*dy(initGrid);
		}
		unsigned int nTot = nSol+nLiq;
		#ifdef MPI_VERSION
		unsigned int mySol(nSol), myLiq(nLiq), myTot(nTot);
		MPI::COMM_WORLD.Allreduce(&mySol, &nSol, 1, MPI_UNSIGNED, MPI_SUM);
		MPI::COMM_WORLD.Allreduce(&myLiq, &nLiq, 1, MPI_UNSIGNED, MPI_SUM);
		MPI::COMM_WORLD.Allreduce(&myTot, &nTot, 1, MPI_UNSIGNED, MPI_SUM);
		#endif
		double C0 = (double(nLiq)*cBs + double(nSol)*cBl) / double(nSol+nLiq); // weighted average of solid + liquid
		assert(C0>0.);
		if (rank==0)
			std::cout<<"System is "<<(100*nSol)/nTot<<"% solid, "<<(100*nLiq)/nTot<<"% liquid."<<std::endl;

		output(initGrid,filename);

		#ifdef MPI_VERSION
		double myct(ctot);
		MPI::COMM_WORLD.Allreduce(&myct, &ctot, 1, MPI_DOUBLE, MPI_SUM);
		#endif
		if (rank==0) {
			std::ofstream cfile("c.log");
			cfile<<ctot<<std::endl;
			cfile.close();
		}
	} else if (dim==3) {
		int L=64;
		double radius=22.0;
		GRID3D initGrid(5,0,L,0,L,0,L);
		for (int d=0; d<dim; d++)
			dx(initGrid,d) = meshres;

		double ctot = 0.0;
		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid,n);
			double r = sqrt(pow(radius-x[0]%64,2)+pow(radius-x[1]%64,2));
			if (r<radius) { // Solid
				nSol++;
				initGrid(n)[0] = ps0;
				initGrid(n)[1] = cBs;
			} else {
				nLiq++;
				initGrid(n)[0] = pl0;
				initGrid(n)[1] = cBl;
			}
			initGrid(n)[4] = interpolateConc(pureconc, initGrid(n)[0], initGrid(n)[1], initGrid(n)[2], initGrid(n)[3]);
			ctot += initGrid(n)[1]*dx(initGrid)*dy(initGrid)*dz(initGrid);
		}
		unsigned int nTot = nSol+nLiq;
		#ifdef MPI_VERSION
		unsigned int mySol(nSol), myLiq(nLiq), myTot(nTot);
		MPI::COMM_WORLD.Allreduce(&mySol, &nSol, 1, MPI_UNSIGNED, MPI_SUM);
		MPI::COMM_WORLD.Allreduce(&myLiq, &nLiq, 1, MPI_UNSIGNED, MPI_SUM);
		MPI::COMM_WORLD.Allreduce(&myTot, &nTot, 1, MPI_UNSIGNED, MPI_SUM);
		#endif
		double C0 = (double(nLiq)*cBs + double(nSol)*cBl) / double(nSol+nLiq); // weighted average of solid + liquid
		assert(C0>0.);
		if (rank==0)
			std::cout<<"System is "<<(100*nSol)/nTot<<"% solid, "<<(100*nLiq)/nTot<<"% liquid."<<std::endl;

		output(initGrid,filename);

		#ifdef MPI_VERSION
		double myct(ctot);
		MPI::COMM_WORLD.Allreduce(&myct, &ctot, 1, MPI_DOUBLE, MPI_SUM);
		#endif
		if (rank==0) {
			std::ofstream cfile("c.log");
			cfile<<ctot<<std::endl;
			cfile.close();
		}
	} else {
		std::cerr<<"ERROR: "<<dim<<"-dimensional domains not supported."<<std::endl;
		exit(-1);
	}

	if (rank==0)
		printf("Equilibrium Cs=%.2f, Cl=%.2f\n", Cs_e(fA, fB, RT), Cl_e(fA, fB, RT));

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
		dV *= meshres;
	}

	std::ofstream cfile;
	if (rank==0)
		cfile.open("c.log",std::ofstream::out | std::ofstream::app);

	for (int step=0; step<steps; step++) {
		if (rank==0)
			print_progress(step, steps);

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

			vector<T> laps = laplacian(oldGrid, x);
			const T lapPhi = laps[0];
			const T lapC   = laps[1];
			const T lapCs  = laps[2];
			const T lapCl  = laps[3];

			// No elegant way to compute dot product of gradients, so...
			vector<vector<T> > grads = gradient(oldGrid, x);
			double gradPgradCs = 0.0;
			double gradPgradCl = 0.0;
			for (int d=0; d<dim; d++) {
				gradPgradCs += grads[d][0]*grads[d][2];
				gradPgradCl += grads[d][0]*grads[d][3];
			} // ... sorry you had to see that.


			/* ============================= *
			 * Solve the Equations of Motion *
			 * ============================= */


			// Update phi (Eqn. 6.97)
			newGrid(n)[0] = phi_old + (dt/omega)*( (eps_sq/omega)*lapPhi - gprime(phi_old)
			                                      + hprime(phi_old)*( fl(Cl_old)-fs(Cs_old)-(Cl_old-Cs_old)*dfl_dc(Cl_old) )/omega );


			// Update c (Eqn. 6.100)
			/*
			const double div_Qh_gradCs   = ( Q(phi_old)*hprime(phi_old) + Qprime(phi_old)*h(phi_old)      )*(gradPgradCs)
			                               + Q(phi_old)*h(phi_old)*lapCs;
			const double div_Q1mh_gradCl = (-Q(phi_old)*hprime(phi_old) + Qprime(phi_old)*(1.0-h(phi_old)))*(gradPgradCl)
			                               + Q(phi_old)*(1.0-h(phi_old))*lapCl;
			/*
			// Remove phi-dependence from Q, for testing ONLY
			const double div_Qh_gradCs   = Q(0.5)*(hprime(phi_old)*(gradPgradCs) + h(phi_old)*lapCs);
			const double div_Q1mh_gradCl = Q(0.5)*(-hprime(phi_old)*(gradPgradCl) + (1.0-h(phi_old))*lapCl);
			*/

			//newGrid(n)[1] = c_old + dt*Dl*(div_Qh_gradCs + div_Q1mh_gradCl);

			// KKS Eqn. 33
			newGrid(n)[1] = c_old + dt*Dl*(Q(phi_old)*lapC + Q(phi_old)*hprime(phi_old)*(Cl_old - Cs_old)*lapPhi);


			// Update Cs, Cl
			bool silent=true, randomize=false;
			newGrid(n)[4] = interpolateConc(pureconc, newGrid(n)[0], newGrid(n)[1], newGrid(n)[2], newGrid(n)[3]);
			newGrid(n)[4] = iterateConc(reftol, refloop, randomize, newGrid(n)[0], newGrid(n)[1], newGrid(n)[2], newGrid(n)[3], silent);

			// ~ fin ~
		}
		swap(oldGrid,newGrid);
		ghostswap(oldGrid);

		// Compute total mass
		double ctot=0.0;
		for (int n=0; n<nodes(oldGrid); n++)
			ctot += oldGrid(n)[1]*dV;
		#ifdef MPI_VERSION
		double myct(ctot);
		MPI::COMM_WORLD.Allreduce(&myct, &ctot, 1, MPI_DOUBLE, MPI_SUM);
		#endif
		if (rank==0)
			cfile<<ctot<<std::endl;
	}
	if (rank==0)
		cfile.close();

	double ptot=0.0;
	unsigned int ntot = nodes(oldGrid);
	for (int n=0; n<nodes(oldGrid); n++)
		ptot += oldGrid(n)[0];

	#ifdef MPI_VERSION
	unsigned int myP(ptot);
	unsigned int myN(ntot);
	MPI::COMM_WORLD.Allreduce(&myP, &ptot, 1, MPI_DOUBLE, MPI_SUM);
	MPI::COMM_WORLD.Allreduce(&myN, &ntot, 1, MPI_UNSIGNED, MPI_SUM);
	#endif
	double wps = (100.0*ptot)/ntot;
	double wpl = (100.0*(ntot-ptot))/ntot;
	if (rank==0)
		printf("System is %.2f%% solid, %.2f%% liquid (%.1f%% total).\n", wps, wpl, wps+wpl);

}


} // namespace MMSP

double fl(const double& c)
{
	#ifdef CALPHAD
	// 10-th order polynomial fit to S. an Mey Cu-Ni CALPHAD database
	return  8.68879987e+06*pow(c,10)
	       -4.34769184e+07*pow(c,9)
	       +9.38854514e+07*pow(c,8)
	       -1.14566000e+08*pow(c,7)
	       +8.69653340e+07*pow(c,6)
	       -4.26686195e+07*pow(c,5)
	       +1.36792909e+07*pow(c,4)
	       -2.86974665e+06*pow(c,3)
	       +4.13883481e+05*pow(c,2)
	       -4.76063602e+04*c
	       -8.79244411e+04;
	#else
	return Al*pow(c-Cle,2.0)+dCl;
	#endif
}

double fs(const double& c)
{
	#ifdef CALPHAD
	// 10-th order polynomial fit to S. an Mey Cu-Ni CALPHAD database
	return  8.69364423e+06*pow(c,10)
	       -4.35012558e+07*pow(c,9)
	        9.39376102e+07*pow(c,8)
	       -1.14628203e+08*pow(c,7)
	        8.70104787e+07*pow(c,6)
	       -4.26891048e+07*pow(c,5)
	        1.36850373e+07*pow(c,4)
	       -2.87088429e+06*pow(c,3)
	        4.14959341e+05*pow(c,2)
	       -5.20719622e+04*c
	       -8.60332083e+04;
	#else
	return As*pow(c-Cse,2.0)+dCs;
	#endif
}


double dfl_dc(const double& c)
{
	#ifdef CALPHAD
	return  10.0*8.68879987e+06*pow(c,9)
	       -9.0*4.34769184e+07*pow(c,8)
	       +8.0*9.38854514e+07*pow(c,7)
	       -7.0*1.14566000e+08*pow(c,6)
	       +6.0*8.69653340e+07*pow(c,5)
	       -5.0*4.26686195e+07*pow(c,4)
	       +4.0*1.36792909e+07*pow(c,3)
	       -3.0*2.86974665e+06*pow(c,2)
	       +2.0*4.13883481e+05*c
	       -4.76063602e+04;
	#else
	return 2.0*Al*(c-Cle);
	#endif
}

double dfs_dc(const double& c)
{
	#ifdef CALPHAD
	return  10.0*8.69364423e+06*pow(c,9)
	       -9.0*4.35012558e+07*pow(c,8)
	        8.0*9.39376102e+07*pow(c,7)
	       -7.0*1.14628203e+08*pow(c,6)
	        6.0*8.70104787e+07*pow(c,5)
	       -5.0*4.26891048e+07*pow(c,4)
	        4.0*1.36850373e+07*pow(c,3)
	       -3.0*2.87088429e+06*pow(c,2)
	        2.0*4.14959341e+05*c
	       -5.20719622e+04;
	#else
	return 2.0*As*(c-Cse);
	#endif
}

double d2fl_dc2(const double& c)
{
	#ifdef CALPHAD
	return  90.0*8.68879987e+06*pow(c,8)
	       -72.0*4.34769184e+07*pow(c,7)
	       +56.0*9.38854514e+07*pow(c,6)
	       -42.0*1.14566000e+08*pow(c,5)
	       +30.0*8.69653340e+07*pow(c,4)
	       -20.0*4.26686195e+07*pow(c,3)
	       +12.0*1.36792909e+07*pow(c,2)
	       -6.0*2.86974665e+06*c
	       +2.0*4.13883481e+05;
	#else
	return 2.0*Al;
	#endif
}

double d2fs_dc2(const double& c)
{
	#ifdef CALPHAD
	return  80.0*8.69364423e+06*pow(c,8)
	       -72.0*4.35012558e+07*pow(c,7)
	        56.0*9.39376102e+07*pow(c,6)
	       -42.0*1.14628203e+08*pow(c,5)
	        30.0*8.70104787e+07*pow(c,4)
	       -20.0*4.26891048e+07*pow(c,3)
	        12.0*1.36850373e+07*pow(c,2)
	       -6.0*2.87088429e+06*c
	        2.0*4.14959341e+05;
	#else
	return 2.0*As;
	#endif
}

double R(const double& p, const double& Cs, const double& Cl)
{
	// denominator for dCs, dCl, df
	return h(p)*d2fl_dc2(Cl) + (1.0-h(p))*d2fs_dc2(Cs);
}

double dCl_dc(const double& p, const double& Cs, const double& Cl)
{
	double invR = R(p, Cs, Cl);
	if (fabs(invR)>epsilon) invR = 1.0/invR;
	return d2fl_dc2(Cl)*invR;
}

double dCs_dc(const double& p, const double& Cs, const double& Cl)
{
	double invR = R(p, Cs, Cl);
	if (fabs(invR)>epsilon) invR = 1.0/invR;
	return d2fs_dc2(Cs)*invR;
}

double Cl_e(const double& fa, const double& fb, const double& rt) {
	#ifdef CALPHAD
	return 1.0 / (1.0 + std::exp((fa-fb)/(rt)));
	#else
	return Cle;
	#endif
}

double Cs_e(const double& fa, const double& fb, const double& rt) {
	#ifdef CALPHAD
	return std::exp((fb-fa)/(rt)) / (1.0 + std::exp((fb-fa)/(rt)));
	#else
	return Cse;
	#endif
}

double k()
{
	// Partition coefficient, from solving dfs_dc = 0 and dfl_dc = 0
	#ifdef CALPHAD
	return 0.48300 / 0.33886;
	#else
	return Cse/Cle;
	#endif
}

double f(const double& p, const double& c, const double& Cs, const double& Cl)
{
	//const double w = 1.0; // well barrier height
	return omega*g(p) + h(p)*fs(Cs) + (1.0-h(p))*fl(Cl);
}

double d2f_dc2(const double& p, const double& c, const double& Cs, const double& Cl)
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
	const int nc=50;
	const int np=50;
	const double cmin=-0.0625, cmax=1.0625;
	const double pmin=-0.3333, pmax=1.3333;

	const double dc = (1.0/nc);
	const double dp = (1.0/np);

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


/* Given const phase fraction (p) and concentration (c), iteratively determine
 * the solid (Cs) and liquid (Cl) fictitious concentrations that satisfy the
 * equal chemical potential constraint. Pass p and c by const value,
 * Cs and Cl by non-const reference to update in place. This allows use of this
 * single function to both populate the LUT and interpolate values based thereupon.
 */
template<class T> double iterateConc(const double tol, const unsigned int maxloops, bool randomize, const T p, const T c, T& Cs, T& Cl, bool silent)
{
	int rank=0;
	#ifdef MPI_VERSION
	rank=MPI::COMM_WORLD.Get_rank();
	#endif

	double f1 = h(p)*Cs + (1.0-h(p))*Cl - c;
	double f2 = dfs_dc(Cs) - dfl_dc(Cl);
	double res = std::sqrt(pow(f1,2.0) + pow(f2,2.0)); // initial residual

	double bestCs(c), bestCl(c), bestRes(res);
	const double cmin(-5.0), cmax(6.0); // min, max values for Cs, Cl before triggering random re-initialization

	// Iterate until either the matrix is solved (residual<tolerance)
	// or patience wears out (loop>maxloops, likely due to infinite loop).
	unsigned int l=0;
	unsigned int resets=0;
	while (l<maxloops && res>tol) {
		// copy current values as "old guesses"
		T Cso = Cs;
		T Clo = Cl;
		double W = h(p)*d2fl_dc2(Clo) + (1.0-h(p))*d2fs_dc2(Cso);
		f1 = h(p)*Cso + (1.0-h(p))*Clo - c;
		f2 = dfs_dc(Cso) - dfl_dc(Clo);
		T ds = (fabs(W)<epsilon)? 0.0: ( d2fl_dc2(Clo)*f1 + (1.0-h(p))*f2)/W;
		T dl = (fabs(W)<epsilon)? 0.0: (-d2fs_dc2(Cso)*f1 + h(p)*f2)/W;

		Cs = Cso + ds;
		Cl = Clo + dl;
		if (Cs<cmin || Cs>cmax || Cl<cmin || Cl>cmax) {
			if (randomize) {
				// If Newton falls out of bounds, shake everything up.
				// Helps the numerics, but won't fix fundamental problems.
				Cs = double(rand())/RAND_MAX;
				Cl = double(rand())/RAND_MAX;
				resets++;
			} /*else {
				l=maxloops;
			}*/
		}

		f1 = h(p)*Cs + (1.0-h(p))*Cl - c; // at convergence, this equals zero
		f2 = dfs_dc(Cs) - dfl_dc(Cl);     // this, too
		res = std::sqrt(pow(f1,2.0) + pow(f2,2.0));

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
			printf("p=%.4f, c=%.4f, iter=%-8u:\tCs=%.4f, Cl=%.4f, res=%.2e, %7lu resets (failed to converge)\n", p, c, l, Cs, Cl, res, resets);
		else
			printf("p=%.4f, c=%.4f, iter=%-8u:\tCs=%.4f, Cl=%.4f, res=%.2e, %7lu resets\n",                      p, c, l, Cs, Cl, res, resets);
	}
	return res;
}

template<class T> double interpolateConc(const LUTGRID& lut, const T p, const T c, T& Cs, T& Cl)
{
	// Determine indices in (p,c) space for LUT access
	int idp_lo = int( (p+dp)*(LUTnp+2)/(1.0+2.0*dp) );
	int idc_lo = int( (c+dc)*(LUTnc+2)/(1.0+2.0*dc) );
	if (idp_lo>LUTnp+1)
		idp_lo = LUTnp;
	if (idp_lo<-1)
		idp_lo = -1;
	if (idc_lo>LUTnc+1)
		idc_lo = LUTnc;
	if (idc_lo<-1)
		idc_lo = -1;
	int idp_hi = idp_lo+1;
	int idc_hi = idc_lo+1;

	// Bound p,c in LUT neighborhood
	const double p_lo = dp*idp_lo;
	const double p_hi = dp*idp_hi;
	const double c_lo = dc*idc_lo;
	const double c_hi = dc*idc_hi;

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
	return std::max(std::max(lut[idp_lo][idc_lo][2], lut[idp_lo][idc_hi][2]), std::max(lut[idp_hi][idc_lo][2], lut[idp_hi][idc_hi][2]));
}
#endif

#include"MMSP.main.hpp"
