// KKS.cpp
// Algorithms for 2D and 3D isotropic binary alloy solidification
// Questions/comments to trevor.keller@nist.gov (Trevor Keller)

#ifndef KKS_UPDATE
#define KKS_UPDATE
#include<cmath>
#include"MMSP.hpp"
#include"KKS.hpp"

// Note: KKS.hpp contains important declarations and comments. Have a look.

const double fA = 1.0;      // equilibrium free energy of pure liquid A
const double fB = 2.5;      // equilibrium free energy of pure liquid B
const double R = 8.314e-3;  // J/cm3-K
const double T = 373.15;    // Isothermal.

const unsigned int maxloops = 1e6;// ceiling to kill infinite loops in iterative scheme
const double tol = 1.0e-8; // tolerance for Cs, Cl scheme to satisfy equal chemical potential
const double epsilon = 1.0e-6; // what to consider zero to avoid log(c) explosions

namespace MMSP{

void generate(int dim, const char* filename)
{
	int rank=0;
	#ifdef MPI_VERSION
	rank=MPI::COMM_WORLD.Get_rank();
	#endif
	/* Grid contains four fields:
	 * 0. phi, phase fraction solid. Phi=1 means Solid.
	 * 1. c, concentration of component A
	 * 2. Cs, fictitious composition of solid
	 * 3. Cl, fictitious composition of liquid
	 */
	const double cBs = 0.45; // initial solid concentration
	const double cBl = 0.65; // initial liquid concentration

	unsigned int nSol=0, nLiq=0;
	if (dim==1) {
		int L=1024;
		GRID1D initGrid(4,0,L);

		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid,n);
			double r = 32-x[0]%64;
			if (r<16.0) { // Solid
				nSol++;
				initGrid(n)[0] = 1.0;
				initGrid(n)[1] = cBs;
			} else {
				nLiq++;
				initGrid(n)[0] = 0.0;
				initGrid(n)[1] = cBl;
			}
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

		for (int n=0; n<nodes(initGrid); n++) {
			initGrid(n)[0] = C0;
			initGrid(n)[1] = C0;
		}

		output(initGrid,filename);
	} else if (dim==2) {
		int L=256;
		GRID2D initGrid(4,0,2*L,0,L);

		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid,n);
			double r = sqrt(pow(32-x[0]%64,2)+pow(32-x[1]%64,2));
			if (r<16.0) { // Solid
				nSol++;
				initGrid(n)[0] = 1.0;
				initGrid(n)[1] = cBs;
			} else {
				nLiq++;
				initGrid(n)[0] = 0.0;
				initGrid(n)[1] = cBl;
			}
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

		for (int n=0; n<nodes(initGrid); n++) {
			initGrid(n)[0] = C0;
			initGrid(n)[1] = C0;
		}

		output(initGrid,filename);
	} else if (dim==3) {
		int L=64;
		GRID3D initGrid(4,0,2*L,0,L,0,L/4);

		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid,n);
			double r = sqrt(pow(32-x[0]%64,2)+pow(32-x[1]%64,2));
			if (r<16.0) { // Solid
				nSol++;
				initGrid(n)[0] = 1.0;
				initGrid(n)[1] = cBs;
			} else {
				nLiq++;
				initGrid(n)[0] = 0.0;
				initGrid(n)[1] = cBl;
			}
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

		for (int n=0; n<nodes(initGrid); n++) {
			initGrid(n)[0] = C0;
			initGrid(n)[1] = C0;
		}

		output(initGrid,filename);
	} else {
		std::cerr<<"ERROR: "<<dim<<"-dimensional domains not supported."<<std::endl;
		exit(-1);
	}

	/* Generate Cs,Cl look-up table (LUT) using Newton-Raphson method, outlined in Provatas' Appendix C3
	 * Store results in pureconc, which contains two fields:
	 * 0. Cl, fictitious composition of pure liquid
	 * 1. Cs, fictitious composition of pure solid
	 *
	 * The grid is discretized over phi (axis 0) and c (axis 1).
	*/
	int LUTres[2] = {4, 4};
	LUTGRID pureconc(2,0,1+LUTres[0],0,1+LUTres[1]);
	double dp = 1.0/LUTres[0];
	double dc = 1.0/LUTres[1];
	dx(pureconc,0) = dp; // different resolution in phi
	dx(pureconc,1) = dc; // and c is not unreasonable

	for (int n=0; n<nodes(pureconc); n++) {
		vector<int> x = position(pureconc,n);
		pureconc(n)[0] = double(rand())/RAND_MAX; //dc*double(x[1]); //(cBl + dc*double(x[1]))/2.0;
		pureconc(n)[1] = double(rand())/RAND_MAX; //dc*double(x[1]); //(cBs + dc*double(x[1]))/2.0;
		iterateConc(dp*double(x[0]), dc*double(x[1]), pureconc(n)[0], pureconc(n)[1]);
	}

	output(pureconc,"consistentC.lut");
}

template <int dim, typename T> void update(grid<dim,vector<T> >& oldGrid, int steps)
{
	int rank=0;
    #ifdef MPI_VERSION
    rank = MPI::COMM_WORLD.Get_rank();
    #endif

	ghostswap(oldGrid);

   	grid<dim,vector<T> > newGrid(oldGrid);

	double dt = 0.01;

	for (int step=0; step<steps; step++) {
		if (rank==0)
			print_progress(step, steps);

		for (int n=0; n<nodes(oldGrid); n++) {
			const vector<T>& phi = oldGrid(n);

			// compute laplacian
			vector<T> lap = laplacian(oldGrid,n);

			// compute sum of squares
			T sum = 0.0;
			for (int j=0; j<fields(oldGrid); j++) {
				sum += phi[j]*phi[j];
			}

			// compute update values
			for (int j=0; j<fields(oldGrid); j++) {
				newGrid(n)[j] = phi[j] - dt*(-phi[j]-pow(phi[j],3)+2.0*(phi[j]*sum-lap[j]));
			}
		}
		swap(oldGrid,newGrid);
		ghostswap(oldGrid);
	}
}


} // namespace MMSP


double fl(const double& c)
{
	// ideal solution model for liquid free energy
	if (c < epsilon)
		return fA;
	else if (1.0-c < epsilon)
		return fB;
	return (1.0-c)*fA + c*fB + R*T*((1.0-c)*log(1.0-c) + c*log(c));
}

double fs(const double& c)
{
	// ideal solution model for solid free energy, transformed from liquid
	double delta = -0.25; // negative solidifies, positive melts
	if (c < epsilon)
		return fB+delta;
	else if (1.0-c < epsilon)
		return fA+delta;
	return c*(fA+delta) + (1.0-c)*(fB+delta) + R*T*(c*log(c) + (1.0-c)*log(1.0-c));
}

double dfl_dc(const double& c)
{
	if (std::min(c,1.0-c) < epsilon)
		return fB-fA;
	return fB - fA + R*T*(log(1.0-c) - log(c));
}

double dfs_dc(const double& c)
{
	if (std::min(c,1.0-c) < epsilon)
		return fA-fB;
	return fA - fB + R*T*(log(1.0-c) - log(c));
}

double d2fl_dc2(const double& c)
{
	return R*T/(c*(1.0-c));
}

double d2fs_dc2(const double& c)
{
	return R*T/(c*(1.0-c));
}

double k()
{
	// Partition coefficient, from solving dfs_dc = 0 and dfl_dc = 0
	double Cl_e = 1.0 / (1.0 + std::exp((fA-fB)/(R*T)));
	double Cs_e = std::exp((fB-fA)/(R*T)) / (1.0 + std::exp((fB-fA)/(R*T)));

	return Cs_e / Cl_e;
}

double f(const double& p, const double& c, const double& Cs, const double& Cl)
{
	const double w = 0.5; // well barrier height
	return w*g(p) + h(p)*fs(Cs) + (1.0-h(p))*fl(Cl);
}

double d2f_dc2(const double& p, const double& c, const double& Cs, const double& Cl)
{
	double R = h(p)*d2fl_dc2(Cl) + (1.0-h(p))*d2fs_dc2(Cs);
	return d2fl_dc2(Cl)*d2fs_dc2(Cs)/R;
}

/* Given const phase fraction (p) and concentration (c), iteratively determine
 * the solid (Cs) and liquid (Cl) fictitious concentrations that satisfy the
 * equal chemical potential constraint. Pass p and c by const value,
 * Cs and Cl by non-const reference to update in place. This allows use ofthis
 * single function to both populate the LUT and interpolate values based thereupon.
 */
template<class T> void iterateConc(const T p, const T c, T& Cs, T& Cl)
{
	const double epsilon=1.0e-12;
	double res = 1.0; // residual for Newton-Raphson scheme

	// Iterate until either the matrix is solved (residual<tolerance)
	// or patience wears out (loop>maxloops, likely due to infinite loop).
	unsigned int l=0;
	while (l<maxloops && res>tol) {
		// copy current values as "old guesses"
		double Cso = Cs;
		double Clo = Cl;
		double W = h(p)*d2fl_dc2(Clo) + (1.0-h(p))*d2fs_dc2(Cso);
		double f1 = h(p)*Cso + (1.0-h(p))*Clo - c;
		double f2 = dfs_dc(Cso) - dfl_dc(Clo);
		double ds = (W<epsilon)?0.0:( d2fl_dc2(Clo)*f1 + (1.0-h(p))*f2)/W;
		double dl = (W<epsilon)?0.0:(-d2fs_dc2(Cso)*f1 + h(p)*f2)/W;

		Cs = Cso + ds;
		Cl = Clo + dl;
		if (Cs<0.0) Cs=0;
		else if (Cs>1.0) Cs=1;
		if (Cl<0.0) Cl=0;
		else if (Cl>1.0) Cl=1;

		res = std::sqrt(pow(Cs-Cso,2.0) + pow(Cl-Clo,2.0));

		/*
		invW = h(p)*d2fl_dc2(Cl) + (1.0-h(p))*d2fs_dc2(Cs);
		if (fabs(invW) > 1.0e-9) invW = 1.0/invW;
		f1 = h(p)*Cs + (1.0-h(p))*Cl - c;
		f2 = dfs_dc(Cs) - dfl_dc(Cl);
		ds = invW*( d2fl_dc2(Cl)*f1 + (1.0-h(p))*f2);
		dl = invW*(-d2fs_dc2(Cs)*f1 + h(p)*f2);

		res = std::sqrt(pow(dl,2.0) + pow(ds,2.0));
		*/
		l++;
		//printf("p=%.2f, c=%.2f, iter=%-8u: Cs=%.3f, Cl=%.3f, res=%.2e\n", p, c, l, Cs, Cl, res);
	}
	if (l>=maxloops)
		printf("p=%.2f, c=%.2f, iter=%-8u: Cs=%.3f, Cl=%.3f, res=%.2e  (failed to converge)\n", p, c, l, Cs, Cl, res);
	else
		printf("p=%.2f, c=%.2f, iter=%-8u: Cs=%.3f, Cl=%.3f, res=%.2e\n", p, c, l, Cs, Cl, res);
}

#endif

#include"MMSP.main.hpp"
