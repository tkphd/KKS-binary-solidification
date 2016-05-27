/* KKS.hpp
 * Declarations for 2D and 3D isotropic binary alloy solidification
 * Questions/comments to trevor.keller@nist.gov (Trevor Keller)
 */

std::string PROGRAM = "KKS";
std::string MESSAGE = "Isotropic phase field solidification example code";

typedef MMSP::grid<1,MMSP::vector<double> > GRID1D;
typedef MMSP::grid<2,MMSP::vector<double> > GRID2D;
typedef MMSP::grid<3,MMSP::vector<double> > GRID3D;

/* KKS requires a look-up table of Cs and Cl consistent with the constraint
 * of constant chemical potential over the full range of C and phi. This provides
 * initial guesses for iterative reconciliation of Cs and Cl with phi and c at each
 * grid point and timestep.
 */
typedef MMSP::grid<2,MMSP::vector<double> > LUTGRID;

/* Given const phase fraction (p) and concentration (c), iteratively determine
 * the solid (Cs) and liquid (Cl) fictitious concentrations that satisfy the
 * equal chemical potential constraint. Pass p and c by const value,
 * Cs and Cl by non-const reference to update in place. This allows use of this
 * single function to both populate the LUT and interpolate values based thereupon.
 */
template<class T> double iterateConc(const double tol, const unsigned int maxloops, bool randomize, const T& p, const T& c, T& Cs, T& Cl, bool silent);

/* Given const LUTGRID, phase fraction (p), and concentration (c), apply
 * linear interpolation to estimate Cs and Cl. For a dense LUT mesh, values
 * can be used directly. Otherwise, they serve as a good "best guess" for
 * iterative calculation, which should converge quickly.
 */
template<typename T> class interpolator
{
public:
	// constructor
    interpolator(const LUTGRID& lut){
		// System size
	    nx = MMSP::g1(lut,0) - MMSP::g0(lut, 0);
    	ny = MMSP::g1(lut,1) - MMSP::g0(lut, 1);
    	const double dx = MMSP::dx(lut,0);
    	const double dy = MMSP::dx(lut,1);

	    // Data arrays
    	xa = new double[nx];
	    ya = new double[ny];
    	CSa = new double[nx*ny];
	    CLa = new double[nx*ny];
    	Ra = new double[nx*ny];

		for (int i=0; i<nx; i++)
			xa[i] = dx*(i-1);
		for (int i=0; i<ny; i++)
			ya[i] = dy*(i-1);

	    // GSL interpolation function
    	algorithm = gsl_interp2d_bilinear; // consider gsl_interp2d_bicubic
	    CSspline = gsl_spline2d_alloc(algorithm, nx, ny);
    	CLspline = gsl_spline2d_alloc(algorithm, nx, ny);
	    Rspline = gsl_spline2d_alloc(algorithm, nx, ny);
    	xacc = gsl_interp_accel_alloc();
	    yacc = gsl_interp_accel_alloc();

    	// Initialize interpolator
	    for (int n=0; n<MMSP::nodes(lut); n++) {
    	    MMSP::vector<int> x = MMSP::position(lut, n);
        	gsl_spline2d_set(CSspline, CSa, x[0]+1, x[1]+1, lut(n)[0]);
	        gsl_spline2d_set(CLspline, CLa, x[0]+1, x[1]+1, lut(n)[1]);
    	    gsl_spline2d_set(Rspline,  Ra,  x[0]+1, x[1]+1, lut(n)[2]);
    	}
	    gsl_spline2d_init(CSspline, xa, ya, CSa, nx, ny);
    	gsl_spline2d_init(CLspline, xa, ya, CLa, nx, ny);
	    gsl_spline2d_init(Rspline, xa, ya, Ra, nx, ny);
    }

    ~interpolator(){
		gsl_spline2d_free(CSspline);
	    gsl_spline2d_free(CLspline);
    	gsl_spline2d_free(Rspline);
	    gsl_interp_accel_free(xacc);
    	gsl_interp_accel_free(yacc);

	    delete [] xa; xa=NULL;
    	delete [] ya; ya=NULL;
	    delete [] CSa; CSa=NULL;
    	delete [] CLa; CLa=NULL;
	    delete [] Ra; Ra=NULL;
	}

	// accessor
	double interpolate(const T& p, const T& c, T& Cs, T& Cl) {
		  Cs = static_cast<T>(gsl_spline2d_eval(CSspline, p, c, xacc, yacc));
		  Cl = static_cast<T>(gsl_spline2d_eval(CLspline, p, c, xacc, yacc));
		return static_cast<T>(gsl_spline2d_eval(Rspline,  p, c, xacc, yacc));
	}

private:
    size_t nx;
    size_t ny;
    double* xa;
    double* ya;

    gsl_interp_accel* xacc;
    gsl_interp_accel* yacc;
    const gsl_interp2d_type* algorithm;

    double* CSa;
    double* CLa;
    double* Ra;

    gsl_spline2d* CSspline;
    gsl_spline2d* CLspline;
    gsl_spline2d* Rspline;

};


template<class T> double interpolateConc(interpolator<T>& LUTinterp, const T& p, const T& c, T& Cs, T& Cl);

double h(const double p)     {return pow(p,3.0) * (6.0*p*p - 15.0*p + 10.0);}
double hprime(const double p){return 30.0 * pow(p,2.0)*pow(1.0-p,2.0); }

double g(const double p)     {return pow(p,2.0) * pow(1.0-p,2.0);}
double gprime(const double p){return 2.0*p * (1.0-p)*(1.0-2.0*p);}

double fl(const double c);       // ideal solution model for liquid free energy density

double fs(const double c);       // ideal solution model for solid free energy density

double dfl_dc(const double c);   // first derivative of fl w.r.t. c

double dfs_dc(const double c);   // first derivative of fs w.r.t. c

double d2fl_dc2(const double c); // second derivative of fl w.r.t. c

double d2fs_dc2(const double c); // second derivative of fs w.r.t. c

double R(const double p, const double Cs, const double Cl); // denominator for dCs, dCl, df

double dCl_dc(const double p, const double Cs, const double Cl); // first derivative of Cl w.r.t. c

double dCs_dc(const double p, const double Cs, const double Cl); // first derivative of Cs w.r.t. c

double f(const double p, const double c, const double Cs, const double Cl); // free energy density

double d2f_dc2(const double p, const double c, const double Cs, const double Cl); // second derivative of f w.r.t. c

double Cl_e(); // equilbrium Cl

double Cs_e(); // equilbrium Cs

double k(const double Cs, const double Cl){return Cs/Cl;} // Partition coefficient, from solving dfs_dc = 0 and dfl_dc = 0

double Q(const double p, const double Cs, const double Cl){return 0.01 + (1.0-p)/(1.0 + k(Cs, Cl) - (1.0-k(Cs, Cl))*p);}

double Qprime(const double p, const double Cs, const double Cl){return (-(1.0+k(Cs, Cl) - (1.0-k(Cs, Cl))*p)-(1.0-p)*(k(Cs, Cl)-1.0))
                                       / pow(1.0+k(Cs, Cl) - (1.0-k(Cs, Cl))*p,2.0);}

void simple_progress(int step, int steps); // thread-compatible pared-down version of print_progress

void export_energy(bool silent); // exports free energy curves to energy.csv

template<int dim, typename T> void print_values(const MMSP::grid<dim,MMSP::vector<T> >& oldGrid, const int rank);
