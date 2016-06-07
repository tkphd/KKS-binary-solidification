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



// Interface interpolation function
double h(const double p);
double hprime(const double p);

// Double well potential
double g(const double p);
double gprime(const double p);

double k(); // equilibrium partition coefficient for solidification

// phase-field diffusivity
double Q(const double p, const double Cs, const double Cl);

double fl(const double c);       // liquid free energy density

double fs(const double c);       // solid free energy density

double dfl_dc(const double c);   // first derivative of fl w.r.t. c

double dfs_dc(const double c);   // first derivative of fs w.r.t. c

double d2fl_dc2(const double c); // second derivative of fl w.r.t. c

double d2fs_dc2(const double c); // second derivative of fs w.r.t. c

double R(const double p, const double Cs, const double Cl); // denominator for dCs, dCl, df

double dCl_dc(const double p, const double Cs, const double Cl); // first derivative of Cl w.r.t. c

double dCs_dc(const double p, const double Cs, const double Cl); // first derivative of Cs w.r.t. c

double f(const double p, const double c, const double Cs, const double Cl); // free energy density

double d2f_dc2(const double p, const double c, const double Cs, const double Cl); // second derivative of f w.r.t. c

void simple_progress(int step, int steps); // thread-compatible pared-down version of print_progress

template<int dim, typename T> void print_values(const MMSP::grid<dim,MMSP::vector<T> >& oldGrid, const int rank);

/* Given const phase fraction (p) and concentration (c), iteratively determine
 * the solid (Cs) and liquid (Cl) fictitious concentrations that satisfy the
 * equal chemical potential constraint. Pass p and c by const value,
 * Cs and Cl by non-const reference to update in place. This allows use of this
 * single function to both populate the LUT and interpolate values based thereupon.
 */

int commonTangent_f(const gsl_vector* x, void* params, gsl_vector* f);
int commonTangent_df(const gsl_vector* x, void* params, gsl_matrix* J);
int commonTangent_fdf(const gsl_vector* x, void* params, gsl_vector* f, gsl_matrix* J);



/* Given const LUTGRID, phase fraction (p), and concentration (c), apply
 * linear interpolation to estimate Cs and Cl. For a dense LUT mesh, values
 * can be used directly. Otherwise, they serve as a good "best guess" for
 * iterative calculation, which should converge quickly.
 */
class interpolator
{
public:
	// constructor
    interpolator(const LUTGRID& lut);
	// destructor
    ~interpolator();
	// accessor
	template <typename T> void interpolate(const T& p, const T& c, T& Cs, T& Cl);

private:
    int nx;
    int ny;
    double* xa;
    double* ya;

    gsl_interp_accel* xacc1;
    gsl_interp_accel* xacc2;
    gsl_interp_accel* xacc3;

    gsl_interp_accel* yacc1;
    gsl_interp_accel* yacc2;
    gsl_interp_accel* yacc3;

    const gsl_interp2d_type* algorithm;

    double* CSa;
    double* CLa;
    double* Ra;

    gsl_spline2d* CSspline;
    gsl_spline2d* CLspline;
    gsl_spline2d* Rspline;

};

struct rparams {
	double p;
	double c;
};

class rootsolver
{
public:
	// constructor
	rootsolver();
	// destructor
	~rootsolver();
	// accessor
	template <typename T> double solve(const T& p, const T& c, T& Cs, T& Cl);

private:
	const size_t n;
	const size_t maxiter;
	const double tolerance;
	gsl_vector* x;
	struct rparams par;
	const gsl_multiroot_fdfsolver_type* algorithm;
	gsl_multiroot_fdfsolver* solver;
	gsl_multiroot_function_fdf mrf;

};

void export_energy(rootsolver& NRGsolver); // exports free energy curves to energy.csv
