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
template<class T> double iterateConc(const double tol, const unsigned int maxloops, const T p, const T c, T& Cs, T& Cl);

double h(const double& p)     {return pow(p,3.0) * (6.0*pow(p,2.0)-15.0*p+10.0);                            }
double hprime(const double& p){return 30.0 * pow(p,2.0)*pow(1.0-p,2.0); }
double g(const double& p)     {return pow(p,2.0) * pow(1.0-p,2.0);}
double gprime(const double& p){return 2.0*p * (2.0*p-1.0)*(p-1.0);    }

double fl(const double& c);       // ideal solution model for liquid free energy density

double fs(const double& c);       // ideal solution model for solid free energy density

double dfl_dc(const double& c);   // first derivative of fl w.r.t. c

double dfs_dc(const double& c);   // first derivative of fs w.r.t. c

double d2fl_dc2(const double& c); // second derivative of fl w.r.t. c

double d2fs_dc2(const double& c); // second derivative of fs w.r.t. c

double R(const double& p, const double& Cs, const double& Cl); // denominator for dCs, dCl, df

double dCl_dc(const double& p, const double& Cs, const double& Cl); // first derivative of Cl w.r.t. c

double dCs_dc(const double& p, const double& Cs, const double& Cl); // first derivative of Cs w.r.t. c

double f(const double& p, const double& c, const double& Cs, const double& Cl); // free energy density

double d2f_dc2(const double& p, const double& c, const double& Cs, const double& Cl); // second derivative of f w.r.t. c

double Cl_e(const double& fa, const double& fb, const double& rt); // equilbrium Cl

double Cs_e(const double& fa, const double& fb, const double& rt); // equilbrium Cs

double k();                       // Partition coefficient, from solving dfs_dc = 0 and dfl_dc = 0

void print_energy(); // exports free energy curves to energy.csv

