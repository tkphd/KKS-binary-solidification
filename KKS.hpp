// KKS.hpp
// Declarations for 2D and 3D isotropic binary alloy solidification
// Questions/comments to trevor.keller@nist.gov (Trevor Keller)

std::string PROGRAM = "KKS";
std::string MESSAGE = "Isotropic phase field solidification example code";

typedef MMSP::grid<1,MMSP::vector<double> > GRID1D;
typedef MMSP::grid<2,MMSP::vector<double> > GRID2D;
typedef MMSP::grid<3,MMSP::vector<double> > GRID3D;

// KKS requires a look-up table of Cs and Cl consistent with the constraint
// of constant chemical potential over the full range of C and phi. This provides
// initial guesses for iterative reconciliation of Cs and Cl with phi and c at each
// grid point and timestep.
typedef MMSP::grid<2,MMSP::vector<double> > LUTGRID;

/* Given const phase fraction (p) and concentration (c), iteratively determine
 * the solid (Cs) and liquid (Cl) fictitious concentrations that satisfy the
 * equal chemical potential constraint. Pass p and c by const value,
 * Cs and Cl by non-const reference to update in place. This allows use ofthis
 * single function to both populate the LUT and interpolate values based thereupon.
 */
template<typename T,typename U> void iterateConc(const T p, const T c, U& Cs, U& Cl);

