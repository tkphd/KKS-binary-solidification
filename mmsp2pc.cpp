// mmsp2pc.cpp
// INPUT: MMSP grid containing vector data with at least two fields
// OUTPUT: Pairs of comma-delimited coordinates representing position in (v0,v1) phase space
//         Expected usage is for (phase,composition) spaces, hence pc
// Questions/comments to trevor.keller@gmail.com (Trevor Keller)

#include"MMSP.hpp"
#include<zlib.h>
#include<map>
#include<set>

int main(int argc, char* argv[]) {
	// command line error check
	if (argc < 2) {
		std::cout << "Usage: " << argv[0] << " [--help] infile [outfile]\n\n";
		exit(-1);
	}

	const int scale = 50000; // resolution, for converting floats to integers

	// help diagnostic
	if (std::string(argv[1]) == "--help") {
		std::cout << argv[0] << ": convert MMSP grid data to (p,c) points.\n";
		std::cout << "Usage: " << argv[0] << " [--help] infile [outfile]\n\n";
		std::cout << "Questions/comments to trevor.keller@gmail.com (Trevor Keller).\n\n";
		exit(0);
	}

	// file open error check
	std::ifstream input(argv[1]);
	if (!input) {
		std::cerr << "File input error: could not open " << argv[1] << ".\n\n";
		exit(-1);
	}

	// generate output file name
	std::stringstream filename;
	if (argc < 3)
		filename << std::string(argv[1]).substr(0, std::string(argv[1]).find_last_of(".")) << ".xy";
	else
		filename << argv[2];

	// file open error check
	std::ofstream output(filename.str().c_str());
	if (!output) {
		std::cerr << "File output error: could not open ";
		std::cerr << filename.str() << "." << std::endl;
		exit(-1);
	}

	// read data type
	std::string type;
	getline(input, type, '\n');

	// grid type error check
	if (type.substr(0, 4) != "grid") {
		std::cerr << "File input error: file does not contain grid data." << std::endl;
		exit(-1);
	}

	// parse data type
	bool bool_type = (type.find("bool") != std::string::npos);
	bool char_type = (type.find("char") != std::string::npos);
	bool unsigned_char_type = (type.find("unsigned char") != std::string::npos);
	bool int_type = (type.find("int") != std::string::npos);
	bool unsigned_int_type = (type.find("unsigned int") != std::string::npos);
	bool long_type = (type.find("long") != std::string::npos);
	bool unsigned_long_type = (type.find("unsigned long") != std::string::npos);
	bool short_type = (type.find("short") != std::string::npos);
	bool unsigned_short_type = (type.find("unsigned short") != std::string::npos);
	bool float_type = (type.find("float") != std::string::npos);
	bool double_type = (type.find("double") != std::string::npos);
	bool long_double_type = (type.find("long double") != std::string::npos);

	bool scalar_type = (type.find("scalar") != std::string::npos);
	bool vector_type = (type.find("vector") != std::string::npos);
	bool sparse_type = (type.find("sparse") != std::string::npos);

	if (not bool_type    and
	    not char_type    and  not unsigned_char_type   and
	    not int_type     and  not unsigned_int_type    and
	    not long_type    and  not unsigned_long_type   and
	    not short_type   and  not unsigned_short_type  and
	    not float_type   and
	    not double_type  and  not long_double_type) {
		std::cerr << "File input error: unknown grid data type." << std::endl;
		exit(-1);
	}

	// read grid dimension
	int dim;
	input >> dim;

	// read number of fields
	int fields;
	input >> fields;

	// read grid sizes
	int x0[3] = {0, 0, 0};
	int x1[3] = {0, 0, 0};
	for (int i = 0; i < dim; i++)
		input >> x0[i] >> x1[i];

	// read cell spacing
	float dx[3] = {1.0, 1.0, 1.0};
	for (int i = 0; i < dim; i++)
		input >> dx[i];

	// ignore trailing endlines
	input.ignore(10, '\n');


	// determine byte order: 01 AND 01 = 01; 01 AND 10 = 00.
	std::string byte_order;
	if (0x01 & static_cast<int>(1)) byte_order = "LittleEndian";
	else byte_order = "BigEndian";

	// read number of blocks
	int blocks;
	input.read(reinterpret_cast<char*>(&blocks), sizeof(blocks));

	std::map<int,std::set<int> > phase;

	for (int i = 0; i < blocks; i++) {
		// read block limits
		int lmin[3] = {0, 0, 0};
		int lmax[3] = {0, 0, 0};
		for (int j = 0; j < dim; j++) {
			input.read(reinterpret_cast<char*>(&lmin[j]), sizeof(lmin[j]));
			input.read(reinterpret_cast<char*>(&lmax[j]), sizeof(lmax[j]));
		}
		int blo[dim];
	    int bhi[dim];
    	// read boundary conditions
	    for (int j = 0; j < dim; j++) {
    	  input.read(reinterpret_cast<char*>(&blo[j]), sizeof(blo[j]));
	      input.read(reinterpret_cast<char*>(&bhi[j]), sizeof(bhi[j]));
    	}

		// write grid data
		if (vector_type && fields>1) {
			if (float_type) {
				if (dim == 1) {
					MMSP::grid<1, MMSP::vector<float> > GRID(argv[1]);
					for (int k = 0; k < MMSP::nodes(GRID); k++)
						phase[int(scale*GRID(k)[0])].insert(int(scale*GRID(k)[1]));
				} else if (dim == 2) {
					MMSP::grid<2, MMSP::vector<float> > GRID(argv[1]);
					for (int k = 0; k < MMSP::nodes(GRID); k++)
						phase[int(scale*GRID(k)[0])].insert(int(scale*GRID(k)[1]));
				} else if (dim == 3) {
					MMSP::grid<3, MMSP::vector<float> > GRID(argv[1]);
					for (int k = 0; k < MMSP::nodes(GRID); k++)
						phase[int(scale*GRID(k)[0])].insert(int(scale*GRID(k)[1]));
				}
			}
			if (double_type) {
				if (dim == 1) {
					MMSP::grid<1, MMSP::vector<double> > GRID(argv[1]);
					for (int k = 0; k < MMSP::nodes(GRID); k++)
						phase[int(scale*GRID(k)[0])].insert(int(scale*GRID(k)[1]));
				} else if (dim == 2) {
					MMSP::grid<2, MMSP::vector<double> > GRID(argv[1]);
					for (int k = 0; k < MMSP::nodes(GRID); k++)
						phase[int(scale*GRID(k)[0])].insert(int(scale*GRID(k)[1]));
				} else if (dim == 3) {
					MMSP::grid<3, MMSP::vector<double> > GRID(argv[1]);
					for (int k = 0; k < MMSP::nodes(GRID); k++)
						phase[int(scale*GRID(k)[0])].insert(int(scale*GRID(k)[1]));
				}
			}
			if (long_double_type) {
				if (dim == 1) {
					MMSP::grid<1, MMSP::vector<long double> > GRID(argv[1]);
					for (int k = 0; k < MMSP::nodes(GRID); k++)
						phase[int(scale*GRID(k)[0])].insert(int(scale*GRID(k)[1]));
				} else if (dim == 2) {
					MMSP::grid<2, MMSP::vector<long double> > GRID(argv[1]);
					for (int k = 0; k < MMSP::nodes(GRID); k++)
						phase[int(scale*GRID(k)[0])].insert(int(scale*GRID(k)[1]));
				} else if (dim == 3) {
					MMSP::grid<3, MMSP::vector<long double> > GRID(argv[1]);
					for (int k = 0; k < MMSP::nodes(GRID); k++)
						phase[int(scale*GRID(k)[0])].insert(int(scale*GRID(k)[1]));
				}
			}
		}
	}

	for (std::map<int,std::set<int> >::const_iterator j=phase.begin(); j!=phase.end(); j++)
		for (std::set<int>::const_iterator i=j->second.begin(); i!=j->second.end(); i++)
			output << double(j->first)/scale << ',' << double(*i)/scale << '\n';

	output.close();
	return 0;
}
