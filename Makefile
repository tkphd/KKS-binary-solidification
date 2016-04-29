# Makefile
# GNU makefile for KKS binary phase transformation model
# Questions/comments to trevor.keller@nist.gov (Trevor Keller)

# includes
incdir = $(MMSP_PATH)/include

# compilers/flags
compiler = icc
flags = -O3 -Wall -I$(incdir)
pcompiler = mpic++
pflags = $(flags) -include mpi.h

# the program
KKS: KKS.cpp
	$(compiler) $(flags) $< -o $@ -lz -fopenmp

gKKS: KKS.cpp
	g++ $(flags) $< -o $@ -lz -fopenmp

parallel: KKS.cpp
	$(pcompiler) $(pflags) $< -o $@ -lz

mmsp2pc: mmsp2pc.cpp
	$(compiler) $(flags) $< -o $@ -lz

clean:
	rm -f KKS gKKS parallel
