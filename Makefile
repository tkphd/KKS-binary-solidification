# Makefile
# GNU makefile for KKS binary phase transformation model
# Questions/comments to trevor.keller@nist.gov (Trevor Keller)

includes = -I$(MMSP_PATH)/include

# compilers/flags
compiler = g++
pcompiler = mpic++
flags = -O3 -Wall

# the program

KKS: KKS.cpp
	$(compiler) $(flags) $(includes) $< -o $@ -lz -lgsl -lgslcblas -fopenmp

gKKS: KKS.cpp
	$(compiler) $(flags) $(includes) $< -o $@ -lz -lgsl -lgslcblas

parallel: KKS.cpp
	$(pcompiler) $(flags) $(includes) -include mpi.h $< -o $@ -lz

# utilities

mmsp2pc: mmsp2pc.cpp
	$(compiler) $(flags) $(includes) $< -o $@ -lz

clean:
	rm -f KKS gKKS parallel
