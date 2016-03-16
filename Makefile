# Makefile
# GNU makefile for KKS binary phase transformation model
# Questions/comments to trevor.keller@nist.gov (Trevor Keller)

# includes
incdir = $(MMSP_PATH)/include

# compilers/flags
compiler = g++
flags = -O3 -std=c++11 -I $(incdir)
pcompiler = mpic++
pflags = $(flags) -include mpi.h

# the program
KKS: KKS.cpp
	$(compiler) $(flags) $< -o $@ -lz

parallel: KKS.cpp
	$(pcompiler) $(pflags) $< -o $@ -lz

clean:
	rm -f KKS parallel
