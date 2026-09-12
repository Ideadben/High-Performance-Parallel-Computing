CXX := g++

#######################################################
# Problem size definition
NFREQ ?= 16384

#######################################################
# Set NUMA_ALLOCATOR = Y to use NUMA aware memory allocator 
NUMA_ALLOCATOR = N

# Convert NUMA_ALLOCATOR=Y to a preprocessor define
ifeq ($(NUMA_ALLOCATOR),Y)
NUMA_FLAG = -DNUMA_ALLOCATOR
else
NUMA_FLAG =
endif

#######################################################
# Optimization flags are chosen as the last definition.
# Comment out using "#" at the begining of the line or rearrange according to your needs.
#
# Fastest executable (-ffast-math removes checking for NaNs and other things)
OPT=-O3 -ffast-math

# Add thread checking to code
#OPT=-O1 -fsanitize=thread,undefined,bounds -fPIE -pie

# Add memory / address error checking to code
#OPT=-O1 -fsanitize=leak,address,undefined,bounds -fPIE -pie

# Faster compilation time
#OPT=-O1

CXXFLAGS := $(OPT) -Wall -march=native -g -std=c++23 -DNFREQ=$(NFREQ) $(NUMA_FLAG)

default: seq mp

seq: seismogram_seq.cpp
	$(CXX) seismogram_seq.cpp $(CXXFLAGS) -o seq

mp: seismogram_omp.cpp
	$(CXX) seismogram_omp.cpp $(CXXFLAGS) -fopenmp -o mp

clean:
	rm -fr seq mp
