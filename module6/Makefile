# instruct h5c++ to use mpic++ as the compiler and linker command
export HDF5_CXX = mpic++
export HDF5_CXXLINKER = mpic++

# set compile flags
FLAGS = -O3 -DNDEBUG -g -Wall -march=native -std=c++17

.PHONY: clean all

all: fwc_sequential fwc_parallel

fwc_sequential: fwc_sequential.cpp
	h5c++ $(FLAGS) $(LIB) $(INC) -o fwc_sequential fwc_sequential.cpp

fwc_parallel: fwc_parallel.cpp
	h5c++ $(FLAGS) $(LIB) $(INC) -o fwc_parallel fwc_parallel.cpp

clean:
	rm -f fwc_sequential fwc_parallel *.o
