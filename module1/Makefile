CXX := g++
OPT=-O1 -Wall

CXXFLAGS := $(OPT) -Wall -Wno-unused-const-variable -std=c++17 -march=native
default: SIR SIR_python
INC := $(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags)

SIR: SIR.cpp
	$(CXX) SIR.cpp -o SIR $(CXXFLAGS)
SIR_python: SIR_python.cpp
	$(CXX) $(INC) SIR_python.cpp -o SIR_python$(shell python3-config --extension-suffix) -fPIC -shared $(CXXFLAGS)

clean:
	rm -f SIR SIR_python 
	rm -f SIR_python$(shell python3-config --extension-suffix) 