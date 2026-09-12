#define _USE_MATH_DEFINES
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <fstream>
#include "random_number.h"

double J = 1; //units of mev
double D = 1; //units of mev
double B = 0;
double T = 5; //units of kelvin
double beta = 1.0/(0.086*T);
int size = 10;
int seed = 0;
bool gif = false;
int steps = 1000;

class Spin{
    public:
        double theta;
        double phi;
        std::vector<int> neighbours {4};
        double local_energy;


        // Constructor
        Spin(double theta, double phi) 
            : theta(theta), phi(phi), neighbours(4), local_energy(0.0) {}

        // Default constructor (zero-initialized)
        Spin() : theta(0.0), phi(0.0), neighbours(4), local_energy(0.0) {}

};

class Lattice{
    public:
        int size;
        std::vector<Spin> spins;
        double energy = 0;
        std::vector<double> mag = {0.0, 0.0, 0.0};

        Lattice(int s) : size(s), spins(s * s) {}

        Spin& at(int row, int col){
            return spins[row*size + col];
        }

        double calc_tot_energy(){
            energy = 0;
            for (size_t i = 0; i <spins.size() ; i++){
                energy += 0.5 * spins[i].local_energy;
            }
            return energy;
        }

        std::vector<double> calc_tot_mag(){
            mag = {0.0, 0.0, 0.0};
            for (size_t i = 0; i <spins.size() ; i++){
                mag[0] += sin(spins[i].theta)*cos(spins[i].phi);
                mag[1] +=sin(spins[i].theta)*sin(spins[i].phi);
                mag[2] += cos(spins[i].theta);
            }
            return mag;
        }

        void save_spins(const std::string& filename) {
        std::ofstream file(filename);
        
        if (!file.is_open()) {
            std::cerr << "Error: could not open file " << filename << "\n";
            return;
        }

        // Header
        file << "index,row,col,theta,phi\n";

        for (size_t i = 0; i < spins.size(); i++) {
            int row = i / size;
            int col = i % size;
            file << i << ","
                << row << ","
                << col << ","
                << spins[i].theta << ","
                << spins[i].phi   << "\n";
        }

        file.close();
}

};



//double dmi_int(double thetai, double thetaj, double phii, double phij){
//    return D*(sin(thetai)*cos(thetaj)*(sin(phii) - cos(phii)) + cos(thetai)*sin(thetaj)*( cos(phij)- sin(phij)));
//
//};

//double diag_int(double thetai, double thetaj, double phii, double phij){
//    return J*(sin(thetai) *sin(thetaj)*cos(phii - phij) + cos(thetai) *cos(thetaj));
//};


//double dmi_int(Spin &spini, Spin &spinj){
//    return D*(sin(spini.theta)*cos(spinj.theta)*(sin(spini.phi) - cos(spini.phi)) + cos(spini.theta)*sin(spinj.theta)*( cos(spinj.phi)- sin(spinj.phi)));
//};


double cross_x(Spin &spini, Spin &spinj){
    // Si X Sj dot xhat
    return  D*(sin(spini.theta)*sin(spini.phi)*cos(spinj.theta) - cos(spini.theta)*sin(spinj.theta)* sin(spinj.phi));
}
double cross_y(Spin &spini, Spin &spinj){
    // Si X Sj dot yhat
    return  D*( cos(spini.theta)*sin(spinj.theta)* cos(spinj.phi) - sin(spini.theta)*cos(spini.phi)*cos(spinj.theta));
}


double diag_int(Spin &spini, Spin &spinj){
    return J*(sin(spini.theta) *sin(spinj.theta)*cos(spini.phi - spinj.phi) + cos(spini.theta) *cos(spinj.theta));
};

double zeeman(Spin &spini){
    return B*cos(spini.theta);
}

double all_int(Spin &spini, Lattice &lattice){
    double dE = 0 ;
    dE -= diag_int(spini, lattice.spins[spini.neighbours[0]]) - cross_y(spini, lattice.spins[spini.neighbours[0]]); // 
    dE -= diag_int(spini, lattice.spins[spini.neighbours[1]]) + cross_y(spini, lattice.spins[spini.neighbours[1]]);
    dE -= diag_int(spini, lattice.spins[spini.neighbours[2]]) - cross_x(spini, lattice.spins[spini.neighbours[2]]);
    dE -= diag_int(spini, lattice.spins[spini.neighbours[3]]) + cross_x(spini, lattice.spins[spini.neighbours[3]]);

    return dE;
};


void config_sim(int argc, char **argv){
    for (int i = 1; i < argc; i += 2){
        std::string arg = argv[i];
        //std::cout << argv[i] << std::endl;
        if (arg == "--J"){J = std::stod(argv[i+1]);}
        else if(arg == "--D"){D = std::stod(argv[i+1]);}
        else if(arg == "--T"){T = std::stod(argv[i+1]); beta = 1.0/(0.086*T);}
        else if(arg == "--B"){B = std::stod(argv[i+1]);}
        else if(arg == "--size"){size = std::stod(argv[i+1]);}
        else if(arg == "--steps"){steps = std::stod(argv[i+1]);}
        else if(arg == "--seed"){seed = std::stoi(argv[i+1]);}
        else if(arg == "--gif"){gif = (std::stoi(argv[i+1]) == 1) ? true: false;}
        else {std::cout<< "Argument not recognized "<< argv[i]<< std::endl;};

    }
};



void setup_sim(Lattice &lattice){
    int N = size*size;
    std::vector<float> random_thetas = randomArray(seed, N, -1.0f, 1.0f);
    std::vector<float> random_phis = randomArray(seed, N, 0.0f, 1.0f);
    for (size_t i= 0; i < lattice.spins.size(); i ++){
        lattice.spins[i].theta = acos(random_thetas[i]);
        lattice.spins[i].phi = random_phis[i] * 2.0 * M_PI;
        int row = i/size;
        int column = i%size;
        lattice.spins[i].neighbours[0] = ((row + 1)%size)*size + column;
        lattice.spins[i].neighbours[1] = ((row - 1 + size)%size)*size + column;
        lattice.spins[i].neighbours[2] = (row )*size + (column +1)%size;
        lattice.spins[i].neighbours[3] = (row )*size + (column -1 + size)%size;

    }

    for (size_t i= 0; i < lattice.spins.size(); i ++){
        lattice.spins[i].local_energy = all_int(lattice.spins[i], lattice) + zeeman(lattice.spins[i]);
    }
    lattice.calc_tot_energy();
};

void update_spin(Lattice &lattice, int j, float &random_theta, float &random_phi, float &random_e){
    Spin new_spin_i(acos(random_theta), random_phi * 2.0 * M_PI);
        
    Spin old_spin_i = lattice.spins[j];
    new_spin_i.neighbours = old_spin_i.neighbours;
    new_spin_i.local_energy = all_int(new_spin_i, lattice) + zeeman(new_spin_i);
    double dH =  new_spin_i.local_energy - old_spin_i.local_energy;

    if (dH < 0 || std::exp(-dH*beta) > random_e){
        lattice.spins[j] = new_spin_i;
        lattice.energy += dH;

        // Update neighbours' local energy
        for (int n : new_spin_i.neighbours){
            lattice.spins[n].local_energy = all_int(lattice.spins[n], lattice) + zeeman(lattice.spins[n]);
        }
    }
};

void iterate(Lattice &lattice){
    // One step consists of iterating through all the atoms. 
    for(int i = 0; i < steps ; i++){
        // iteration in chessboard pattern
        std::vector<float> random_thetas = randomArray(seed + i, size*size, -1.0f, 1.0f);
        std::vector<float> random_phis = randomArray(seed + i + steps, size*size, 0.0f, 1.0f);
        std::vector<float> random_e = randomArray(seed + i + 2*steps, size*size, 0.0f, 1.0f);

        // update all parity-0 sites: (row+col) % 2 == 0
        for (int idx = 0; idx < size*size; idx++){
            int row = idx / size;
            int col = idx % size;
            if ((row + col) % 2 == 0)
                update_spin(lattice, idx, random_thetas[idx], random_phis[idx], random_e[idx]);
        }
        // update all parity-1 sites: (row+col) % 2 == 1
        for (int idx = 0; idx < size*size; idx++){
            int row = idx / size;
            int col = idx % size;
            if ((row + col) % 2 == 1)
                update_spin(lattice, idx, random_thetas[idx], random_phis[idx], random_e[idx]);
        }
        if (gif && i%100 == 0){
            std::ofstream gifFile("frames.csv", std::ios_base::app);
            for (size_t i = 0; i < lattice.spins.size(); i++){
                gifFile << lattice.spins[i].theta << "," << lattice.spins[i].phi << "\n";
            }
        }
    }
};


int main(int argc, char **argv) {
    config_sim(argc, argv);
    if (gif){
        std::ofstream gifFile("frames.csv");
        // header consist of theta, phi and value of the gridSize
        gifFile << "theta,phi," << size << "\n";
        std::cout << "Creating GIF frames..." << std::endl;
    }
    auto begin = std::chrono::steady_clock::now();
    Lattice lattice(size);
    setup_sim(lattice);
    iterate(lattice);
    std::cout<< lattice.calc_tot_energy() <<std::endl;
    auto end = std::chrono::steady_clock::now();
    std::cout << (end - begin).count()/1000000000.0/(size*size*steps) << " sec/step serial" << std::endl; 
    std::cout << "Total time: " << (end - begin).count()/1000000000.0 << " sec" << std::endl;
    // save final frame
    if (gif){
        std::ofstream gifFile("frames.csv", std::ios_base::app);
        for (size_t i = 0; i < lattice.spins.size(); i++){
            gifFile << lattice.spins[i].theta << "," << lattice.spins[i].phi << "\n";
        }
        gifFile.close();
    }
    lattice.save_spins("test.csv");
    return 0;
}