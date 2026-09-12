#include <iostream>
#include <fstream>
#include <vector>

// Function to take a step in the SIR model
// state: vector of S, I, R
// beta: infection rate
// gamma: recovery rate
// dt: time step

std::vector<double> take_step(std::vector<double> state, double beta, double gamma, double dt, int N){

    std::vector<double> new_state;
    //todo: implement the SIR model

    double dSdt = -beta * state[1] * state[0]/N;
    double dIdt = beta * state[1] * state[0]/N - gamma * state[1];
    double dRdt = gamma * state[1];

    double susceptibles = state[0] + dSdt * dt;
    double infected = state[1] + dIdt * dt;
    double recovered = state[2] + dRdt * dt;

    new_state = {susceptibles, infected, recovered};
    return new_state;
}

//======================================================================================================
//======================== Main function ===============================================================
//======================================================================================================

void simulation(int N, double beta, double gamma, double T, double dt, std::string filename, int initial_infected, int sampling_dt = 1){
    
    // Initial state: S, I, R
    std::vector<double> state = {double(N) - double(initial_infected), double(initial_infected), 0};
    int current_time = 0;

    // Open a file
    std::ofstream outFile(filename);
    // Create header
    outFile << "Time,Susceptible,Infected,Recovered\n";

    while (current_time < T){
        outFile << current_time << "," 
                    << state[0] << "," 
                    << state[1] << "," 
                    << state[2] << "\n";

        for (int step = 0; step < int(sampling_dt/dt); step++)
            state = take_step(state, beta, gamma, dt, N);

        current_time += sampling_dt;
    }
    
    // Close the file
    outFile.close();
}


int main(int argc, char* argv[]){
    //Arguments: Initial infected, dt_1, dt_2, ..., dt_n
    int initial_infected = std::stoi(argv[1]);
    int sampling_dt = 1;

    // TODO: Define the parameters of the SIR model
    int N = 1000; // Total population
    double beta = 0.2; // Infection rate
    double gamma = 0.1; // Recovery rate
    double T = 200; // Total time

    for (int i = 2; i < argc; i++){
        double current_dt = std::stod(argv[i]);
        std::string filename = "simulation_dt_" + std::to_string(current_dt) + ".csv";
        simulation(N, beta, gamma, T, current_dt, filename, initial_infected, sampling_dt);
    }
    return 0;
}
