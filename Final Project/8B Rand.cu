#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// ---- Simulation parameters ----
static double J = 1.0;
static double D = 1.0;
static double B = 0.0;
static double T = 5.0;
static double beta = 1.0 / (0.086 * T);
static int size = 10;
static int steps = 1000;
static int seed = 0;
static bool saveframe = false;
static std::string outfile = "timing_results_float_8B_2.csv";

static void check_cuda(cudaError_t code, const char* context) {
    if (code != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", context, cudaGetErrorString(code));
        exit(EXIT_FAILURE);
    }
}

#define M_PI_F 3.14159265f

// ---- PCG32 - 8-byte state, high-quality 32-bit output ----
// State: one uint64 per site (8 bytes vs 64 bytes for Philox).
struct Pcg32 { unsigned long long state; };

__host__ __device__ inline unsigned int pcg32_next(Pcg32& rng) {
    unsigned long long old = rng.state;
    rng.state = old * 6364136223846793005ULL + 1442695040888963407ULL;
    unsigned int xorshifted = (unsigned int)(((old >> 18u) ^ old) >> 27u);
    unsigned int rot         = (unsigned int)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

__host__ __device__ inline Pcg32 pcg32_seed(unsigned long long seed, unsigned long long idx) {
    unsigned long long s = seed + idx * 6364136223846793005ULL;
    s = (s ^ (s >> 30)) * 0xbf58476d1ce4e5b9ULL;
    s = (s ^ (s >> 27)) * 0x94d049bb133111ebULL;
    s =  s ^ (s >> 31);
    Pcg32 rng; rng.state = s;
    pcg32_next(rng); pcg32_next(rng);
    return rng;
}

__host__ __device__ inline float pcg32_float(Pcg32& rng) {
    return (pcg32_next(rng) >> 8) * (1.0f / 16777216.0f);
}


// Fast periodic neighbour index: avoids modulo by using conditional add.
// Valid when row/col are already in [0, n-1].
__device__ inline int nbr(int base, int delta, int n) {
    int v = base + delta;
    if (v < 0) v += n;
    if (v >= n) v -= n;
    return v;
}

// ---- Initialize PCG32 states ----
__global__ void init_rng_kernel(Pcg32* rng_states, unsigned long long rng_seed, int n_sites) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_sites) return;
    rng_states[idx] = pcg32_seed(rng_seed, (unsigned long long)idx);
}

// ---- Initialize spins - float intrinsics ----
__global__ void init_spins_kernel(
    float* x, float* y, float* z,
    Pcg32* rng_states, int n_sites) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_sites) return;

    Pcg32 state = rng_states[idx];
    float r0 = pcg32_float(state);
    float r1 = pcg32_float(state);
    rng_states[idx] = state;

    float pz = 2.0f * r0 - 1.0f;
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - pz * pz));
    float sphi, cphi;
    __sincosf(r1 * 2.0f * M_PI_F, &sphi, &cphi);

    x[idx] = cphi * sin_theta;
    y[idx] = sphi * sin_theta;
    z[idx] = pz;
}

// ---- MC kernel - compute-optimised ----
__global__ void mc_sweep_kernel(
    float* x, float* y, float* z,
    Pcg32* rng_states,
    int n,
    int parity,
    float j, float d, float b,
    float inv_temperature) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_half = n * (n / 2);
    if (tid >= n_half) return;

    int row = tid / (n / 2);
    int half_col = tid % (n / 2);
    int col = 2 * half_col + ((row & 1) ^ parity);
    int site = row * n + col;

    Pcg32 state = rng_states[site];
    float r0 = pcg32_float(state);
    float r1 = pcg32_float(state);
    float r2 = pcg32_float(state);
    rng_states[site] = state;

    float pz = 2.0f * r0 - 1.0f;
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - pz * pz));
    float sphi, cphi;
    __sincosf(r1 * 2.0f * M_PI_F, &sphi, &cphi);
    float px = cphi * sin_theta;
    float py = sphi * sin_theta;
    float accept_r = r2;

    float xi = x[site], yi = y[site], zi = z[site];

    int row_p = nbr(row, +1, n), row_m = nbr(row, -1, n);
    int col_p = nbr(col, +1, n), col_m = nbr(col, -1, n);

    float xn0 = x[row_p*n+col], yn0 = y[row_p*n+col], zn0 = z[row_p*n+col];
    float xn1 = x[row_m*n+col], yn1 = y[row_m*n+col], zn1 = z[row_m*n+col];
    float xn2 = x[row*n+col_p], yn2 = y[row*n+col_p], zn2 = z[row*n+col_p];
    float xn3 = x[row*n+col_m], yn3 = y[row*n+col_m], zn3 = z[row*n+col_m];

    float dx = px - xi, dy = py - yi, dz = pz - zi;
    float delta_h =
        - j*(dx*xn0 + dy*yn0 + dz*zn0) + d*(dz*xn0 - dx*zn0)
        - j*(dx*xn1 + dy*yn1 + dz*zn1) - d*(dz*xn1 - dx*zn1)
        - j*(dx*xn2 + dy*yn2 + dz*zn2) + d*(dy*zn2 - dz*yn2)
        - j*(dx*xn3 + dy*yn3 + dz*zn3) - d*(dy*zn3 - dz*yn3)
        + b * dz;

    bool take = (delta_h < 0.0f) || (__expf(-delta_h * inv_temperature) > accept_r);

    if (take) {
        x[site] = px;
        y[site] = py;
        z[site] = pz;
    }
}

static void config_sim(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string arg = argv[i];
        if (arg == "--J") {
            J = std::stod(argv[i + 1]);
        } else if (arg == "--D") {
            D = std::stod(argv[i + 1]);
        } else if (arg == "--T") {
            T = std::stod(argv[i + 1]);
            beta = 1.0 / (0.086 * T);
        } else if (arg == "--B") {
            B = std::stod(argv[i + 1]);
        } else if (arg == "--size") {
            size = std::stoi(argv[i + 1]);
        } else if (arg == "--steps") {
            steps = std::stoi(argv[i + 1]);
        } else if (arg == "--seed") {
            seed = std::stoi(argv[i + 1]);
        } else if (arg == "--saveframe") {
            saveframe = (std::stoi(argv[i + 1]) == 1) ? true : false;
        } else if (arg == "--outfile") {
            outfile = argv[i + 1];
        } else {
            std::cout << "Argument not recognized " << arg << std::endl;
        }
    }
}

static void save_spins(const std::vector<float>& x, const std::vector<float>& y, const std::vector<float>& z, int n, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file " << filename << std::endl;
        return;
    }
    file << "index,row,col,sx,sy,sz\n";
    for (int i = 0; i < n * n; ++i) {
        int row = i / n;
        int col = i % n;
        file << i << "," << row << "," << col << ","
             << x[i] << "," << y[i] << "," << z[i] << "\n";
    }
}

static void save_frame(float* d_x, float* d_y, float* d_z, int n_sites, int frame) {
    std::vector<float> h_x(n_sites), h_y(n_sites), h_z(n_sites);

    check_cuda(cudaMemcpy(h_x.data(), d_x, n_sites * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H x");
    check_cuda(cudaMemcpy(h_y.data(), d_y, n_sites * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H y");
    check_cuda(cudaMemcpy(h_z.data(), d_z, n_sites * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H z");

    double total_energy = 0.0;
    for (int i = 0; i < n_sites; ++i) {
        int row = i / size;
        int col = i % size;

        int n0 = ((row + 1 + size) % size) * size + col;
        int n1 = ((row - 1 + size) % size) * size + col;
        int n2 = row * size + ((col + 1 + size) % size);
        int n3 = row * size + ((col - 1 + size) % size);

        double xi = h_x[i], yi = h_y[i], zi = h_z[i];
        double e = 0.0;
        e -= J * (xi*h_x[n0] + yi*h_y[n0] + zi*h_z[n0]) - D * (zi*h_x[n0] - xi*h_z[n0]);
        e -= J * (xi*h_x[n1] + yi*h_y[n1] + zi*h_z[n1]) + D * (zi*h_x[n1] - xi*h_z[n1]);
        e -= J * (xi*h_x[n2] + yi*h_y[n2] + zi*h_z[n2]) - D * (yi*h_z[n2] - zi*h_y[n2]);
        e -= J * (xi*h_x[n3] + yi*h_y[n3] + zi*h_z[n3]) + D * (yi*h_z[n3] - zi*h_y[n3]);
        total_energy += 0.5 * e + B * zi;
    }

    if (saveframe) {
        std::string fileName = "frames/test_cuda" + std::to_string(frame) + ".csv";
        save_spins(h_x, h_y, h_z, size, fileName);
    }
}

// ---- Append one timing row to the output CSV ----
// Format: size, time_per_step_sec
// Creates the file with a header if it doesn't exist yet.
static void append_timing(const std::string& filename, int lattice_size, double sec_per_step) {
    // Check whether the file already exists (to decide whether to write a header)
    bool write_header = false;
    {
        std::ifstream test(filename);
        write_header = !test.good();
    }

    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: could not open timing file " << filename << std::endl;
        return;
    }

    if (write_header) {
        file << "size,sec_per_step\n";
    }
    file << lattice_size << "," << sec_per_step << "\n";
}

int main(int argc, char** argv) {
    auto begin = std::chrono::steady_clock::now();
    config_sim(argc, argv);

    int n_sites = size * size;
    int n_half = size * (size / 2);
    int threads = 256;
    int blocks_all = (n_sites + threads - 1) / threads;
    int blocks_half = (n_half + threads - 1) / threads;
    int frame = 0;

    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_z = nullptr;
    Pcg32* d_rng_states = nullptr;

    check_cuda(cudaMalloc(&d_x, n_sites * sizeof(float)), "cudaMalloc d_x");
    check_cuda(cudaMalloc(&d_y, n_sites * sizeof(float)), "cudaMalloc d_y");
    check_cuda(cudaMalloc(&d_z, n_sites * sizeof(float)), "cudaMalloc d_z");
    check_cuda(cudaMalloc(&d_rng_states, n_sites * sizeof(Pcg32)), "cudaMalloc d_rng_states");

    init_rng_kernel<<<blocks_all, threads>>>(d_rng_states, (unsigned long long)seed, n_sites);
    check_cuda(cudaGetLastError(), "init_rng_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "init_rng_kernel sync");

    init_spins_kernel<<<blocks_all, threads>>>(d_x, d_y, d_z, d_rng_states, n_sites);
    check_cuda(cudaGetLastError(), "init_spins_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "init_spins_kernel sync");
    
    for (int sweep = 0; sweep < steps; ++sweep) {
        for (int parity = 0; parity < 2; ++parity) {
            mc_sweep_kernel<<<blocks_half, threads>>>(
                d_x, d_y, d_z,
                d_rng_states,
                size, parity,
                (float)J, (float)D, (float)B, (float)beta);
        }

        if (sweep % 1000 == 0 && saveframe == 1) {
            std::cout << "printing frame..." << std::endl;
            frame = sweep / 1000;
            save_frame(d_x, d_y, d_z, n_sites, frame);
        }
    }
    
    std::vector<float> h_x(n_sites), h_y(n_sites), h_z(n_sites);

    check_cuda(cudaMemcpy(h_x.data(), d_x, n_sites * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H x");
    check_cuda(cudaMemcpy(h_y.data(), d_y, n_sites * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H y");
    check_cuda(cudaMemcpy(h_z.data(), d_z, n_sites * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H z");

    double total_energy = 0.0;
    for (int i = 0; i < n_sites; ++i) {
        int row = i / size;
        int col = i % size;

        int n0 = ((row + 1 + size) % size) * size + col;
        int n1 = ((row - 1 + size) % size) * size + col;
        int n2 = row * size + ((col + 1 + size) % size);
        int n3 = row * size + ((col - 1 + size) % size);

        double xi = h_x[i], yi = h_y[i], zi = h_z[i];
        double e = 0.0;
        e -= J * (xi*h_x[n0] + yi*h_y[n0] + zi*h_z[n0]) - D * (zi*h_x[n0] - xi*h_z[n0]);
        e -= J * (xi*h_x[n1] + yi*h_y[n1] + zi*h_z[n1]) + D * (zi*h_x[n1] - xi*h_z[n1]);
        e -= J * (xi*h_x[n2] + yi*h_y[n2] + zi*h_z[n2]) - D * (yi*h_z[n2] - zi*h_y[n2]);
        e -= J * (xi*h_x[n3] + yi*h_y[n3] + zi*h_z[n3]) + D * (yi*h_z[n3] - zi*h_y[n3]);
        total_energy += 0.5 * e + B * zi;
    }

    auto end = std::chrono::steady_clock::now();
    double n_ops = (double)size * size * steps;
    double sec_per_step = (end - begin).count() / 1000000000.0 / n_ops;

    // Print to stdout as before
    std::cout << sec_per_step << " sec/step" << std::endl;

    // Append size + timing to CSV
    append_timing(outfile, size, sec_per_step);

    if (saveframe == 1) {
        save_spins(h_x, h_y, h_z, size, "frames/test_cuda.csv");
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_rng_states);
    return 0;
}
