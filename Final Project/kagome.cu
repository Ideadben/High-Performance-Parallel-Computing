#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <fstream>

#include <cuda_runtime.h>
#include <curand_kernel.h>

static void check_cuda(cudaError_t code, const char* context) {
    if (code != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(code));
    }
}


#define PREC 4  // 4 for float, 8 for double

#if PREC == 4
    using real_t = float;
#elif PREC == 8
    using real_t = double;
#else
    using real_t = float;
#endif


double J = 1; //units of mev
double D = 1; //units of mev
double B = 0;
double T = 5; //units of kelvin
double beta = 1.0/(0.086*T);
int sizex = 10;
int sizey = 10;

int save_n_frames = -1;

int steps = 1000;


//For going between cells
__device__ inline int idx_wrap(int row, int col, int sizex, int sizey) {
    int r = (row + sizey) % sizey;
    int c = (col + sizex) % sizex;
    return r * sizex + c;
}

// We have vectors 12, 13, 23
// v13 = (1/2, sqrt(3)/2) = (0.5, 0.866)
// v12 = (1 , 0)
// v23 = (-1/2, sqrt(3)/2) = (0.5, 0.866)


// For subgrid 1
__device__ double local_energy_site_1(
    real_t xc,  real_t yc,  real_t zc,
    const real_t* __restrict__ x2,  const real_t* __restrict__ y2,  const real_t* __restrict__ z2,
    const real_t* __restrict__ x3,  const real_t* __restrict__ y3,  const real_t* __restrict__ z3,
    int row,
    int col,
    int sizex,
    int sizey,
    double j,
    double d,
    double b) {
    //int center = row * sizex + col;
    int parity =  row%2;
    int center = idx_wrap(row, col, sizex, sizey);
    int left = idx_wrap(row, col - 1, sizex, sizey);
    int down_left = idx_wrap(row + 1, col - parity , sizex, sizey);


    double dE = 0.0;
    dE -= j*(xc*x3[center]  + yc*y3[center]  + zc*z3[center])        + d * ( 0.5 * (yc*z3[center]  - zc*y3[center]) - 0.866*(zc*x3[center]  - xc*z3[center])); // 3 top right in same cell
    dE -= j*(xc*x2[center]    + yc*y2[center]    + zc*z2[center])    + d * (yc*z2[center]  - zc*y2[center]); // 2 right in same cell
    dE -= j*(xc*x2[left]  + yc*y2[left]  + zc*z2[left])              - d * (yc*z2[left]  - zc*y2[left]); // 2 left
    dE -= j*(xc*x3[down_left] + yc*y3[down_left] + zc*z3[down_left]) + d * ( -0.5 * (yc*z3[down_left]  - zc*y3[down_left]) + 0.866*(zc*x3[down_left]  - xc*z3[down_left])); // 3 bottom left
    dE += b*zc;
    return dE;
}

// For subgrid 2
__device__ double local_energy_site_2(
    real_t xc,  real_t yc,  real_t zc,
    const real_t* __restrict__ x1,  const real_t* __restrict__ y1,  const real_t* __restrict__ z1,
    const real_t* __restrict__ x3,  const real_t* __restrict__ y3,  const real_t* __restrict__ z3,
    int row,
    int col,
    int sizex,
    int sizey,
    double j,
    double d,
    double b) {
    //int center = row * sizex + col;
    int parity =  row%2;
    int center = idx_wrap(row, col, sizex, sizey);
    int right = idx_wrap(row, col + 1, sizex, sizey);
    int down_right = idx_wrap(row + 1, col + 1 - parity, sizex, sizey);

    double dE = 0.0;
    dE -= j*(xc*x1[center]  + yc*y1[center]  + zc*z1[center])     - d * (yc*z1[center] - zc*y1[center]); // 1 left same cell
    dE -= j*(xc*x3[center]    + yc*y3[center]    + zc*z3[center]) + d * ( - 0.5 * (yc*z3[center]  - zc*y3[center]) - 0.866*(zc*x3[center]  - xc*z3[center])); // 3 top left same cell
    dE -= j*(xc*x1[right]  + yc*y1[right]  + zc*z1[right])        +  d * (yc*z1[right] - zc*y1[right]); // 1 right
    dE -= j*(xc*x3[down_right] + yc*y3[down_right] + zc*z3[down_right]) + d * (0.5 * (yc*z3[down_right]  - zc*y3[down_right]) + 0.866*(zc*x3[down_right]  - xc*z3[down_right])); // 3 down right
    dE += b*zc;
    return dE;
}


// For subgrid 3
__device__ double local_energy_site_3(
    real_t xc,  real_t yc,  real_t zc,
    const real_t* __restrict__ x1,  const real_t* __restrict__ y1,  const real_t* __restrict__ z1,
    const real_t* __restrict__ x2,  const real_t* __restrict__ y2,  const real_t* __restrict__ z2,
    int row,
    int col,
    int sizex,
    int sizey,
    double j,
    double d,
    double b) {
    //int center = row * sizex + col;
    int parity =  row%2;
    int center = idx_wrap(row, col, sizex, sizey);
    int up_left = idx_wrap(row - 1, col - parity, sizex, sizey);
    int up_right = idx_wrap(row - 1, col + 1 - parity, sizex, sizey);


    double dE = 0.0;
    dE -= j*(xc*x1[center]  + yc*y1[center]  + zc*z1[center])     + d * ( - 0.5 * (yc*z1[center]  - zc*y1[center]) + 0.866*(zc*x1[center]  - xc*z1[center])); // 1 bottom left same cell
    dE -= j*(xc*x2[center]    + yc*y2[center]    + zc*z2[center]) + d * (0.5 * (yc*z2[center]  - zc*y2[center]) + 0.866*(zc*x2[center]  - xc*z2[center])); // 2 bottom right same cell
    dE -= j*(xc*x2[up_left]  + yc*y2[up_left]  + zc*z2[up_left])  + d * (-0.5 * (yc*z2[up_left]  - zc*y2[up_left]) - 0.866*(zc*x2[up_left]  - xc*z2[up_left])); // 2 top left
    dE -= j*(xc*x1[up_right] + yc*y1[up_right] + zc*z1[up_right]) + d * (0.5 * (yc*z1[up_right]  - zc*y1[up_right]) - 0.866*(zc*x1[up_right]  - xc*z1[up_right])); // 1 top right
    dE += b*zc;
    return dE;
    }

__global__ void init_rng_kernel(curandStatePhilox4_32_10_t* rng_states, unsigned long long seed, int n_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) {
        return;
    }
    curand_init(seed, idx, 0, &rng_states[idx]);
}





void config_sim(int argc, char **argv){
    for (int i = 1; i < argc; i += 2){
        std::string arg = argv[i];
        //std::cout << argv[i] << std::endl;
        if (arg == "--J"){J = std::stod(argv[i+1]);}
        else if(arg == "--D"){D = std::stod(argv[i+1]);}
        else if(arg == "--T"){T = std::stod(argv[i+1]); beta = 1.0/(0.086*T);}
        else if(arg == "--B"){B = std::stod(argv[i+1]);}
        else if(arg == "--sizex"){sizex = std::stod(argv[i+1]);}
        else if(arg == "--sizey"){sizey = std::stod(argv[i+1]);}
        else if(arg == "--steps"){steps = std::stod(argv[i+1]);}
        else if(arg == "--save_every"){save_n_frames = std::stod(argv[i+1]);}
        else if(arg == "--save_n_frames"){save_n_frames = std::stod(argv[i+1]);}
        else {std::cout<< "Argument not regonized "<< argv[i]<< std::endl;};

    }
};

__global__ void init_spins(real_t* x, real_t* y, real_t* z, curandStatePhilox4_32_10_t* rng_states, int n_cells){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) {
        return;
    }

    curandStatePhilox4_32_10_t state = rng_states[idx];
    double u = curand_uniform(&state);
    double v = curand_uniform(&state);
    double r = 2.0 * u - 1.0;

    double theta = acosf(r);
    double phi = v * 2.0 * M_PI;
    x[idx] = cosf(phi)*sinf(theta);
    y[idx] = sinf(phi)*sinf(theta);
    z[idx] = cosf(theta);
    rng_states[idx] = state;

}


__global__ void mc_step_1(
    real_t* x1, real_t* y1, real_t* z1, 
    const real_t* __restrict__  x2,const real_t* __restrict__  y2,const real_t* __restrict__  z2,
    const real_t* __restrict__  x3,const real_t* __restrict__  y3,const real_t* __restrict__  z3,
    int sizex,
    int sizey,
    double j,
    double d,
    double b,
    double beta,
    curandStatePhilox4_32_10_t* rng_states
    ){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = sizex * sizey;

    if (idx >= n_cells) {
        return;
    }

    curandStatePhilox4_32_10_t state = rng_states[idx];
    double u = curand_uniform(&state);
    double v = curand_uniform(&state);
    double r = 2.0 * u - 1.0;

    double theta = acosf(r);
    double phi = v * 2.0 * M_PI;
    real_t xp = cosf(phi)*sinf(theta);
    real_t yp = sinf(phi)*sinf(theta);
    real_t zp = cosf(theta);



    int row = idx / sizex;
    int col = idx % sizex;

    real_t local_energy1 = local_energy_site_1(x1[idx], y1[idx], z1[idx], x2, y2, z2, x3, y3, z3, row, col, sizex, sizey, j, d, b);
    real_t local_energyp = local_energy_site_1(xp, yp, zp, x2, y2, z2, x3, y3, z3, row, col, sizex, sizey, j, d, b);

    real_t dH = local_energyp - local_energy1;
    real_t mu = curand_uniform(&state);
    bool take = (dH < 0.0) || (expf(-dH * beta) > mu);
    if (take){
        x1[idx] = xp;
        y1[idx] = yp;
        z1[idx] = zp;
    }
    rng_states[idx] = state;

}

__global__ void mc_step_2(
    const real_t* __restrict__  x1, const real_t* __restrict__  y1, const real_t* __restrict__  z1, 
    real_t* x2,real_t* y2,real_t* z2,
    const real_t* __restrict__  x3, const real_t* __restrict__  y3, const real_t* __restrict__  z3, 
    int sizex,
    int sizey,
    double j,
    double d,
    double b,
    double beta,
    curandStatePhilox4_32_10_t* rng_states
    ){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = sizex * sizey;

    if (idx >= n_cells) {
        return;
    }

    curandStatePhilox4_32_10_t state = rng_states[idx];
    double u = curand_uniform(&state);
    double v = curand_uniform(&state);
    double r = 2.0 * u - 1.0;

    double theta = acosf(r);
    double phi = v * 2.0 * M_PI;
    real_t xp = cosf(phi)*sinf(theta);
    real_t yp = sinf(phi)*sinf(theta);
    real_t zp = cosf(theta);


    int row = idx / sizex;
    int col = idx % sizex;

    real_t local_energy1 = local_energy_site_2(x2[idx], y2[idx], z2[idx], x1, y1, z1, x3, y3, z3, row, col, sizex, sizey, j, d, b);
    real_t local_energyp = local_energy_site_2(xp, yp, zp, x1, y1, z1, x3, y3, z3, row, col, sizex, sizey, j, d, b);

    real_t dH = local_energyp - local_energy1;
    real_t mu = curand_uniform(&state);
    bool take = (dH < 0.0) || (expf(-dH * beta) > mu);
    if (take){
        x2[idx] = xp;
        y2[idx] = yp;
        z2[idx] = zp;
    }
    rng_states[idx] = state;

}

__global__ void mc_step_3(
    const real_t* __restrict__  x1, const real_t* __restrict__  y1, const real_t* __restrict__  z1, 
    const real_t* __restrict__  x2, const real_t* __restrict__  y2, const real_t* __restrict__  z2, 
    real_t* x3,real_t* y3,real_t* z3,
    int sizex,
    int sizey,
    double j,
    double d,
    double b,
    double beta,
    curandStatePhilox4_32_10_t* rng_states
    ){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = sizex * sizey;

    if (idx >= n_cells) {
        return;
    }

    curandStatePhilox4_32_10_t state = rng_states[idx];
    double u = curand_uniform(&state);
    double v = curand_uniform(&state);
    double r = 2.0 * u - 1.0;

    double theta = acosf(r);
    double phi = v * 2.0 * M_PI;
    real_t xp = cosf(phi)*sinf(theta);
    real_t yp = sinf(phi)*sinf(theta);
    real_t zp = cosf(theta);


    int row = idx / sizex;
    int col = idx % sizex;

    real_t local_energy1 = local_energy_site_3(x3[idx], y3[idx], z3[idx], x1, y1, z1, x2, y2, z2, row, col, sizex, sizey, j, d, b);
    real_t local_energyp = local_energy_site_3(xp, yp, zp, x1, y1, z1, x2, y2, z2, row, col, sizex, sizey, j, d, b);

    real_t dH = local_energyp - local_energy1;
    real_t mu = curand_uniform(&state);
    bool take = (dH < 0.0) || (expf(-dH * beta) > mu);
    if (take){
        x3[idx] = xp;
        y3[idx] = yp;
        z3[idx] = zp;
    }
    rng_states[idx] = state;

}


__global__ void energy_site_1(
    const real_t* __restrict__ x1, const real_t* __restrict__ y1, const real_t* __restrict__ z1,
    const real_t* __restrict__ x2, const real_t* __restrict__ y2, const real_t* __restrict__ z2,
    const real_t* __restrict__ x3, const real_t* __restrict__ y3, const real_t* __restrict__ z3,
    int sx, int sy, double j, double d, double b, double* __restrict__ out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sx*sy) return;
    int row = idx/sx, col = idx%sx;
    double e = local_energy_site_1(x1[idx],y1[idx],z1[idx],x2,y2,z2,x3,y3,z3,row,col,sx,sy,j,d,b);
    double ze = b*(double)z1[idx];
    out[idx] = 0.5*(e-ze)+ze;
}

__global__ void energy_site_2(
    const real_t* __restrict__ x1, const real_t* __restrict__ y1, const real_t* __restrict__ z1,
    const real_t* __restrict__ x2, const real_t* __restrict__ y2, const real_t* __restrict__ z2,
    const real_t* __restrict__ x3, const real_t* __restrict__ y3, const real_t* __restrict__ z3,
    int sx, int sy, double j, double d, double b, double* __restrict__ out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sx*sy) return;
    int row = idx/sx, col = idx%sx;
    double e = local_energy_site_2(x2[idx],y2[idx],z2[idx],x1,y1,z1,x3,y3,z3,row,col,sx,sy,j,d,b);
    double ze = b*(double)z2[idx];
    out[idx] = 0.5*(e-ze)+ze;
}

__global__ void energy_site_3(
    const real_t* __restrict__ x1, const real_t* __restrict__ y1, const real_t* __restrict__ z1,
    const real_t* __restrict__ x2, const real_t* __restrict__ y2, const real_t* __restrict__ z2,
    const real_t* __restrict__ x3, const real_t* __restrict__ y3, const real_t* __restrict__ z3,
    int sx, int sy, double j, double d, double b, double* __restrict__ out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sx*sy) return;
    int row = idx/sx, col = idx%sx;
    double e = local_energy_site_3(x3[idx],y3[idx],z3[idx],x1,y1,z1,x2,y2,z2,row,col,sx,sy,j,d,b);
    double ze = b*(double)z3[idx];
    out[idx] = 0.5*(e-ze)+ze;
}

double compute_total_energy(
    real_t* d_x1, real_t* d_y1, real_t* d_z1,
    real_t* d_x2, real_t* d_y2, real_t* d_z2,
    real_t* d_x3, real_t* d_y3, real_t* d_z3,
    int n_cells, int threads, int blocks)
{
    double *d_e1, *d_e2, *d_e3;
    check_cuda(cudaMalloc(&d_e1,n_cells*sizeof(double)),"me1");
    check_cuda(cudaMalloc(&d_e2,n_cells*sizeof(double)),"me2");
    check_cuda(cudaMalloc(&d_e3,n_cells*sizeof(double)),"me3");
    energy_site_1<<<blocks,threads>>>(d_x1,d_y1,d_z1,d_x2,d_y2,d_z2,d_x3,d_y3,d_z3,sizex,sizey,J,D,B,d_e1);
    energy_site_2<<<blocks,threads>>>(d_x1,d_y1,d_z1,d_x2,d_y2,d_z2,d_x3,d_y3,d_z3,sizex,sizey,J,D,B,d_e2);
    energy_site_3<<<blocks,threads>>>(d_x1,d_y1,d_z1,d_x2,d_y2,d_z2,d_x3,d_y3,d_z3,sizex,sizey,J,D,B,d_e3);
    check_cuda(cudaDeviceSynchronize(),"esync");
    std::vector<double> h_e1(n_cells),h_e2(n_cells),h_e3(n_cells);
    check_cuda(cudaMemcpy(h_e1.data(),d_e1,n_cells*sizeof(double),cudaMemcpyDeviceToHost),"ce1");
    check_cuda(cudaMemcpy(h_e2.data(),d_e2,n_cells*sizeof(double),cudaMemcpyDeviceToHost),"ce2");
    check_cuda(cudaMemcpy(h_e3.data(),d_e3,n_cells*sizeof(double),cudaMemcpyDeviceToHost),"ce3");
    cudaFree(d_e1); cudaFree(d_e2); cudaFree(d_e3);
    double total = 0.0;
    for (int i = 0; i < n_cells; i++) total += h_e1[i]+h_e2[i]+h_e3[i];
    return total;
}


int main(int argc, char **argv) {
    auto tstart = std::chrono::high_resolution_clock::now();
    config_sim(argc, argv);

    int n_cells = sizex * sizey;
    int threads = 256;
    int blocks = (n_cells + threads - 1) / threads;

    real_t* d_x1 = nullptr; real_t* d_y1 = nullptr; real_t* d_z1 = nullptr;

    real_t* d_x2 = nullptr; real_t* d_y2 = nullptr; real_t* d_z2 = nullptr;

    real_t* d_x3 = nullptr; real_t* d_y3 = nullptr; real_t* d_z3 = nullptr;

    curandStatePhilox4_32_10_t* d_rng_states = nullptr;


    check_cuda(cudaMalloc(&d_x1, n_cells * sizeof(real_t)), "cudaMalloc d_x1");
    check_cuda(cudaMalloc(&d_y1, n_cells * sizeof(real_t)), "cudaMalloc d_y1");
    check_cuda(cudaMalloc(&d_z1, n_cells * sizeof(real_t)), "cudaMalloc d_z1");

    check_cuda(cudaMalloc(&d_x2, n_cells * sizeof(real_t)), "cudaMalloc d_x2");
    check_cuda(cudaMalloc(&d_y2, n_cells * sizeof(real_t)), "cudaMalloc d_y2");
    check_cuda(cudaMalloc(&d_z2, n_cells * sizeof(real_t)), "cudaMalloc d_z2");

    check_cuda(cudaMalloc(&d_x3, n_cells * sizeof(real_t)), "cudaMalloc d_x3");
    check_cuda(cudaMalloc(&d_y3, n_cells * sizeof(real_t)), "cudaMalloc d_y3");
    check_cuda(cudaMalloc(&d_z3, n_cells * sizeof(real_t)), "cudaMalloc d_z3");

    check_cuda(cudaMalloc(&d_rng_states, n_cells * sizeof(curandStatePhilox4_32_10_t)), "cudaMalloc d_rng_states");

    init_rng_kernel<<<blocks, threads>>>(d_rng_states, 1234567ULL, n_cells);
    check_cuda(cudaGetLastError(), "init_rng_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "init_rng_kernel sync");

    init_spins<<<blocks, threads>>>(d_x1, d_y1, d_z1, d_rng_states, n_cells);
    init_spins<<<blocks, threads>>>(d_x2, d_y2, d_z2, d_rng_states, n_cells);
    init_spins<<<blocks, threads>>>(d_x3, d_y3, d_z3, d_rng_states, n_cells);
    check_cuda(cudaGetLastError(), "init_spins_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "init_spins_kernel sync");

    std::ofstream elog("data/energy.csv");
    elog << "step,energy,energy_per_site\n";
    double e0 = compute_total_energy(d_x1,d_y1,d_z1,d_x2,d_y2,d_z2,d_x3,d_y3,d_z3,n_cells,threads,blocks);
    elog << 0 << "," << e0 << "," << e0/(3.0*n_cells) << "\n";


    for (size_t i = 0; i< steps; i++){
        mc_step_1<<<blocks, threads>>>(d_x1, d_y1, d_z1, d_x2, d_y2, d_z2, d_x3, d_y3, d_z3, sizex, sizey, J, D, B, beta , d_rng_states);
        mc_step_2<<<blocks, threads>>>(d_x1, d_y1, d_z1, d_x2, d_y2, d_z2, d_x3, d_y3, d_z3,  sizex, sizey, J, D, B, beta, d_rng_states);
        mc_step_3<<<blocks, threads>>>(d_x1, d_y1, d_z1, d_x2, d_y2, d_z2, d_x3, d_y3, d_z3,  sizex, sizey, J, D, B, beta, d_rng_states);
        if (save_n_frames > 0 && (i+1) % save_n_frames == 0) {
            double e = compute_total_energy(d_x1,d_y1,d_z1,d_x2,d_y2,d_z2,d_x3,d_y3,d_z3,n_cells,threads,blocks);
            elog << (i+1) << "," << e << "," << e/(3.0*n_cells) << "\n";
        }
    }
    // Copy data back to host
    std::vector<real_t> h_x1(n_cells), h_y1(n_cells), h_z1(n_cells);
    std::vector<real_t> h_x2(n_cells), h_y2(n_cells), h_z2(n_cells);
    std::vector<real_t> h_x3(n_cells), h_y3(n_cells), h_z3(n_cells);

    check_cuda(cudaMemcpy(h_x1.data(), d_x1, n_cells*sizeof(real_t), cudaMemcpyDeviceToHost), "memcpy x1");
    check_cuda(cudaMemcpy(h_y1.data(), d_y1, n_cells*sizeof(real_t), cudaMemcpyDeviceToHost), "memcpy y1");
    check_cuda(cudaMemcpy(h_z1.data(), d_z1, n_cells*sizeof(real_t), cudaMemcpyDeviceToHost), "memcpy z1");
    check_cuda(cudaMemcpy(h_x2.data(), d_x2, n_cells*sizeof(real_t), cudaMemcpyDeviceToHost), "memcpy x2");
    check_cuda(cudaMemcpy(h_y2.data(), d_y2, n_cells*sizeof(real_t), cudaMemcpyDeviceToHost), "memcpy y2");
    check_cuda(cudaMemcpy(h_z2.data(), d_z2, n_cells*sizeof(real_t), cudaMemcpyDeviceToHost), "memcpy z2");
    check_cuda(cudaMemcpy(h_x3.data(), d_x3, n_cells*sizeof(real_t), cudaMemcpyDeviceToHost), "memcpy x3");
    check_cuda(cudaMemcpy(h_y3.data(), d_y3, n_cells*sizeof(real_t), cudaMemcpyDeviceToHost), "memcpy y3");
    check_cuda(cudaMemcpy(h_z3.data(), d_z3, n_cells*sizeof(real_t), cudaMemcpyDeviceToHost), "memcpy z3");

    // Save subgrid 1
    auto save_subgrid = [&](const std::string& filename,
                            const std::vector<real_t>& x, 
                            const std::vector<real_t>& y,
                            const std::vector<real_t>& z) {
        std::ofstream f(filename);
        f << "index,row,col,sx,sy,sz\n";
        for (int i = 0; i < n_cells; i++){
            int row = i / sizex;
            int col = i % sizex;
            f << i << "," << row << "," << col << ","
            << x[i] << "," << y[i] << "," << z[i] << "\n";
        }
    };

    auto tend = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(tend - tstart).count();
    std::cout <<  elapsed  <<std::endl;
    save_subgrid("data/subgrid1.csv", h_x1, h_y1, h_z1);
    save_subgrid("data/subgrid2.csv", h_x2, h_y2, h_z2);
    save_subgrid("data/subgrid3.csv", h_x3, h_y3, h_z3);

     // Save simulation parameters and results
    std::ofstream params("data/params.txt");
    params << "J="           << J            << "\n"
        << "D="           << D            << "\n"
        << "B="           << B            << "\n"
        << "T="           << T            << "\n"
        << "beta="        << beta         << "\n"
        << "sizex="       << sizex        << "\n"
        << "sizey="       << sizey        << "\n"
        << "steps="       << steps        << "\n"
        << "elapsed_s="   << elapsed      << "\n";


    
    cudaFree(d_x1); cudaFree(d_y1); cudaFree(d_z1);
    cudaFree(d_x2); cudaFree(d_y2); cudaFree(d_z2);
    cudaFree(d_x3); cudaFree(d_y3); cudaFree(d_z3);
    cudaFree(d_rng_states);
    return 0;
}