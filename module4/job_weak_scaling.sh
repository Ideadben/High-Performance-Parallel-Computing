#!/usr/bin/env bash
#SBATCH --job-name=Seismogram
#SBATCH --partition=modi_HPPC
#SBATCH --nodes=1 --ntasks=1 --threads-per-core=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive

# set loop scheduling to static
export OMP_SCHEDULE=static

# Schedule one thread per core. Change to "threads" for hyperthreading
export OMP_PLACES=cores
#export OMP_PLACES=threads

# Place threads as close to each other as possible
export OMP_PROC_BIND=close

# Container path
CONTAINER=~/modi_images/ucphhpc/hpc-notebook:latest

# Loop through powers of 2 from 1 to 64 (change to from 2 to 128 if using hyperthreading)
for OMP_NUM_THREADS in 1 2 4 8 16 32 64; do
    echo
    echo "=============================================="
    echo " Running test with OMP_NUM_THREADS = $OMP_NUM_THREADS"
    echo "=============================================="

    export OMP_NUM_THREADS

    # Compute NFREQ for this number of threads
    nfreq=$(( 16384 * OMP_NUM_THREADS ))
    echo "NFREQ = $nfreq"

    # Clean + Build inside container
    apptainer exec "$CONTAINER" make clean
    apptainer exec "$CONTAINER" make NFREQ="$nfreq" mp

    # Run the program 5 times to get error bars on the scaling plot
    apptainer exec "$CONTAINER" ./mp
    apptainer exec "$CONTAINER" ./mp
    apptainer exec "$CONTAINER" ./mp
    apptainer exec "$CONTAINER" ./mp
    apptainer exec "$CONTAINER" ./mp
done

# Set and print number of threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo Number of threads=$OMP_NUM_THREADS

# uncomment to write info about binding and environment variables to screen
#export OMP_DISPLAY_ENV=true

# Compute NFREQ to scale with number of cores
nfreq=$(( 16384 * OMP_NUM_THREADS ))
echo "NFREQ = $nfreq"

# Run inside the Apptainer container
CONTAINER=~/modi_images/ucphhpc/hpc-notebook:latest

apptainer exec "$CONTAINER" make clean
apptainer exec "$CONTAINER" make NFREQ="$nfreq" mp
apptainer exec "$CONTAINER" ./mp
