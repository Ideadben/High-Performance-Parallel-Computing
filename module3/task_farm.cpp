/*
  Assignment: Make an MPI task farm. A "task" is a randomly generated integer.
  To "execute" a task, the worker sleeps for the given number of milliseconds.
  The result of a task should be send back from the worker to the master. It
  contains the rank of the worker
*/

#include <iostream>
#include <random>
#include <chrono>
#include <thread>
#include <array>

// To run an MPI program we always need to include the MPI headers
#include <mpi.h>

const int NTASKS=5000;  // number of tasks
const int RANDOM_SEED=1234;

void master (int nworker) {
    std::array<int, NTASKS> task, result;

    // set up a random number generator
    std::random_device rd;
    //std::default_random_engine engine(rd());
    std::default_random_engine engine;
    engine.seed(RANDOM_SEED);
    // make a distribution of random integers in the interval [0:30]
    std::uniform_int_distribution<int> distribution(0, 30);

    for (int& t : task) {
        t = distribution(engine);   // set up some "tasks"
    }

    /*
    IMPLEMENT HERE THE CODE FOR THE MASTER
    ARRAY task contains tasks to be done. Send one element at a time to workers
    ARRAY result should at completion contain the ranks of the workers that did
    the corresponding tasks
    */
    int complete;
    int next_task = 0;
    int next_result = 0;
    MPI_Status status;

    //Send out initial tasks to workers
    for (int worker = 1; worker <= nworker; worker++){
        MPI_Send(&task[next_task], 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
        next_task++;
    }

    //Keep sending out tasks
    while(next_task < NTASKS){

        //Receive result
        MPI_Recv(&complete, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

        //Determine worker
        int worker = status.MPI_SOURCE;
        result[next_result] = worker;
        next_result++;

        //Send out new task
        MPI_Send(&task[next_task], 1, MPI_INT, worker, 0, MPI_COMM_WORLD);

        //Move to new task
        next_task++;
    }

    //Receive final task
    for (int i = 0; i < nworker; i++) {
        MPI_Recv(&complete, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        result[next_result] = status.MPI_SOURCE; 
        next_result++;
    }

    //Kill workers
    int kill_val = -1;
    for (int worker = 1; worker <= nworker; worker++){
        MPI_Send(&kill_val, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
    }
    


    // Print out a status on how many tasks were completed by each worker
    for (int worker=1; worker<=nworker; worker++) {
        int tasksdone = 0; int workdone = 0;
        for (int itask=0; itask<NTASKS; itask++)
        if (result[itask]==worker) {
            tasksdone++;
            workdone += task[itask];
        }
        std::cout << "Master: Worker " << worker << " solved " << tasksdone << 
                    " tasks\n";    
    }
}

// call this function to complete the task. It sleeps for task milliseconds
void task_function(int task) {
    std::this_thread::sleep_for(std::chrono::milliseconds(task));
}

void worker (int rank) {
    /*
    IMPLEMENT HERE THE CODE FOR THE WORKER
    Use a call to "task_function" to complete a task
    */
    while (true) {
        int task;

        MPI_Recv(&task, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (task == -1) {
            break;
        }

        task_function(task);

        MPI_Send(&rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char *argv[]) {
    int nrank, rank;

    MPI_Init(&argc, &argv);                // set up MPI
    MPI_Comm_size(MPI_COMM_WORLD, &nrank); // get the total number of ranks
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // get the rank of this process

    if (rank == 0)       // rank 0 is the master
        master(nrank-1); // there is nrank-1 worker processes
    else                 // ranks in [1:nrank] are workers
        worker(rank);

    MPI_Finalize();      // shutdown MPI
    return 0;
}
