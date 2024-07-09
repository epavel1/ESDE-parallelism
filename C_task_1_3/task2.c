#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
/**This segment will need to be filled by the participants.**/
    //TODO:  

    if (world_rank == 0) {
        printf("Global sum: %f\n", global_sum);
    }

    MPI_Finalize();
    return 0;
}

