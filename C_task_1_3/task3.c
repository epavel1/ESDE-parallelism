#include <mpi.h>
#include <omp.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    double global_sum = 0.0;

 /**This segment will need to be filled by the participants.**/
   // TODO: Scatter the array sum and gather it back in final_sum
        
    if (world_rank == 0) {
        printf("Final sum: %f\n", final_sum);
      }

    MPI_Finalize();
    return 0;
}
