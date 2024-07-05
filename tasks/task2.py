from mpi4py import MPI
import timeit
 
def main(rank, size):
    array_length = 10000000
    local_partial_array = array_length // size
    
    ## These two lines will be removed
    # TODO: sum partial arrays HERE and return total_sum with total sum
    local_sum = sum(i * 0.001 for i in range(rank * local_partial_array, (rank + 1) * local_partial_array))
 
    total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
    ## Remove until HERE
 
    return total_sum
        
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()  
    if rank == 0:
        start_time = timeit.default_timer()
    total_sum = main(rank, size)
    if rank == 0:
        execution_time = (timeit.default_timer() - start_time)
        print(f"Sum of array: {total_sum}")
        print(f"Execution time: {execution_time}")
