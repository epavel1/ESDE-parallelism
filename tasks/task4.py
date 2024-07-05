import cupy
import mpi4py.MPI as MPI
from numba import cuda
import timeit

TPB = 256

@cuda.jit
def sum_values(v, a):
    """Sum all values in v.
    Parameters
    ----------
    v: numpy.ndarray
        One-dimensional array with values to be summed.
    a: numpy.ndarray
        One-element array to store the sum.
    """
    i = cuda.grid(1)
    if i < v.shape[0]:
        cuda.atomic.add(a, 0, v[i])

 
def main():
    # Set up for MPI
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    number_of_ranks = comm.Get_size()
    number_of_gpus_per_node = len(cuda.gpus)
    
    device_id = cuda.current_context().device.id
    gpu_info = comm.gather((my_rank, device_id), root=0)
 
    N = 10000
    a_partial = cupy.empty(N)
    
    # Set up the launch configuration for the GPU
    block = TPB
    grid = N // block if N % block == 0 else N // block + 1
    
    # Printting GPU information from rank 0 to verify
    if my_rank == 0:
        for rank, gpu_id in gpu_info:
            print(f"Rank {rank} is using GPU {gpu_id}")
            
    # Create an array with N * number_of_ranks elements

    
    ## This block will be removed
    #TODO: Perform Distributed CUDA aware MPI summation using code snippets
    # from above and collect sum as total_sum
    if my_rank == 0:
        a = cupy.random.random(N * number_of_ranks)
    else:
        a = cupy.empty(1)
        
    comm.Scatter(a, a_partial, root=0)

    partial_sum = cupy.zeros(1, dtype=cupy.float32)
    sum_values[grid, block](a_partial, partial_sum)
 
    total_sum = cupy.zeros(1, dtype=cupy.float32)
    comm.Reduce(partial_sum, total_sum, op=MPI.SUM, root=0)
    
    ## Remove until HERE

    if my_rank == 0:
        return total_sum
    
if __name__ == "__main__":
    start_time = timeit.default_timer()
    result = main()
    print(f"Sum of array: {result}")
    execution_time = (timeit.default_timer() - start_time)
    print(f"Execution time: {execution_time}")
