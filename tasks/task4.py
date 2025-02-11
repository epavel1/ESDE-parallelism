import cupy
import mpi4py.MPI as MPI
from numba import cuda
import timeit

TPB = 1024

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
    # Shared memory array for partial sums
    shared = cuda.shared.array(shape=TPB, dtype=cupy.float32)
    
    tid = cuda.threadIdx.x
    i = cuda.grid(1)
    
    if i < v.size:
        shared[tid] = v[i]
    else:
        shared[tid] = 0.0

    cuda.syncthreads()

    # TODO: Perform parallel reduction in shared memory

    if tid == 0:
        a[cuda.blockIdx.x] = shared[0]

def main():
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    number_of_ranks = comm.Get_size()
    number_of_gpus_per_node = len(cuda.gpus)
    device_id = my_rank % number_of_gpus_per_node
    cuda.select_device(device_id)
    gpu_info = comm.gather((my_rank, device_id), root=0)
 
    N = 10000000
    a_partial = cupy.empty(N // number_of_ranks, dtype=cupy.float32)
    partial_sum = cupy.zeros((a_partial.size + TPB - 1) // TPB, dtype=cupy.float32)
   
    # Set up the launch configuration for the GPU
    block = TPB
    grid = (a_partial.size + block - 1) // block 
    
    # Printing GPU information from rank 0 to verify
    if my_rank == 0:
        for rank, gpu_id in gpu_info:
            print(f"Rank {rank} is using GPU {gpu_id}")
            
    # Create an array with N elements on the root process
    if my_rank == 0:
        a = cupy.arange(N, dtype=cupy.float32)
    else:
        a = None

    # TODO: Use MPI to scatter the array to all processes and sum the values into total_Sum
    # Hint: Use Task2 and Task3 for reference 

    if my_rank == 0:
        return total_sum  
    
if __name__ == "__main__":
    start_time = timeit.default_timer()
    result = main()
    if result is not None:
        print(f"Sum of array: {result}")
        execution_time = (timeit.default_timer() - start_time)
        print(f"Execution time: {execution_time}")
