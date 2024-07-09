from numba import cuda, float32
import numpy as np
import timeit

TPB = 1024  # Number of threads per block

@cuda.jit
def parallel_sum(arr, result):
    sdata = cuda.shared.array(shape=TPB, dtype=float32)

    tid = cuda.threadIdx.x
    i = cuda.grid(1)
    
    if i < arr.size:
        sdata[tid] = arr[i]
    else:
        sdata[tid] = 0.0

    cuda.syncthreads()

    stride = 1
    while stride < cuda.blockDim.x:
        index = 2 * stride * tid
        if index < cuda.blockDim.x:
            sdata[index] += sdata[index + stride]
        cuda.syncthreads()
        stride *= 2

    if tid == 0:
        result[cuda.blockIdx.x] = sdata[0]

def main(arr):
    number_of_gpus_per_node = len(cuda.gpus)
    n = arr.size
    d_arr = cuda.to_device(arr)
    d_result = cuda.device_array((n + TPB - 1) // TPB, dtype=np.float32)

    block = TPB
    grid = (n + block - 1) // block  # Ensure enough blocks to cover the array size 
        
    parallel_sum[grid, block](d_arr, d_result)

    result = d_result.copy_to_host()
    
    return np.sum(result)

if __name__ == "__main__":
    arr = np.arange(10000000)
    start_time = timeit.default_timer()
    result = main(arr)
    print(f"Sum of array: {result}")
    execution_time = (timeit.default_timer() - start_time)
    print(f"Execution time: {execution_time}")
