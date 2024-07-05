from multiprocessing import Pool
import timeit
 
def compute_sum(start, end):
    sum_value = 0
    for i in range(start, end):
        sum_value += i * 0.001
    return sum_value
 
def main():
    n = 10000000
    num_workers = 64
    chunk_size = n // num_workers #156250
    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_workers)]
 
    ## These three lines will be removed
    # TODO: sum arrays with 64 worker threads HERE and return total_sum with total sum
    with Pool(num_workers) as pool:
        results = pool.starmap(compute_sum, ranges)
    total_sum = sum(results)
    ## Remove until HERE
    
    return total_sum
   
if __name__ == "__main__":
    total_sum = main()
    print(f"Sum of array:{total_sum}")
    print(f"Execution time:{timeit.timeit('main()', globals=globals(), number=7)/7}")
