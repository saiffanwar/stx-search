from multiprocessing import Pool, cpu_count
import time

def my_function(x):
    # Your computation-intensive function here
    print(f'Starting worker {x}')
    time.sleep(30)
    print(f'Finishing worker {x}')
    return x * x

if __name__ == "__main__":
    inputs = [1, 2, 3, 4, 5]  # Example input list

    # Create a pool of workers, using the number of available CPU cores
    with Pool(processes=cpu_count()) as pool:
        # Distribute the function across the input data
        results = pool.map(my_function, inputs)

    print(results)
