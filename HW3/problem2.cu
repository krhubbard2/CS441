// Kelby Hubbard
// CS F441
// Homework 3
// Problem 2
// CUDA Minimum Finding

#include <stdio.h>
#include <iostream>

#define N 8000000 // size of array to generate

int findmin(int *array, int low, int high){
    int min = array[low];
    for(int i = low; i < high; i++){
        if(array[i] < min){
        min = array[i];
        }
    }
    return min;
}

__global__ void min(int *a, int *final, int p){
    int tId = threadIdx.x;
    int numToSort = N / p;
    int low = tId * numToSort;
    int high = low + numToSort - 1;

    int min = a[low];
    for(int i = low; i < high; i++){
        if(a[i] < min){
        min = a[i];
        }
    }
    final[tId] = min;


}

int main(){
    int *a, *final;
    int *dev_a, *dev_final;
    int p = 8; // number of threads to use
    dim3 grid(1);
    dim3 threads(p);
    a = new int[N];
    final = new int[p];

    // Used for confirmation that array is size as expected
    int arrSize = 0;

    srand (time(NULL)); // seed based off system time random number generator
    for (int i = 0; i < N; i++){
        a[i] = rand() % 100000000;
        arrSize++;
    }
    std::cout << std::endl;
    std::cout << "Generated an array of " << arrSize << " integers with values ranging from 0 to 1,000,000,000." << std::endl;
    

    // allocate memory on the device
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_final, p * sizeof(int));
    // copy the array to the device
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    min <<<grid, threads>>> (dev_a, dev_final, p);
    // copy the result back to the host
    cudaMemcpy(final, dev_final, p * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); // Wait for threads to finish
    cudaFree(a);
    cudaFree(dev_final);

    for (int i = 0; i < p; i++){
        std::cout << "Min Value Found in Thread " << i << ": " << final[i] << std::endl;
    }

    int finalMin = findmin(final, 0, p);
    std::cout << "Min Value Found in All Threads: " << finalMin << std::endl;

    std::cout << "Host will now sequentially find the minimum value in the entire array to confirm answer." << std::endl;

    int min = findmin(a, 0, N);
    std::cout << "Min Value Found in Host: " << min << std::endl;

    delete a;
    delete final;
    return 0;
}