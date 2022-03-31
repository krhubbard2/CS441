// Kelby Hubbard
// CS F441
// Homework 3
// Problem 4
// CUDA 2d Array Sum Again

#include <stdio.h>
#include <iostream>

#define DIM 8 // dimension of the (square) array

__global__ void columnSum(int *a, int *b) {
    int bId = blockIdx.x;
    int tId = threadIdx.x;
    int i = bId * DIM + tId;
    int sum = 0;
    for (int j = 0; j < DIM; j++) {
        sum += a[i * DIM + j];
    }
    b[i] = sum;
}

int main(){
    int m[DIM][DIM];
    int *final;
    int *dev_m, *dev_final;

    final = new int[DIM];

    // initialize the array
    for(int i = 0; i < DIM; i++){
        for(int j = 0; j < DIM; j++){
            m[i][j] = 1+i;
        }
    }

    // allocate memory on the device
    cudaMalloc((void**)&dev_m, sizeof(int)*DIM*DIM);
    cudaMalloc((void**)&dev_final, sizeof(int)*DIM);
    
    // copy the array to the device
    cudaMemcpy(dev_m, m, sizeof(int)*DIM*DIM, cudaMemcpyHostToDevice);
    dim3 grid(DIM);     // blocks
    dim3 threads(DIM);  // threads per block
    columnSum <<< grid, threads >>> (dev_m, dev_final);

    // copy the array back to the host
    cudaMemcpy(final, dev_final, sizeof(int)*DIM, cudaMemcpyDeviceToHost);
    cudaFree(dev_m);
    cudaFree(dev_final);

    std::cout << "Sum of each block found using CUDA: " << std::endl;
    int totalSum = 0;
    for (int i = 0; i < DIM; i++) {
        std::cout << "Block " << i << ": " << final[i] << std::endl;
        totalSum += final[i];
    }

    std::cout << "Total sum: (found by host adding all blocks) = " << totalSum << std::endl;

    delete final;
    return 0;
}