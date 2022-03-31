// Kelby Hubbard
// CS F441
// Homework 3
// Problem 3
// CUDA 2d Array Sum

#include <stdio.h>
#include <iostream>

#define ROW 3 // 3 rows
#define COLUMN 4 // 4 columns

__global__ void columnSum(int *a, int *final) {
    int tId = threadIdx.x;
    for (int i = 0; i < ROW; i++) {
         final[tId] += a[i * COLUMN + tId]; // a[i * COLUMN + tId] = a[i][tId]
    }
}

int main(){
    int m[ROW][COLUMN];
    int *final;
    int *dev_m, *dev_final;

    final = new int[COLUMN]; // Will store sum of each column

    // initialize the array
    m[0][0] = 1; m[0][1] = 2; m[0][2] = 3; m[0][3] = 4;
    m[1][0] = 5; m[1][1] = 6; m[1][2] = 7; m[1][3] = 8;
    m[2][0] = 9; m[2][1] = 10; m[2][2] = 11; m[2][3] = 12;

    // allocate memory on the device
    cudaMalloc((void**)&dev_m, sizeof(int)*ROW*COLUMN);
    cudaMalloc((void**)&dev_final, sizeof(int)*COLUMN);

    // copy the array to the device
    cudaMemcpy(dev_m, m, sizeof(int)*ROW*COLUMN, cudaMemcpyHostToDevice);
    dim3 grid(1);
    dim3 threads(COLUMN);
    columnSum <<< grid, threads >>> (dev_m, dev_final);

    // copy the result back to the host
    cudaMemcpy(final, dev_final, sizeof(int)*COLUMN, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); // Wait for threads to finish
    cudaFree(dev_m);
    cudaFree(dev_final);

    std::cout << "Sum of each column found using CUDA: " << std::endl;
    int totalSum = 0;
    for (int i = 0; i < COLUMN; i++) {
        std::cout << "Column " << i << ": " << final[i] << std::endl;
        totalSum += final[i];
    }

    std::cout << "Total sum (found by host adding all column sums computed by each thread) = " << totalSum << std::endl;

    delete final;
    return 0;
}