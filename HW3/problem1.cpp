// Kelby Hubbard
// CS F441
// Homework 3
// Problem 1
// MPI Minimum Finding

#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <algorithm>

// Returns the minimum of an array from indicies array[low] and array[high]
int findmin(int *array, int low, int high){
    int min = array[low];
    for(int i = low; i < high; i++){
        if(array[i] < min){
        min = array[i];
        }
    }
    return min;
}


int main(int argc, char *argv[]){
    int *a, *final;
    int rank, p;
    int N = 8000000; // size of array to generate

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // Allocate arrays to be used in program
    a = new int[N];
    final = new int[8];

    // Used for confirmation that array is size as expected
    int arrSize = 0;

    srand (time(NULL)); // seed based off system time random number generator
    if (rank == 0){
        for (int i = 0; i < N; i++){
            a[i] = rand() % 100000000;
            arrSize++;
        }
        std::cout << std::endl;
        std::cout << "Generated an array of " << arrSize << " integers with values ranging from 0 to 1,000,000,000." << std::endl;
    }
    MPI_Bcast(a, N, MPI_INT, 0, MPI_COMM_WORLD);

    int numToSort = N / p;
    int low = rank * numToSort;
    int high = low + numToSort - 1;

    // Each process find the min in their chunk of the array
    int min = findmin(a, low, high);    
    // Wait for each process to find the minimum value
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank != 0){
        // All processes (other than 0) send their minimum value to process 0
        MPI_Send(&min, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    if (rank == 0){
        // Add process 0 to final array
        final[0] = min;    
        // Receive all other processes' minimum values
        for (int i = 1; i < p; i++){
            MPI_Recv(&final[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for (int i = 0; i < p; i++){
            std::cout << "Process " << i << " has the minimum value: " << final[i] << std::endl;
        }
     
        int finalMin = findmin(final, 0, p);
        std::cout << "\nThe minimum value found using MPI is: " << finalMin << std::endl;

        std::cout << "Using process 0 to find minimum value by sequentially searching now." << std::endl;

        int min = a[0];
        for (int i = 0; i < N; i++){
            if (a[i] < min){
                min = a[i];
            }
        }

        std::cout << "The minimum value found using process 0 is: " << min << std::endl;
    }
    
    delete final;
    delete a;
    MPI_Finalize();

    return 0;
}