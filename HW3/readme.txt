Commands to run programs:
    hw 1:
        mpiCC problem1.cpp
        mpirun -np 8 ./a.out

    hw 2:
        nvcc -o minFindCuda problem2.cu
        ./minFindCuda

    hw 3:
        nvcc -o 2dArraySum problem3.cu
        ./2dArraysum

    hw 4:
        nvcc -o 2dArraySumAgain problem4.cu
        ./2dArraySumAgain

    hw 5:
        nvcc -o rayGPU problem5.cu -lfreeimage
        ./rayGPU

All programs work as intended, except for problem 5. Compiles and outputs, but spheres aren't shown correctly.