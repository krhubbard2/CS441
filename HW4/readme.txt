Commands to run program:
    nvcc -o sobel-gpu sobel-gpu.cu -lfreeimage
    ./sobel-gpu
    
Note, this program requires coins.png, not found in this directory.
All programs work as intended.

Time Comparison (magnitudes faster):
time ./sobel-cpu

real    0m2.150s
user    0m2.079s
sys     0m0.071s

time ./sobel-gpu

real    0m0.377s
user    0m0.189s
sys     0m0.057s
