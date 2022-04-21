// Kelby Hubbard
// CS 441
// Assignment 4

/***********************************************************************
 * sobel-cpu.cu
 *
 * Implements a Sobel filter on the image that is hard-coded in main.
 * You might add the image name as a command line option if you were
 * to use this more than as a one-off assignment.
 *
 * See https://stackoverflow.com/questions/17815687/image-processing-implementing-sobel-filter
 * or https://blog.saush.com/2011/04/20/edge-detection-with-the-sobel-operator-in-ruby/
 * for info on how the filter is implemented.
 *
 * Compile/run with:  nvcc sobel-cpu.cu -lfreeimage
 *
 ***********************************************************************/
#include "FreeImage.h"
#include "stdio.h"
#include "math.h"

// Returns the index into the 1d pixel array
// Given te desired x,y, and image width
// For use on device
__device__ int pixelIndexGPU(int x, int y, int width)
{
    return (y*width + x);
}

// For use on host
int pixelIndex(int x, int y, int width)
{
    return (y*width + x);
}

__global__ void sobelGPU(char *pixels, char *out, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ( x > 0 && y > 0 && x < width && y < height){
      int x00 = -1;  int x20 = 1;
      int x01 = -2;  int x21 = 2;
      int x02 = -1;  int x22 = 1;
      x00 *= pixels[pixelIndexGPU(x-1,y-1,width)];
      x01 *= pixels[pixelIndexGPU(x-1,y,width)];
      x02 *= pixels[pixelIndexGPU(x-1,y+1,width)];
      x20 *= pixels[pixelIndexGPU(x+1,y-1,width)];
      x21 *= pixels[pixelIndexGPU(x+1,y,width)];
      x22 *= pixels[pixelIndexGPU(x+1,y+1,width)];
      
      int y00 = -1;  int y10 = -2;  int y20 = -1;
      int y02 = 1;  int y12 = 2;  int y22 = 1;
      y00 *= pixels[pixelIndexGPU(x-1,y-1,width)];
      y10 *= pixels[pixelIndexGPU(x,y-1,width)];
      y20 *= pixels[pixelIndexGPU(x+1,y-1,width)];
      y02 *= pixels[pixelIndexGPU(x-1,y+1,width)];
      y12 *= pixels[pixelIndexGPU(x,y+1,width)];
      y22 *= pixels[pixelIndexGPU(x+1,y+1,width)];

      int px = x00 + x01 + x02 + x20 + x21 + x22;
      int py = y00 + y10 + y20 + y02 + y12 + y22;

      out[y*width + x] = sqrt( (float) px*px + py*py);
    }

}

int main()
{
    FreeImage_Initialise();
    atexit(FreeImage_DeInitialise);

    // Load image and get the width and height
    FIBITMAP *image;
    image = FreeImage_Load(FIF_PNG, "coins.png", 0);
    if (image == NULL)
    {
        printf("Image Load Problem\n");
        exit(0);
    }
    int imgWidth;
    int imgHeight;
    imgWidth = FreeImage_GetWidth(image);
    imgHeight = FreeImage_GetHeight(image);
    
    RGBQUAD aPixel;
    char *pixels; 
    int pixIndex = 0;
    pixels = (char *) malloc(sizeof(char)*imgWidth*imgHeight);

    // Convert image into a flat array of chars with the value 0-255 of the
    // greyscale intensity
    for (int i = 0; i < imgHeight; i++)
     for (int j = 0; j < imgWidth; j++)
     {
       FreeImage_GetPixelColor(image,j,i,&aPixel);
       char grey = ((aPixel.rgbRed + aPixel.rgbGreen + aPixel.rgbBlue)/3);
       pixels[pixIndex++]=grey;
     }

    FIBITMAP *bitmap = FreeImage_Allocate(imgWidth, imgHeight, 24);
   
    // GPU:
    char *dev_pixels;
    char *dev_sobel;
    // Allocate memory on the GPU
    cudaMalloc((void**)&dev_pixels, sizeof(char)*imgWidth*imgHeight);
    cudaMalloc((void**)&dev_sobel, sizeof(char)*imgWidth*imgHeight);
    // Copy the image to the GPU
    cudaMemcpy(dev_pixels, pixels, sizeof(char)*imgWidth*imgHeight, cudaMemcpyHostToDevice);
    // Execute the kernel
    dim3 blocks((imgWidth+31), (imgHeight+31));
    dim3 threads(16, 16);
    sobelGPU<<<blocks, threads>>>(dev_pixels, dev_sobel, imgWidth, imgHeight);
    // Copy the result back to the host
    cudaDeviceSynchronize();
    cudaMemcpy(pixels, dev_sobel, sizeof(char)*imgWidth*imgHeight, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Apply the sobel filter to the image
    for (int i = 1; i < imgHeight - 1; i++)
    {
        for (int j = 1; j < imgWidth - 1; j++)
        {
            int index = pixelIndex(j, i, imgWidth);
            aPixel.rgbRed = pixels[index];
            aPixel.rgbGreen = pixels[index];
            aPixel.rgbBlue = pixels[index];
            FreeImage_SetPixelColor(bitmap, j, i, &aPixel);
        }
    }

    // Save the image
    FreeImage_Save(FIF_PNG, bitmap, "coins-edge.png", 0);
    cudaFree(dev_pixels);
    cudaFree(dev_sobel);
    free(pixels);
    FreeImage_Unload(bitmap);
    FreeImage_Unload(image);
    return 0;
}
