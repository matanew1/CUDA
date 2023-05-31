#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"


/*
  tid - is the thread index within the grid.
  blockIdx.x - is the index of the current block in the x dimension.
  blockDim.x - is the size of each block in the x dimension.
  threadIdx.x - is the thread index within the block.
  stride - is the total number of threads in the grid.
*/

__global__  void initArr(int * h) {

  int index = threadIdx.x;
  h[index] = 0;

}


__global__ void histogram_kernel(int *input, int *histogram, int split_size)
{
  // gridDim.x = 10
  // blockDim.x = 20

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // range 0 - 199
    int stride = blockDim.x * gridDim.x; // 200

    /*        tid = 0
      0 -> 200 -> 400 ->...->499800
    */
    while (tid < split_size) 
    {
      atomicAdd(&(histogram[input[tid]]),1);
      tid += stride;
    }
}

int computeOnGPU(int *local_array, int *split_size, int *hist)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate data on device
    int *d_array = NULL;
    int *d_hist = NULL;

    // Malloc device local_array and hist on device
    err = cudaMalloc((void **)&d_array, (*split_size) * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_hist, RANGE * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy data to the device
    err = cudaMemcpy(d_array, local_array, (*split_size) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Initialize vectors on device
    initArr <<< 1 , RANGE >>> (d_hist);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Initialize hist on device
    histogram_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_array, d_hist, *split_size);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the result back to the host
    err = cudaMemcpy(hist, d_hist, RANGE * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(d_array);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_hist);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}
