#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

__global__ void histogram_kernel(int* input, int* histogram, int *split_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    while (tid < (*split_size)) {
        atomicAdd(&(histogram[input[tid]]), 1);
        tid += stride;
    }
}


int computeOnGPU(int *local_array, int* split_size, int* hist) {

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate data on device
    int* d_array = NULL;
    int* d_hist = NULL;

    // Malloc device local_array and hist on device
    err = cudaMalloc((void**)&d_array, (*split_size) * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&d_hist, RANGE * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Copy data to the device
    err = cudaMemcpy(d_array, local_array, (*split_size), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    /**
    Invalid MIT-MAGIC-COOKIE-1 keyInvalid MIT-MAGIC-COOKIE-1 keyError in line 49 (error code invalid argument)!
    Error in line 49 (error code invalid argument)!
    make: *** [Makefile:12: run] Error 1
    */

    // Initialize hist on device
    histogram_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_array, d_hist, split_size);
    
    err = cudaMemcpy(hist, d_hist, (*split_size) * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Print the histogram array
    for (int i = 0; i < (*split_size); i++) {
        printf("Histogram[%d] = %d\n", i, hist[i]);
    }

    // Free device global memory
    cudaFree(d_array);
    cudaFree(d_hist);
    printf("\nDone");
    return 0;
}

