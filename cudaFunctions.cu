#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

__global__ void computeHistogram(int* array, int* hist, int* size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < *size) {
      atomicAdd(&hist[array[tid]], 1);
    }
}

int computeOnGPU(int *local_array, int* split_size, int* hist) {

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate data on device
    int* d_array = NULL;
    int* d_hist = NULL;

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
    err = cudaMemcpy(d_array, local_array, (*split_size) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Initialize hist on device
    err = cudaMemset(d_hist, 0, RANGE * sizeof(int));
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Launch kernel
    computeHistogram<<<10, 20>>>(d_array, d_hist, split_size);

    // Copy histogram data back to host
    err = cudaMemcpy(hist, d_hist, RANGE * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Free device global memory
    cudaFree(d_array);
    cudaFree(d_hist);

    printf("Done\n");
    return 0;
}
