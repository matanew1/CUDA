#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

__global__ void computeHistogram(int* array, int* hist, int* size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < *size) {
      hist[array[tid]]++;
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
    err = cudaMemcpy(d_array, local_array, (*split_size), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
    //Copy data to the device
    err = cudaMemcpy(d_hist, hist, RANGE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Initialize hist on device
    computeHistogram<<<NUM_BLOCKS, NUM_THREADS>>>(d_array, d_hist, split_size);
    
    cudaMemcpy(hist, d_hist, RANGE * sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i=0; i < RANGE; i++) 
    // {
    //   printf("\n%d %d", i,hist[i]);
    // }

    // Free device global memory
    cudaFree(d_array);
    cudaFree(d_hist);

    printf("Done\n");
    return 0;
}

