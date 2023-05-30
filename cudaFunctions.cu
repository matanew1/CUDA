#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

__global__ void calculateHistogram(int* data, int* histogram, int* size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < *size) {
        atomicAdd(&histogram[data[tid]], 1);
    }
}

int computeOnGPU(int *data, int* split_size, int* histogram) {

    int* dev_data;
    int* dev_histogram;

        // Allocate memory on GPU
    cudaMalloc((void**)&dev_data, (*split_size) * sizeof(int));
    cudaMalloc((void**)&dev_histogram, NUM_BINS * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(dev_data, data, (*split_size) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(dev_histogram, 0, NUM_BINS * sizeof(int));

    // Launch CUDA kernels
    int num_blocks = (*split_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    calculateHistogram<<<num_blocks, BLOCK_SIZE>>>(dev_data, dev_histogram, split_size);
    
    // Copy histogram from device to host
    cudaMemcpy(histogram, dev_histogram, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device global memory
    cudaFree(dev_data);
    cudaFree(dev_histogram);

    printf("Done\n");
    return 0;
}
