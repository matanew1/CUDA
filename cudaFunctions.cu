#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

  __global__  void buildHist(int *h, int *temp) {
    int index = threadIdx.x;
  
    for (int i = 0; i < NUM_THREADS;   i++)
      h[index] += temp[index + i*RANGE];
  }
  __global__  void buildTemp(int *A, int *temp) {
  
    int index = threadIdx.x;
    int offset_A = SIZE / NUM_THREADS * index;
    int offset_temp = RANGE * index;
  
    for (int i = 0;    i < SIZE / NUM_THREADS;   i++) {
      int value = A[offset_A + i];
      temp[offset_temp + value]++;
    }
  
  }
  __global__  void initHist(int * h) {
    int index = threadIdx.x;
    printf("[%d]",index);
    h[index] = 0;
  
  }
  __global__  void initTemp(int * temp) {
  
    int index = threadIdx.x;
    int offset = RANGE * index;
    for (int i = 0;    i < RANGE;   i++)
      temp[offset + i] = 0;
  }

int computeOnGPU(int *data, int numElements, int* hist) {

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    size_t size = numElements * sizeof(int);

    // Allocate data on device
    int* device_data = NULL;
    err = cudaMalloc((void **)&device_data, SIZE / 2 * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int* device_temp = NULL;
    err = cudaMalloc((void **)&device_temp, NUM_BLOCKS * NUM_THREADS * RANGE * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Allocate hist on device
    int* device_hist = NULL;
    err = cudaMalloc((void **)&device_hist, RANGE * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Copy data to the device
    err = cudaMemcpy(device_data, data, SIZE / 2, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    //Copy data to the device
    err = cudaMemcpy(device_hist, hist, RANGE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Initialize hist on device
    initHist<<<1, RANGE>>>(device_hist); // 1 block with 256 threads
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Initialize data on device
    initTemp <<<1, RANGE >>> (device_data);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Unify the results
    buildHist<<< NUM_BLOCKS, NUM_THREADS >>>(device_hist, device_data);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Copy the final histogram to the host
    err = cudaMemcpy(hist, device_hist, RANGE * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }


    // Free device global memory
    err = cudaFree(device_data);
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
    err = cudaFree(device_hist);
    if (err != cudaSuccess) {
      fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

