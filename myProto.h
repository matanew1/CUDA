#pragma once

#define SIZE  1000000
#define RANGE  256
#define NUM_THREADS  20
#define NUM_BLOCKS  10

void test(int *data, int n);
int computeOnGPU(int *local_array, int* split_size, int* hist) ;
