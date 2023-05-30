#pragma once

#define ARRAY_SIZE 1000000
#define NUM_BINS 256
#define BLOCK_SIZE 20
#define NUM_BLOCKS  10

void test(int *data, int n);
int computeOnGPU(int *local_array, int* split_size, int* hist) ;
