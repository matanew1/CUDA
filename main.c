#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include "myProto.h"

int main(int argc, char* argv[]) {
    int num_procs, rank;
    int* data;
    int histogram[NUM_BINS] = {0};
    int* dev_data;
    int* dev_histogram;
    int i, j;
    int split_size = ARRAY_SIZE / 2;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Generate random data on root process
    if (rank == 0) {
        data = (int*)malloc(ARRAY_SIZE * sizeof(int));
        for (i = 0; i < ARRAY_SIZE; i++) {
            data[i] = rand() % NUM_BINS;
        }
    }
    /*
    mpiexec -np 2 ./mpiCudaOpemMP
Invalid MIT-MAGIC-COOKIE-1 keyInvalid MIT-MAGIC-COOKIE-1 keyAbort(1073365505) on node 0 (rank 0 in comm 0): Fatal error in internal_Scatter: Invalid buffer pointer, error stack:
internal_Scatter(141): MPI_Scatter(sendbuf=0x7ff8729a3010, sendcount=500000, MPI_INT, recvbuf=0x7ff8729a3010, recvcount=500000, MPI_INT, 0, MPI_COMM_WORLD) failed
internal_Scatter(109): Buffers must not be aliased
make: *** [Makefile:12: run] Error 1*/

    // Scatter data to all processes
    MPI_Scatter(data, split_size, MPI_INT, data, split_size, MPI_INT, 0, MPI_COMM_WORLD);
   
   if (computeOnGPU(data, &split_size, histogram) != 0)
      MPI_Abort(MPI_COMM_WORLD, __LINE__);

       // Reduce histograms from all processes
   MPI_Reduce(histogram, dev_histogram, NUM_BINS, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

   // Print the histogram on the root process
   if (rank == 0) {
      for (i = 0; i < NUM_BINS; i++) {
         printf("Bin %d: %d\n", i, histogram[i]);
      }
   }
   free(data);
    
   MPI_Finalize();

   return 0;
}
