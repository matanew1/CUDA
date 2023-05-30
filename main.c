#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include "myProto.h"

int main(int argc, char *argv[])
{
   MPI_Status status;
   int* array = NULL, *hist = NULL, *local_array = NULL;
   int *d_array = NULL, *d_hist = NULL;
    
   int rank, size;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   if (size != 2)
   {
      printf("Run the example with two processes only\n");
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
   }
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   int split_size = SIZE / size;

   hist = (int* )calloc(RANGE, sizeof(int));
   array = (int *)malloc(SIZE * sizeof(int));
   local_array = (int*)malloc(split_size * sizeof(int));

      
   srand(time(NULL));

   if (rank == 0) 
   {
      #pragma omp parallel
      {
         #pragma omp parallel for
         for (int i = 0; i < SIZE; i++) {
            array[i] = rand() % RANGE;
         }
      }
   }

   MPI_Scatter(array, split_size, MPI_INT, local_array, split_size, MPI_INT, 0, MPI_COMM_WORLD);
   
   if (computeOnGPU(local_array, &split_size, hist) != 0)
      MPI_Abort(MPI_COMM_WORLD, __LINE__);

   // int* gathered_hist = NULL;
   // if (rank == 0) {
   //    gathered_hist = (int*)malloc(split_size * sizeof(int));
   // }
    
   // MPI_Gather(hist, split_size, MPI_INT, gathered_hist, split_size, MPI_INT, 0, MPI_COMM_WORLD);

   // if (rank == 0) {
   //    // Print the histogram
   //    for (int i = 0; i < split_size; i++) {
   //       printf("hist[%d]: %d\n", i, gathered_hist[i]);
   //    }       
   //    free(gathered_hist);
   // }
   // free(local_array);
    
   MPI_Finalize();

   return 0;
}
