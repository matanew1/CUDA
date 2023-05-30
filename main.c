#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include "myProto.h"


int main(int argc, char *argv[])
{
    MPI_Status status;
    int *array = NULL, *hist = NULL, *local_array = NULL;

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

    hist = (int *)calloc(RANGE, sizeof(int));
    array = (int *)malloc(SIZE * sizeof(int));
    local_array = (int *)malloc(split_size * sizeof(int));

    srand(time(NULL));

    if (rank == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < SIZE; i++)
        {
            array[i] = rand() % RANGE;
            printf("\n%d %d",i,array[i]);
        } 
    }

    MPI_Scatter(array, split_size, MPI_INT, local_array, split_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (computeOnGPU(local_array, &split_size, hist) != 0)
    {
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }

    int *global_hist = NULL;
    if (rank == 0)
    {
        global_hist = (int *)calloc(RANGE, sizeof(int)); // Allocate memory for the global histogram
    }

    // Perform reduction to merge local histograms into the global histogram
    MPI_Reduce(hist, global_hist, RANGE, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Only process 0 has the merged histogram
    if (rank == 0)
    {
        int sum = 0;
        for (int i = 0; i < RANGE; i++)
        {
            printf("\nglobal_hist[%d] = %d", i, global_hist[i]);
            sum += global_hist[i];
        }
        printf("\n\nglobal_sum = %d\n", sum);
        // Free the memory allocated for the global histogram
        free(global_hist);
    }

    free(array);
    free(local_array);
    free(hist);
    MPI_Finalize();

    return 0;
}
