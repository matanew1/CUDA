#include "myProto.h"
#include <stdio.h>

void test(int *data, int* sum) {
    for (int i = 0; i < RANGE; i++)
    {
        printf("\nhist[%d] = %d", i, data[i]);
        *sum += data[i];
    }
    printf("\nThe sum of the hist = %d",*sum);
    if( *sum == SIZE)
        printf("\nThe test passed successfully !\n"); 
    else
        printf("\nThe test failed...\n"); 
}
