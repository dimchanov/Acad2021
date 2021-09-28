#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>

__global__ void staticReverse(int *d, int n)
{
    __shared__ int staticMem[12288];
    int idx = threadIdx.x;
    if (n <= blockDim.x & idx < n) {
        staticMem[n - 1 - idx] = d[idx];
    } else {
        int k = idx * 12;
        if (k < n) {
            int i = 0;
            while (i < 12 && (k + i) < n) {
                staticMem[n - 1 - (k + i)] = d[k + i];
                ++i;
            }
        }
    }
    __syncthreads();
    idx = threadIdx.x;
    if (n <= blockDim.x && idx < n) {
        d[idx] = staticMem[idx];
    } else {
        int k = idx * 12;
        if (k < n) {
            int i = 0;
            while (i < 12 && (k + i) < n) {
                d[k + i] = staticMem[k + i];
                ++i;
            }
        }
    }
}

__global__ void dynamicReverse(int *d, int n)
{
    extern __shared__ int dynamicMem[];
    int idx = threadIdx.x;
    int *arr = dynamicMem;
    if (n <= blockDim.x & idx < n) {
        arr[n - 1 - idx] = d[idx];
    } else {
        int k = idx * 12;
        if (k < n) {
            int i = 0;
            while (i < 12 && (k + i) < n) {
                arr[n - 1 - (k + i)] = d[k + i];
                ++i;
            }
        }
    }
    __syncthreads();
    idx = threadIdx.x;
    if (n <= blockDim.x && idx < n) {
        d[idx] = arr[idx];
    } else {
        int k = idx * 12;
        if (k < n) {
            int i = 0;
            while (i < 12 && (k + i) < n) {
                d[k + i] = arr[k + i];
                ++i;
            }
        }
    }
}

int main(int argc, char **argv)
{
    int n = atoi(argv[1]); // FIX ME TO max possible size
    int  r[n]; // FIX ME TO dynamic arrays if neccesary
    size_t bytes = n * sizeof(int);
    int *a = (int *) malloc(bytes);
    int *d = (int *) malloc(bytes);
    for (int i = 0; i < n; i++) {
        a[i] = i;
        r[i] = n-i-1;
        d[i] = 0;
    }

    int *d_d;
    cudaMalloc(&d_d, bytes); 
    int blockSize = 1024;



    // run version with static shared memory


    cudaMemcpy(d_d, a, bytes, cudaMemcpyHostToDevice);
    staticReverse<<<1, blockSize>>>(d_d, n); // FIX kernel execution params
    cudaMemcpy(d, d_d, bytes, cudaMemcpyDeviceToHost);
    int flag = 1;
    for (int i = 0; i < n; i++) {
        if (d[i] != r[i]) {
            printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
            flag = 0;
        }
    }
    if (flag) {
        printf("staticReverse OK\n");
    }

    for (int i = 0; i < n; ++i) {
        d[i] = 0;
    } 

    // run dynamic shared memory version
    cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
    dynamicReverse<<<1, blockSize,  n*sizeof(int)>>>(d_d, n); // FIX kernel executon params
    cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
    flag = 1;
    for (int i = 0; i < n; i++) { 
        if (d[i] != r[i]) {
            printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
            flag = 0;
        }
    }
    if (flag) {
        printf("dynamicReverse OK\n");
    }

    free(a);
    free(d);
    cudaFree(d_d);
    return 0;
}
