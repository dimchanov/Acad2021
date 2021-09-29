#include <iostream>
#include <cuda.h>
#include <chrono>
#include <stdlib.h>
#include <ctime>
#include <cmath>
#include <limits>


__global__ void sum_vectors_if(double *a, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if ((int(a[idx]) + idx) % 3 == 0) {
            a[idx] = a[idx / 3] * (idx + 2);
        } else {
            if ((int(a[idx]) + idx) % 3 == 1) {
                a[idx] = a[idx / 3] * (idx + 1);
            } else {
                a[idx] = a[idx / 3] * (idx - 2);
            }
        }
    }
}

__global__ void sum_vectors(double *a, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] += idx % 3;
    }
}

int main(int argc, char **argv){
    int n = (int)strtol(argv[1], NULL, 10);

    size_t bytes = n * sizeof(double);

    double *h_a;
    h_a = (double *) malloc(bytes);

    srand(3333);
    for (int i = 0; i < n; i++){
        int num = rand();
        h_a[i] = num - num % 3;
    }

    double *d_a;
    cudaMalloc(&d_a, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (n - 1) / 1024 + 1;

    cudaEvent_t start, stop;
    //  sum_vectors_if
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sum_vectors_if<<<gridSize, blockSize>>>(d_a, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float sum_vectors_if = 0;
    cudaEventElapsedTime(&sum_vectors_if, start, stop);
    std::cout << "Gpu time: " << sum_vectors_if << " sum_vectors_if" << std::endl;

    //  sum_vectors
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sum_vectors<<<gridSize, blockSize>>>(d_a, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float sum_vectors = 0;
    cudaEventElapsedTime(&sum_vectors, start, stop);
    
    std::cout << "Gpu time: " << sum_vectors << " sum_vectors" << std::endl;


    cudaFree(d_a);

    free(h_a);

    return 0;
}
