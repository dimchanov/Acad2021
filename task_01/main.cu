#include <iostream>
#include <cuda.h>
#include <chrono>
#include <stdlib.h>
#include <ctime>
#include <cmath>
#include <limits>

using namespace std;

__global__ void sum_vectors(double *a, double *b, double *c, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

int check(double *a, double *b, double *c, int size) {
    for (int i = 0; i < size; ++i) {
        if (std::fabs(c[i] - (a[i] + b[i])) > std::numeric_limits<double>::epsilon()) {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv){
    int n = (int)strtol(argv[1], NULL, 10);

    double *h_a, *h_b, *h_c;

    size_t bytes = n * sizeof(double);

    h_a = (double *) malloc(bytes);
    h_b = (double *) malloc(bytes);
    h_c = (double *) malloc(bytes);

    for (int i = 0; i < n; i++){
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }

    double *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (n - 1) / 1024 + 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    sum_vectors<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    auto start_cpu = std::chrono::high_resolution_clock::now();

    int check_flag = check(h_a, h_b, h_c, n);

    auto stop_cpu = std::chrono::high_resolution_clock::now();
    auto elapsed_time_cpu = stop_cpu - start_cpu;

    if (check_flag) {
        std::cout << "Correct sum" << std::endl;
    } else {
        std::cout << "Not correct sum" << std::endl;
    }

    std::cout << "Cpu time: " << elapsed_time_cpu.count() / 1000 << " milliseconds" << std::endl;
    std::cout << "Gpu time: " << milliseconds << " milliseconds" << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
