#include <iostream>
#include <cuda.h>
#include <chrono>
#include <stdlib.h>
#include <ctime>
#include <cmath>
#include <limits>

using namespace std;

__global__ void sum_vectors_for(double *a, double *b, double *c, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (int i = 0; i < 1000; ++i) {
            double num;
            if (idx + i < size) {
                num = a[idx + i] + b[idx + i];
            }
            if (i == 0) {
                c[idx] = num;
            }
        }
    }
}

__global__ void sum_vectors_if(double *a, double *b, double *c, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int i = idx % 5;
        double num;
        if (i) {
            if (idx + i < n) {
                num = a[idx + i % 5] + b[idx + i % 5];
            }
        } else {
            num = a[idx + i] + b[idx + i];
        }
        c[idx] = num;
    }
}

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

    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (n - 1) / 1024 + 1;

    cudaEvent_t start, stop;

    //  sum_vectors_for_time
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sum_vectors_for<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float sum_vectors_for_time = 0;
    cudaEventElapsedTime(&sum_vectors_for_time, start, stop);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    int check_flag = check(h_a, h_b, h_c, n);
    if (check_flag) {
        std::cout << "Correct sum sum_vectors_for" << std::endl;
    } else {
        std::cout << "Not correct sum sum_vectors_for" << std::endl;
    }
    std::cout << "Gpu time: " << sum_vectors_for_time << " sum_vectors_for_time" << std::endl;

    //  sum_vectors_if
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sum_vectors_if<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float sum_vectors_if = 0;
    cudaEventElapsedTime(&sum_vectors_if, start, stop);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    check_flag = check(h_a, h_b, h_c, n);
    if (check_flag) {
        std::cout << "Correct sum sum_vectors_if" << std::endl;
    } else {
        std::cout << "Not correct sum sum_vectors_if" << std::endl;
    }
    std::cout << "Gpu time: " << sum_vectors_if << " sum_vectors_if" << std::endl;

    //  sum_vectors
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sum_vectors<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float sum_vectors = 0;
    cudaEventElapsedTime(&sum_vectors, start, stop);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    check_flag = check(h_a, h_b, h_c, n);
    if (check_flag) {
        std::cout << "Correct sum sum_vectors" << std::endl;
    } else {
        std::cout << "Not correct sum sum_vectors" << std::endl;
    }
    std::cout << "Gpu time: " << sum_vectors << " sum_vectors" << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
