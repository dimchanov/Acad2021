#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>
#include <cmath>
#include <limits>

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
    int n = atoi(argv[1]);
    int n_device = (n - 1)/2 + 1;
    int bytes_device = n_device * sizeof(double);

    double *h_a, *h_b, *h_c;
    size_t bytes = n * sizeof(double);

    h_a = (double *) malloc(bytes);
    h_b = (double *) malloc(bytes);
    h_c = (double *) malloc(bytes);
    cudaHostRegister(h_a, bytes, 0);
    cudaHostRegister(h_b, bytes, 0);
    cudaHostRegister(h_c, bytes, 0);

    for (int i = 0; i < n; i++){
        h_a[i] = i;
        h_b[i] = 3 * i;
    }

    double *d_a1, *d_b1, *d_c1;
    double *d_a2, *d_b2, *d_c2;
    cudaMalloc(&d_a1, bytes_device);
    cudaMalloc(&d_b1, bytes_device);
    cudaMalloc(&d_c1, bytes_device);
    cudaSetDevice(1);
    cudaMalloc(&d_a2, bytes_device);
    cudaMalloc(&d_b2, bytes_device);
    cudaMalloc(&d_c2, bytes_device);
    cudaSetDevice(0);
    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (n_device - 1) / 1024 + 1;

    cudaSetDevice(0);
    cudaMemcpyAsync(d_a1, &h_a[0], bytes_device, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b1, &h_b[0], bytes_device, cudaMemcpyHostToDevice);

    sum_vectors<<<gridSize, blockSize>>>(d_a1, d_b1, d_c1, n_device);

    cudaMemcpyAsync(&h_c[0], d_c1, bytes_device, cudaMemcpyDeviceToHost);

    cudaSetDevice(1);
    cudaMemcpyAsync(d_a2, &h_a[n_device], bytes_device, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b2, &h_b[n_device], bytes_device, cudaMemcpyHostToDevice);

    sum_vectors<<<gridSize, blockSize>>>(d_a2, d_b2, d_c2, n_device);

    cudaMemcpyAsync(&h_c[n_device], d_c2, bytes_device, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaSetDevice(0);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Gpu time: " << milliseconds << " milliseconds" << std::endl;
    
    int res = check(h_a, h_b, h_c, n);

    if (res) {
        std::cout << "Correct result" << std::endl;
    } else {
        std::cout << "Not correct result" << std::endl;
    }

    cudaFree(d_a1);
    cudaFree(d_b1);
    cudaFree(d_c1);
    cudaFree(d_a2);
    cudaFree(d_b2);
    cudaFree(d_c2);
    cudaHostUnregister(h_a);
    cudaHostUnregister(h_b);
    cudaHostUnregister(h_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
