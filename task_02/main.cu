#include <iostream>
#include <cuda.h>
#include <chrono>
#include <stdlib.h>
#include <ctime>
#include <cmath>
#include <limits>

#define BLOCK_SIZE 1024

__global__ void gpu_transposition(double *a, double *b, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {
        b[(idx / n) + (idx % n)*m] = a[idx];
    }
}

void cpu_transposition(double *a, double *b, int m, int n)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            b[j * m + i] = a[i * n + j];
        }
    }
}

void print_matrix(double *a, int m, int n) {
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            std::cout << a[i * m + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
int main(int argc, char const **argv)
{
    int m, n;
    srand(3333);
    m = atoi(argv[1]);
    n = atoi(argv[2]);

    size_t bytes = m * n * sizeof(double);
    

    double *h_a, *h_b, *h_c;
    h_a = (double *) malloc(bytes);
    h_b = (double *) malloc(bytes);
    h_c = (double *) malloc(bytes);


    // random initialize matrix
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            h_a[i * n + j] = (double)(rand()%1024); //для наглядности
        }
    }
    // print_matrix(h_a, n, m);

    double *d_a, *d_b;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    int gridSize;
    gridSize = (m * n - 1) / BLOCK_SIZE + 1;
    
    float gpu_elapsed_time, cpu_elapsed_time;
    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // start the GPU version
    cudaEventRecord(start, 0);
    
    gpu_transposition<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, m, n);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
      
    //time elapse on GPU
    cudaEventElapsedTime(&gpu_elapsed_time, start, stop);
    printf("Time elapsed on matrix transposition of %dx%d on GPU: %f ms.\n\n", m, n, gpu_elapsed_time);
    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
    // print_matrix(h_b, m, n);
    // start the CPU version
    cudaEventRecord(start, 0);
    
    cpu_transposition(h_a, h_c, m, n);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time, start, stop);
    printf("Time elapsed on matrix transposition of %dx%d on CPU: %f ms.\n\n", m, n, cpu_elapsed_time);
    // print_matrix(h_c, m, n);
    // validate results computed by GPU
    int all_ok = 1;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            if(h_b[i*m + j] != h_c[i*m + j])
            {
                all_ok = 0;
            }
        }
    }
    
    // roughly compute speedup
    if(all_ok)
    {
        printf("all results are correct!!!, speedup = %f\n", cpu_elapsed_time / gpu_elapsed_time);
    }
    else
    {
        printf("incorrect results\n");
    }
    
    // free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    
    return 0;
}
