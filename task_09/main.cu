//
//  main.cpp
//  
//
//  Created by Elijah Afanasiev on 25.09.2018.
//
//

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

__global__ void vectorAddGPU(float *a, float *b, float *c, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

void unified_samle(int size = 1048576)
{
    int n = size;
    int bytes = size * sizeof(float);

    float *h_a, *h_b, *h_c;
    
    dim3 block(256);
    dim3 grid((unsigned int)ceil(n/(float)block.x));

    printf("Allocating device unified memory on host and device\n");
    cudaMallocManaged(&h_a, bytes);
    cudaMallocManaged(&h_b, bytes);
    cudaMallocManaged(&h_c, bytes);

    for(int i=0;i<n;i++)
    {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    printf("Doing GPU Vector add\n");
    
    vectorAddGPU<<<grid, block>>>(h_a, h_b, h_c, n);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Unified memory time: %f ms\n", milliseconds);

    cudaThreadSynchronize();
}

void pinned_samle(int size = 1048576)
{
    int n = size;
    int bytes = size * sizeof(float);

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    
    dim3 block(256);
    dim3 grid((unsigned int)ceil(n/(float)block.x));

    printf("Allocating device pinned memory on host..\n");
    cudaMallocHost(&h_a, bytes);
    cudaMallocHost(&h_b, bytes);
    cudaMallocHost(&h_c, bytes);
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    for(int i=0;i<n;i++)
    {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
        h_c[i] = 0;
    }
    printf("Copying to device..\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    cudaMemcpy(d_a, h_a, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n*sizeof(float), cudaMemcpyHostToDevice);
    
    printf("Doing GPU Vector add\n");
    
    vectorAddGPU<<<grid, block>>>(d_a, d_b, d_c, n);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Pinned memory time: %f ms\n", milliseconds);

    cudaThreadSynchronize();
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}

void usual_sample(int size = 1048576)
{
    int n = size;
    
    int nBytes = n*sizeof(float);
    
    float *a, *b;  // host data
    float *c;  // results
    
    a = (float *)malloc(nBytes);
    b = (float *)malloc(nBytes);
    c = (float *)malloc(nBytes);
    
    float *a_d,*b_d,*c_d;
    
    dim3 block(256);
    dim3 grid((unsigned int)ceil(n/(float)block.x));
    
    for(int i=0;i<n;i++)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }
    
    printf("Allocating device memory on host..\n");
    
    cudaMalloc((void **)&a_d,n*sizeof(float));
    cudaMalloc((void **)&b_d,n*sizeof(float));
    cudaMalloc((void **)&c_d,n*sizeof(float));
    
    printf("Copying to device..\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    cudaMemcpy(a_d,a,n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b,n*sizeof(float), cudaMemcpyHostToDevice);
    
    printf("Doing GPU Vector add\n");
    
    vectorAddGPU<<<grid, block>>>(a_d, b_d, c_d, n);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time: %f ms\n", milliseconds);
    
    cudaThreadSynchronize();
    
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}


int main(int argc, char **argv)
{
    usual_sample(atoi(argv[1]));
    pinned_samle(atoi(argv[1]));
    unified_samle(atoi(argv[1]));
    
    return 0;
}
