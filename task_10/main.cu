
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

__global__ void vectorAddGPU(float *a, float *b, float *c, int N, int offset)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx < N)
    {
        c[offset + idx] = a[offset + idx] + b[offset + idx];
    }
}

int check(float *a, float *b, float *c, int size) {
    for (int i = 0; i < size; ++i) {
        if (std::fabs(c[i] - (a[i] + b[i])) > std::numeric_limits<double>::epsilon()) {
            return 0;
        }
    }
    return 1;
}

void sample_vec_add(int size = 1048576)
{
    int n = size;
    
    int nBytes = n*sizeof(int);
    
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
    
    vectorAddGPU<<<grid, block>>>(a_d, b_d, c_d, n, 0);
    cudaMemcpy(c,c_d,n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time: %f ms\n", milliseconds);
    
    cudaDeviceSynchronize();

    int res = check(a, b, c, n);
    if (res) {
        std::cout << "Correct result" << std::endl;
    } else {
        std::cout << "Not correct result" << std::endl;
    }
    
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

void streams_vec_add(int n_streams, int size = 1048576)
{
    int n = size;
    
    float *a, *b;  // host data
    float *c;  // results
    
    cudaHostAlloc( (void**) &a, n * sizeof(float) ,cudaHostAllocDefault );
    cudaHostAlloc( (void**) &b, n * sizeof(float) ,cudaHostAllocDefault );
    cudaHostAlloc( (void**) &c, n * sizeof(float) ,cudaHostAllocDefault );
    
    float *a_d,*b_d,*c_d;
    
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
    
    printf("Doing GPU-stream Vector add\n");
    
    
    const int NbStreams = n_streams;
    const int StreamSize = n / NbStreams;
    cudaStream_t Stream[NbStreams];
    for ( int i = 0; i < NbStreams; i++ )
        cudaStreamCreate(&Stream[i]);
    

    for ( int i = 0; i < NbStreams; i++ )
    {
        int Offset = i * StreamSize;
        
        cudaMemcpyAsync(&a_d[Offset], &a[Offset], StreamSize * sizeof(float), cudaMemcpyHostToDevice, Stream[ i ]);
        cudaMemcpyAsync(&b_d[Offset], &b[Offset], StreamSize * sizeof(float), cudaMemcpyHostToDevice, Stream[ i ]);
        cudaMemcpyAsync(&c_d[Offset], &c[Offset], StreamSize * sizeof(float), cudaMemcpyHostToDevice, Stream[ i ]);
        
        dim3 block(1024);
        dim3 grid((StreamSize - 1)/1024 + 1);
        vectorAddGPU<<<grid, block, 0, Stream[i]>>>(a_d, b_d, c_d, StreamSize, Offset);
    
        cudaMemcpyAsync(&a[Offset], &a_d[Offset], StreamSize * sizeof(float), cudaMemcpyDeviceToHost, Stream[ i ]);
        cudaMemcpyAsync(&b[Offset], &b_d[Offset], StreamSize * sizeof(float), cudaMemcpyDeviceToHost, Stream[ i ]);
        cudaMemcpyAsync(&c[Offset], &c_d[Offset], StreamSize * sizeof(float), cudaMemcpyDeviceToHost, Stream[ i ]);

    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "STREAMS NUMBERS: " << NbStreams << std::endl;
    printf("GPU-stream time: %f ms\n", milliseconds);
    
    cudaDeviceSynchronize();
    int res = check(a, b, c, n);
    if (res) {
        std::cout << "Correct result" << std::endl;
    } else {
        std::cout << "Not correct result" << std::endl;
    }
    
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
}


int main(int argc, char **argv)
{
    sample_vec_add(atoi(argv[1]));
    sample_vec_add(atoi(argv[1]));
    int n_streams = (argc == 3) ? atoi(argv[2]) : 8;
    std::cout << "=================================================" << std::endl;
    std::cout << "STREAMS NUMBERS: " << n_streams << std::endl;
    streams_vec_add(n_streams, atoi(argv[1]));

    return 0;
}
