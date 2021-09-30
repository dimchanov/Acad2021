// UCSC CMPE220 Advanced Parallel Processing 
// Prof. Heiner Leitz
// Author: Marcelo Siero.
// Modified from code by:: Andreas Goetz (agoetz@sdsc.edu)
// CUDA program to perform 1D stencil operation in parallel on the GPU
//
// /* FIXME */ COMMENTS ThAT REQUIRE ATTENTION

#include <iostream>
#include <stdio.h>
#include <cuda_device_runtime_api.h>

// define vector length, stencil radius, 
#define N (1024*1024*512l)
#define RADIUS 3
#define GRIDSIZE 128
#define BLOCKSIZE 256

int gridSize  = GRIDSIZE;
int blockSize = BLOCKSIZE;

float time_milliseconds = 0;

void cudaErrorCheck() {
  std::cout << "=====================================================" << std::endl;
  cudaError_t error = cudaGetLastError();
  std::string errorName = std::string(cudaGetErrorName(error));
  std::cout << "Error name: " << errorName << std::endl;

  std::string errorDescription = std::string(cudaGetErrorString(error));
  std::cout << "**** " << errorDescription << " ****" << std::endl;
  std::cout << "=====================================================" << std::endl;
}

void start_timer(cudaEvent_t* start) {
  cudaEventCreate(start);
  cudaEventRecord(*start);
}

float stop_timer(cudaEvent_t* start, cudaEvent_t* stop) {
  cudaEventCreate(stop);
  cudaEventRecord(*stop);

  cudaEventSynchronize(*stop);
  cudaEventElapsedTime(&time_milliseconds, *start, *stop);
  return(time_milliseconds);
}

cudaDeviceProp prop;
int device;
void getDeviceProperties() {
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  std::cout << "Major and minor cuda capabilities are: " << prop.major << ", " << prop.minor << std::endl;
  std::cout << "Total device global memory is : " << static_cast<int>(prop.totalGlobalMem) << " bytes" << std::endl;
  std::cout << "Size of shared memory per block is : " << static_cast<int>(prop.sharedMemPerBlock) << " bytes" << std::endl;
  std::cout << "Number of registers per block is: " << prop.regsPerBlock << std::endl;
  std::cout << "Warp size is : " << prop.warpSize << " threads" << std::endl;
  std::cout << "Max number of threads per block is : " << prop.maxThreadsPerBlock << std::endl;
  std::cout << "Number of multiprocessors is : " << prop.multiProcessorCount << " per device" << std::endl;
  std::cout << "Number of Maximum number of threads per block dimension (x,y,z) per device: " << prop.maxThreadsDim[0] <<", "<< prop.maxThreadsDim[1] << ", "<< prop.maxThreadsDim[2] << std::endl;
  std::cout << "Maximumum number of blocks per grid dimension " << prop.maxGridSize[0] << ", " << prop.maxGridSize[0] <<", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << std::endl;
}

void newline() { std::cout << std::endl; };

void printThreadSizes() {
   int noOfThreads = gridSize * blockSize;
   printf("Blocks            = %d\n", gridSize);  // no. of blocks to launch.
   printf("Threads per block = %d\n", blockSize); // no. of threads to launch.
   printf("Total threads     = %d\n", noOfThreads);
   printf("Number of grids   = %d\n", (N + noOfThreads -1)/ noOfThreads);
}

// -------------------------------------------------------
// CUDA device function that performs 1D stencil operation
// -------------------------------------------------------
__global__ void stencil_1D(int *in, int *out, long dim){
  __shared__ int sh_mem[BLOCKSIZE + 2*RADIUS];
  long gindex = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = gridDim.x * blockDim.x;

  while ( gindex < (dim + blockDim.x) ) {
    /* FIXME PART 2 - MODIFIY PROGRAM TO USE SHARED MEMORY. */
    if (gindex < dim) {
      sh_mem[threadIdx.x + RADIUS] = in[gindex];
    } else {
      sh_mem[threadIdx.x + RADIUS] = 0;
    }

    if (threadIdx.x < RADIUS) {
      if (gindex < RADIUS) {
        sh_mem[threadIdx.x] = 0;
      } else {
        sh_mem[threadIdx.x] = in[gindex - RADIUS];  
      }
      if (gindex + BLOCKSIZE >= dim) {
        sh_mem[threadIdx.x + RADIUS + BLOCKSIZE] = 0;  
      } else {
        sh_mem[threadIdx.x + RADIUS + BLOCKSIZE] = in[gindex + BLOCKSIZE];  
      }
    }
    __syncthreads();
    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++) {
      if (int(threadIdx.x + RADIUS + offset) < dim && int(threadIdx.x + RADIUS + offset) > -1)
	      result += sh_mem[threadIdx.x + RADIUS + offset];
    }

    // Store the result
    if (gindex < dim)
      out[gindex] = result;

    // Update global index and quit if we are done
    gindex += stride;

    __syncthreads();

  }
}

#define True  1
#define False 0
void checkResults(int *h_in, int *h_out, int DoCheck=True) {
   // DO NOT CHANGE THIS CODE.
   // CPU calculates the stencil from data in *h_in
   // if DoCheck is True (default) it compares it with *h_out
   // to check the operation of this code.
   // If DoCheck is set to False, it can be used to time the CPU.
   int i, j, ij, result, err;
   err = 0;
   for (i=0; i<N; i++){  // major index.
      result = 0;
      for (j=-RADIUS; j<=RADIUS; j++){
         ij = i+j;
         if (ij>=0 && ij<N)
            result += h_in[ij];
      }
      if (DoCheck) {  // print out some errors for debugging purposes.
         if (h_out[i] != result) { // count errors.
            err++;
            if (err < 8) { // help debug
               printf("h_out[%d]=%d should be %d\n",i,h_out[i], result);
            };
         }
      } else {  // for timing purposes.
         h_out[i] = result;
      }
   }

   if (DoCheck) { // report results.
      if (err != 0){
         printf("Error, %d elements do not match!\n", err);
      } else {
         printf("Success! All elements match CPU result.\n");
      }
   }
}

// ------------
// main program
// ------------
int main(void)
{
  int *h_in, *h_out;
  int *d_in, *d_out;
  long size = N * sizeof(int);
  int i;

  // allocate host memory
  h_in = new int[N];
  h_out = new int[N];

  getDeviceProperties();

  // initialize vector
  for (i=0; i<N; i++){
    //    h_in[i] = i+1;
    h_in[i] = 1;
  }

  // allocate device memory
  cudaMalloc((void **)&d_in, size);
  cudaMalloc((void **)&d_out, size);
  cudaErrorCheck();

  // copy input data to device
  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
  cudaErrorCheck();

  // Apply stencil by launching a sufficient number of blocks
  printf("\n---------------------------\n");
  printf("Launching 1D stencil kernel\n");
  printf("---------------------------\n");
  printf("Vector length     = %ld (%ld MB)\n",N,N*sizeof(int)/1024/1024);
  printf("Stencil radius    = %d\n",RADIUS);

  //----------------------------------------------------------
  // CODE TO RUN AND TIME THE STENCIL KERNEL.
  //----------------------------------------------------------

  newline();
  printThreadSizes();
  cudaEvent_t start, stop;
  start_timer(&start);

  stencil_1D<<<gridSize,blockSize>>>(d_in, d_out, N);
  std::cout << "Elapsed time: " << stop_timer(&start, &stop) << std::endl;
  // copy results back to host
  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
  cudaErrorCheck();
  checkResults(h_in, h_out);
  //----------------------------------------------------------

  // deallocate device memory
  cudaFree(d_in);
  cudaFree(d_out);
  cudaErrorCheck();
  //=====================================================
  // Evaluate total time of execution with just the CPU.
  //=====================================================
  newline();
  std::cout << "Running stencil with the CPU.\n";
  start_timer(&start);
  // Use checkResults to time CPU version of the stencil with False flag.
  checkResults(h_in, h_out, False);
  std::cout << "Elapsed time: " << stop_timer(&start, &stop) << std::endl;
  //=====================================================

  // deallocate host memory
  free(h_in);
  free(h_out);

  return 0;
}
