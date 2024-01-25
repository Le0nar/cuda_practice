#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <omp.h>

// TODO: поменять на #define KS 100000
#define INTERVALS 1000000

// Max number of threads per block
#define THREADS 512
#define BLOCKS 64

// Synchronous error checking call. Enable with nvcc -DDEBUG
inline void checkCUDAError(const char *fileName, const int line)
{ 
	#ifdef DEBUG 
		cudaThreadSynchronize();
		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess) 
		{
			printf("Error at %s: line %i: %s\n", fileName, line, cudaGetErrorString(error));
			exit(-1); 
		}
	#endif
}


__global__ void integrate(float *sum, float step, int threads, int blocks)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	for (int i = idx; i < INTERVALS; i+=threads*blocks)
	{
		float x = (i+0.5f) * step;
		sum[idx] += 4.0f / (1.0f+ x*x);
	}
}

int main()
{
	const float PI25DT = 3.141592653589793238462643;
	int deviceCount = 0;

	printf("Starting...");
    
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        return 1;
    }

	deviceCount == 0 ? printf("There are no available CUDA device(s)\n") : printf("%d CUDA Capable device(s) detected\n", deviceCount);

	/*--------- Simple Kernel ---------*/

	// TODO: поменять на 10, 10
	int threads = 8, blocks = 30;
	dim3 block(threads);
	dim3 grid(blocks);

	
	float *sum_h, *sum_d;
	float step = 1.0f / INTERVALS;
	float piSimple = 0;
	
	// Allocate host memory
	sum_h = (float *)malloc(blocks*threads*sizeof(float));	

	// Allocate device memory
	cudaMalloc((void **) &sum_d, blocks*threads*sizeof(float));

	// CUDA events needed to measure execution time
	cudaEvent_t start, stop;
	float gpuTime, optimizedGpuTime;

	// Start timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	printf("\nCalculating Pi using simple GPU kernel over %i intervals...\n", (int)INTERVALS);
	integrate<<<grid, block>>>(sum_d, step, threads, blocks);	
	checkCUDAError(__FILE__, __LINE__);

	// Stop timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	// Retrieve result from device
	cudaMemcpy(sum_h, sum_d, blocks*threads*sizeof(float), cudaMemcpyDeviceToHost);
	
	// Sum result on host
	for (int i=0;i < threads*blocks; i++)
	{
		piSimple += sum_h[i];	
	}
	
	piSimple *= step;
	printf("Pi is approximately %.16f, Error: %.16f\n", piSimple, fabs(piSimple - PI25DT));

	free(sum_h);
	cudaFree(sum_d);

	return 0;
}
