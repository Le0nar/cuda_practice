#include <vector>
#include <iostream>
#include <numeric>

typedef float real_t;

const size_t BLOCK_SIZE = 16;

__global__ void partial_dot(const real_t* v1, const real_t* v2, real_t* out, int N) {
    __shared__ real_t cache[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    cache[threadIdx.x] = 0.f;
    while (i < N) {
        cache[threadIdx.x] += v1[i] * v2[i];
        i += gridDim.x * blockDim.x;
    }
    __syncthreads(); // required because later on the current thread is accessing
    // data written by another thread
    i = BLOCK_SIZE / 2;
    while (i > 0) {
        if (threadIdx.x < i) cache[threadIdx.x] += cache[threadIdx.x + i];
        __syncthreads();
        i /= 2; //not sure bitwise operations are actually faster
    }

    if (threadIdx.x == 0) out[blockIdx.x] = cache[0];
}

real_t dot(const real_t* v1, const real_t* v2, int N) {
    real_t s = 0;
    for (int i = 0; i != N; ++i) {
        s += v1[i] * v2[i];
    }
    return s;
}


__global__ void init_vector(real_t* v, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < N) {
        v[i] = 1.0f;//real_t( i ) / 1000000.f;
        i += gridDim.x * blockDim.x;
    }
}


//------------------------------------------------------------------------------
int main(int argc, char** argv) {

    const size_t ARRAY_SIZE = 1024;//1024 * 1024; //1Mi elements
    const int BLOCKS = 64;//512;
    const int THREADS_PER_BLOCK = BLOCK_SIZE;//256; // total threads = 512 x 256 = 128ki threads;                                   // each thread spans 8 array elements  
    const size_t SIZE = ARRAY_SIZE * sizeof(real_t);

    // device storage
    real_t* dev_v1 = 0; // vector 1
    real_t* dev_v2 = 0; // vector 2
    real_t* dev_vout = 0; // partial redution = number of blocks
    cudaMalloc(&dev_v1, SIZE);
    cudaMalloc(&dev_v2, SIZE);
    cudaMalloc(&dev_vout, BLOCKS * sizeof(real_t));

    // host storage
    std::vector< real_t > host_v1(ARRAY_SIZE);
    std::vector< real_t > host_v2(ARRAY_SIZE);
    std::vector< real_t > host_vout(BLOCKS);

    // initialize vector 1 with kernel; much faster than using for loops on the cpu
    init_vector << < 1024, 256 >> > (dev_v1, ARRAY_SIZE);
    cudaMemcpy(&host_v1[0], dev_v1, SIZE, cudaMemcpyDeviceToHost);
    // initialize vector 2 with kernel; much faster than using for loops on the cpu
    init_vector << < 1024, 256 >> > (dev_v2, ARRAY_SIZE);
    cudaMemcpy(&host_v2[0], dev_v2, SIZE, cudaMemcpyDeviceToHost);

    // execute kernel
    partial_dot << <BLOCKS, THREADS_PER_BLOCK >> > (dev_v1, dev_v2, dev_vout, ARRAY_SIZE);

    // copy output data from device(gpu) to host(cpu)
    cudaMemcpy(&host_vout[0], dev_vout, BLOCKS * sizeof(real_t), cudaMemcpyDeviceToHost);

    // print dot product by summing up the partially reduced vectors
    std::cout << "GPU: " << std::accumulate(host_vout.begin(), host_vout.end(), real_t(0)) << std::endl;

    // print dot product on cpu
    std::cout << "CPU: " << dot(&host_v1[0], &host_v2[0], ARRAY_SIZE) << std::endl;

    // free memory
    cudaFree(dev_v1);
    cudaFree(dev_v2);
    cudaFree(dev_vout);

    return 0;
}