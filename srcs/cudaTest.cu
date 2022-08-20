#include "cudaTest.cuh"

#include <chrono>
#include <iostream>
#include <time.h>
#include <ctime>
#include <assert.h>

// CUDA include
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

void initWith(float num, float* a, int N)
{
    for (int i = 0; i < N; ++i)
    {
        a[i] = num;
    }
}

void addVectorsInto(float* result, float* a, float* b, int N)
{
    for (int i = 0; i < N; ++i)
    {
        result[i] = a[i] + b[i];
    }
}

__global__ void addVectorsInto_cuda(float* result, float* a, float* b, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride)
    {
        result[i] = a[i] + b[i];
    }
}

void checkElementsAre(float target, float* array, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (array[i] != target)
        {
            printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
            exit(1);
        }
    }
    printf("SUCCESS! All values added correctly.\n");
}

void Test()
{
    int N = 2<<20;

    size_t size = N * sizeof(float);

    float* a;
    float* b;
    float* c;

    a = new float[N];
    b = new float[N];
    c = new float[N];

    initWith(3, a, N);
    initWith(4, b, N);
    initWith(0, c, N);

    std::cout << "Run on CPU" << std::endl;
    auto currentMillisecTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    addVectorsInto(c, a, b, N);
    auto DeltaMillisecTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - currentMillisecTime;
    checkElementsAre(7, c, N);
    std::cout << "Delta time : " << DeltaMillisecTime << " (ms)" << std::endl;

    delete a;
    delete b;
    delete c;

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    initWith(3, a, N);
    initWith(4, b, N);
    initWith(0, c, N);

    size_t threads_per_block = 256;
    size_t NumofBlock = (N + threads_per_block - 1) / (threads_per_block);

    std::cout << "Run on GPU" << std::endl;
    currentMillisecTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    addVectorsInto_cuda<<<NumofBlock , threads_per_block >>>(c, a, b, N);

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    DeltaMillisecTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - currentMillisecTime;
    checkElementsAre(7, c, N);
    std::cout << "Delta time : " << DeltaMillisecTime << " (ms)" << std::endl;

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}