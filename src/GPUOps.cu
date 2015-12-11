#include "n3rd/GPUOps.h"
#include <stdio.h>

const float EPS = 1e-6;
__global__ void devTanh(double* x, double* output, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        output[i] = tanh(x[i]);
    }
}


__global__ void devDtanh(double* chainGrad, double* output, double* grads, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        grads[i] = chainGrad[i] * (1 - output[i] * output[i]);
    }

}

//grads[i] = chainGrad[i] * (1 - output[i]) * output[i];
__device__ double devSigmoid(double x)
{
    return 1.0 / (1.0 + expf(-x));
}
__global__ void devSigmoid(double* x, double* output, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        output[i] = devSigmoid(x[i]);
    }
}

__global__ void devDsigmoid(double* chainGrad, double* output, double* grads, int N)
{

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        grads[i] = chainGrad[i] * (1 - output[i]) * output[i];
    }
}

// When we run this, we are going to have 256 threads per block, which means
// threadIdx.x = 0..255
// blockDim.x = 256
// blockIdx.x = numFeatureMapsIn (kL)
// A reminder that our memory on the device is organized as follows
// data(numFeatureMapsIn, 1, signalLength)
// The way I have set this up, the numFeatureMapsIn is actually number of blocks we are going to
// run (blockIdx.x above)
// In the end, we should get a signal that looks like
// output(numFeatureMapsIn, 1, 1)
// TODO: switch to reduction 6
__global__ void devMaxOverTime(double *g_idata, double *g_odata, int* g_oidx, unsigned int n)
{

    // The thread index is unique in the block.
    unsigned int tid = threadIdx.x;
    // A global index of input.  We need ()
    unsigned int i = blockIdx.x * n + threadIdx.x;

    __shared__ int sidx[256];
    __shared__ double sdata[256];

    sidx[tid] = tid;
    sdata[tid] = (tid < n) ? g_idata[i] : -100000;


    //printf("Block idx: %d, sdata: %f\n", blockIdx.x, sdata[tid]);
    __syncthreads();



    // For each thread, we are going to get one pixel from our area, and one from 2x our area
    // on the first pass, and the number of threads active is cut in half each time
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1)
    {
        // We are throwing out half of our threads
        if (tid < s)
        {
            if (sdata[tid + s] > sdata[tid])
            {
                sidx[tid] = sidx[tid + s];
                sdata[tid] = sdata[tid + s];
            }

        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        //printf("Block idx: %d, sdata: %f\n", blockIdx.x, sdata[0]);
        g_odata[blockIdx.x] = sdata[0];
        g_oidx[blockIdx.x] = blockIdx.x * n + sidx[0];
    }
}

__global__ void devMaxOverTimeBackward(double* dChainGrad, int* dOrigin, double* dGrads, int M)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < M)
    {
        int inAddr = dOrigin[i];
        dGrads[inAddr] = dChainGrad[i];
    }
}
// Perform adagrad weight updates
__global__ void devAdagradWeightUpdates(double * weights, double* weightGrads, double* gg, float eta, float lambda, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        gg[i] += weightGrads[i] * weightGrads[i];
        double etaThis = eta * rsqrt(gg[i] + EPS);
        double delta = -etaThis * weightGrads[i];
        weights[i] *= (1. - eta * lambda);
        weights[i] += delta;
        weightGrads[i] = 0;
    }
}

__global__ void devBiasUpdates(double* biasParams, double* biasGrads, float eta, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        auto delta = -(biasGrads[i] * eta);// * 0.01; // last number is total fudge
        biasParams[i] += delta;
        biasGrads[i] = 0;
    }

}

void n3rdgAdagradWeightUpdates(double *weights, double *weightGrads, double *gg, float eta, float lambda, int N)
{

    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    devAdagradWeightUpdates<<<blocksPerGrid, threadsPerBlock>>>(weights, weightGrads, gg, eta, lambda, N);
    cudaDeviceSynchronize();

}

void n3rdgBiasUpdates(double* biasParams, double* biasGrads, float eta, int N)
{
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    devBiasUpdates<<<blocksPerGrid, threadsPerBlock>>>(biasParams, biasGrads, eta, N);
    cudaDeviceSynchronize();
}

void n3rdgMaxOverTimeForward(double* dX, double* dOutput, int* dIdx, int M, int N)
{

    int blocksPerGrid = M;
    devMaxOverTime<<<blocksPerGrid, 256>>>(dX, dOutput, dIdx, N);
    cudaDeviceSynchronize();
}

void n3rdgMaxOverTimeBackward(double* dChainGrad, int *dOrigin, double* dGrads, int M)
{
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (M + threadsPerBlock - 1) / threadsPerBlock;
    devMaxOverTimeBackward<<<blocksPerGrid, threadsPerBlock>>>(dChainGrad, dOrigin, dGrads, M);
}
void n3rdgTanhForward(double* dX, double* dOutput, int N)
{
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    devTanh<<<blocksPerGrid, threadsPerBlock>>>(dX, dOutput, N);
    cudaDeviceSynchronize();
}

void n3rdgTanhBackward(double* dChainGrad, double* dOutput, double* dGrads, int N)
{
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    devDtanh<<<blocksPerGrid, threadsPerBlock>>>(dChainGrad, dOutput, dGrads, N);
    cudaDeviceSynchronize();
}