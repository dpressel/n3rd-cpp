#include "n3rd/GPUOps.h"
#include <stdio.h>

const float EPS = 1e-6;
const int TILE_DIM = 16;
const int BLOCK_ROWS = 16;



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
        auto delta = -(biasGrads[i] * eta);
        biasParams[i] += delta;
        biasGrads[i] = 0;
    }

}


// Grad should be zerod out!
__global__ void devMaxPooling2Backward(double* grad, int* origin, double* chainGrad, int kL, int iH, int iW, int oH, int oW, int dh, int dw)
{

    int oi = blockDim.x * blockIdx.x + threadIdx.x;
    int oj = blockDim.y * blockIdx.y + threadIdx.y;

    if (oj < oW && oi < oH)
    {
        for (int l = 0; l < kL; ++l)
        {

            int outAddr = (l * oH + oi) * oW + oj;
            int inAddr = origin[outAddr];
            grad[inAddr] = chainGrad[outAddr];
        }
    }
}

__global__ void devMaxPooling2(double *idata, int* origin, double* odata, int kL, int iH, int iW, int oH, int oW, int dh, int dw)
{

    int oi = blockDim.x * blockIdx.x + threadIdx.x;
    int oj = blockDim.y * blockIdx.y + threadIdx.y;

    int iiStart = min(oi * dh, iH);
    int iiEnd = min((oi + 1) * dh, iH);
    int ijStart = min(oj * dw, iW);
    int ijEnd = min((oj + 1) * dw, iW);

    if (oj < oW && oi < oH)
    {
        for (int l = 0; l < kL; ++l)
        {
            double mx = -1000000;
            double imx = 0;
            int outAddr = (l * oH + oi) * oW + oj;
            for (int ii = iiStart; ii < iiEnd; ++ii)
            {
                for (int ij = ijStart; ij < ijEnd; ++ij)
                {
                    int inAddr = (l * iH + ii) * iW + ij;
                    double zi = idata[inAddr];

                    if (mx < zi)
                    {
                        mx = zi;
                        imx = inAddr;
                    }
                }
            }
            odata[outAddr] = mx;
            origin[outAddr] = imx;
        }
    }

}

__global__ void devAddBias2(double* x, double *bias, int nK, int oH, int oW)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < oW && i < oH)
    {
        for (int l = 0; l < nK; ++l)
        {
           x[(l * oH + i) * oW + j] = bias[l];
        }
    }
}

__global__ void devGradBias2(double* biasGrad, double* grad, int nK, int oH, int oW)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < oW && i < oH)
    {
        for (int l = 0; l < nK; ++l)
        {
            biasGrad[l] += grad[(l * oH + i) * oW + j];
        }
    }
}

__global__ void devUnwrapInput2(double *x, double* unwrappedInput, const int kL, int kH, int kW, int iH, int iW)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int oH = iH - kH + 1;
    int oW = iW - kW + 1;

    if (j < oW && i < oH)
    {
        for (int k = 0; k < kL; ++k)
        {
            for (int m = 0; m < kH; ++m)
            {
                for (int n = 0; n < kW; ++n)
                {
                    int offset = (k * iH + i + m) * iW + j + n;
                    int z = (((k * kH + m) * kW + n) * oH + i) * oW + j;
                    unwrappedInput[z] = x[offset];
                }

            }
        }
    }

}


__global__ void devWrapGrad2(double *unwrapped, double* grads, const int kL, int kH, int kW, int iH, int iW)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int oH = iH - kH + 1;
    int oW = iW - kW + 1;

    if (j < oW && i < oH)
    {
        for (int k = 0; k < kL; ++k)
        {
            for (int m = 0; m < kH; ++m)
            {
                for (int n = 0; n < kW; ++n)
                {
                    int offset = (k * iH + i + m) * iW + j + n;
                    int z = (((k * kH + m) * kW + n) * oH + i) * oW + j;
                    grads[offset] += unwrapped[z];
                }

            }
        }
    }
}

__global__ void devTranspose(double *odata, double *idata, int height, int width)
{
    __shared__ double tile[TILE_DIM][TILE_DIM+1];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
    {
        if (xIndex < width && (yIndex + i) < height)
        {
            tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
        }
    }

    __syncthreads();

    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
    {
        if ((yIndex + i) < width && xIndex < height)
        {
            odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
        }
    }
}

// TODO: Not optimized at all!
__global__ void devWrapGrad(double* unwrapped, double* grads, int kW, int oT)
{
    const int k = blockIdx.x;
    const int iT = oT + kW - 1;

    // iT = threadDims.x
    for (int m = 0; m < kW; ++m)
    {

        if (threadIdx.x < oT)
        {
            int inputIdx = (k * kW + m) * oT + threadIdx.x;
            int outputIdx = k * iT + threadIdx.x + m;
            grads[outputIdx] += unwrapped[inputIdx];
        }

    }
}

/*
 *  int offset = (k * iH + i + m) * iW + j + n;
                        grads[offset] += unwrapped[z];
                        z++;
 */



__global__ void devUnwrapInput(double *x, double* unwrappedInput, int kW, int iT)
{
    const int k = blockIdx.x;

    for (int m = 0; m < kW; ++m)
    {
        // We are expecting the number of threads to be less than the length of the signal
        // this makes sense for some 1D data, like NLP, but probably not others
        if (threadIdx.x < iT)
        {
            const int oT = iT - kW + 1;
            int offset = k * iT + threadIdx.x + m;
            int n = (k * kW + m) * oT + threadIdx.x;
            unwrappedInput[n] = x[offset];
        }
    }
}

void n3rdgTranspose(double *outputMx, double* inputMx, int height, int width)
{
    dim3 grid((int)ceil(width/(double)TILE_DIM), (int)ceil(height/(double)TILE_DIM));
    dim3 threads(TILE_DIM,BLOCK_ROWS);
    devTranspose<<<grid, threads>>>(outputMx, inputMx, height, width);
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


// We are putting a hard limit on N here, notice its capped at 256!
void n3rdgWrapGrad(double* unwrapped, double* grads, int kL, int kW, int oT)
{
    int blocksPerGrid = kL;
    devWrapGrad<<<blocksPerGrid, 256>>>(unwrapped, grads, kW, oT);
    cudaDeviceSynchronize();
}


void n3rdgUnwrapInput(double *x, double* unwrapped, int kL, int kW, int iT)
{
    int blocksPerGrid = kL;
    devUnwrapInput<<<blocksPerGrid, 256>>>(x, unwrapped, kW, iT);
    cudaDeviceSynchronize();
}


void n3rdgAddBias2(double* x, double *bias, int nK, int oH, int oW)
{
    dim3 threadsPerBlock;
    threadsPerBlock.x = 16;
    threadsPerBlock.y = 16;
    dim3 blocksPerGrid;

    blocksPerGrid.x = (oH + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocksPerGrid.y = (oW + threadsPerBlock.y - 1) / threadsPerBlock.y;

    devAddBias2<<<blocksPerGrid, threadsPerBlock>>>(x, bias, nK, oH, oW);
    cudaDeviceSynchronize();
}


void n3rdgBiasGrad2(double* biasGrad, double *grad, int nK, int oH, int oW)
{
    dim3 threadsPerBlock;
    threadsPerBlock.x = 16;
    threadsPerBlock.y = 16;
    dim3 blocksPerGrid;

    blocksPerGrid.x = (oH + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocksPerGrid.y = (oW + threadsPerBlock.y - 1) / threadsPerBlock.y;

    devGradBias2<<<blocksPerGrid, threadsPerBlock>>>(biasGrad, grad, nK, oH, oW);
    cudaDeviceSynchronize();
}



void n3rdgUnwrapInput2(double* x, double* unwrapped, int kL, int kH, int kW, int iH, int iW)
{

    dim3 threadsPerBlock;
    threadsPerBlock.x = 16;
    threadsPerBlock.y = 16;
    dim3 blocksPerGrid;

    blocksPerGrid.x = (iH + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocksPerGrid.y = (iW + threadsPerBlock.y - 1) / threadsPerBlock.y;

    devUnwrapInput2<<<blocksPerGrid, threadsPerBlock>>>(x, unwrapped, kL, kH, kW, iH, iW);
    cudaDeviceSynchronize();
}


void n3rdgWrapGrad2(double *unwrapped, double* grads, const int kL, int kH, int kW, int iH, int iW)
{
    dim3 threadsPerBlock;
    threadsPerBlock.x = 16;
    threadsPerBlock.y = 16;
    dim3 blocksPerGrid;

    blocksPerGrid.x = (iH + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocksPerGrid.y = (iW + threadsPerBlock.y - 1) / threadsPerBlock.y;
    devWrapGrad2<<<blocksPerGrid, threadsPerBlock>>>(unwrapped, grads, kL, kH, kW, iH, iW);
    cudaDeviceSynchronize();
}


// We are putting a hard limit on N here, notice its capped at 256!
void n3rdgMaxPooling2Forward(double* dX, int* dOrigin, double* dOutput, int kL, int iH, int iW, int oH, int oW, int dh, int dw)
{
    dim3 threadsPerBlock;
    threadsPerBlock.x = 16;
    threadsPerBlock.y = 16;
    dim3 blocksPerGrid;

    blocksPerGrid.x = (oH + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocksPerGrid.y = (oW + threadsPerBlock.y - 1) / threadsPerBlock.y;

    devMaxPooling2<<<blocksPerGrid, threadsPerBlock>>>(dX, dOrigin, dOutput, kL, iH, iW, oH, oW, dh, dw);
    cudaDeviceSynchronize();
}


void n3rdgMaxPooling2Backward(double* dGrad, int* dOrigin, double* dChainGrad, int kL, int iH, int iW, int oH, int oW, int dh, int dw)
{

    dim3 threadsPerBlock;
    threadsPerBlock.x = 16;
    threadsPerBlock.y = 16;
    dim3 blocksPerGrid;

    blocksPerGrid.x = (oH + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocksPerGrid.y = (oW + threadsPerBlock.y - 1) / threadsPerBlock.y;

    devMaxPooling2Backward<<<blocksPerGrid, threadsPerBlock>>>(dGrad, dOrigin, dChainGrad, kL, iH, iW, oH, oW, dh, dw);
}

// We are putting a hard limit on N here, notice its capped at 256!
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