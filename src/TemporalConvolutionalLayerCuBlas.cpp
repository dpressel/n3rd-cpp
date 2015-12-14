#include "n3rd/TemporalConvolutionalLayerCuBlas.h"
#include <cassert>

using namespace n3rd;

TemporalConvolutionalLayerCuBlas::TemporalConvolutionalLayerCuBlas(int nK, int kL, int kW)
{
    this->nK = nK;
    this->kL = kL;
    this->kW = kW;

    sgdtk::Tensor cpuWeights({kL * kW, nK});
    weights.resize(cpuWeights.dims);
    weightAccum.resize(weights.dims, 0);
    ////dWeights.resize(weights.dims);
    ////dWeightGrads.resize(weights.dims);
    gradsW.resize(weights.dims);
    biases.resize({nK}, 0);
    biasGrads.resize({nK}, 0);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double stdv = 1. / std::sqrt(6. / 28.);
    double stdv2 = stdv * 2;

    for (int i = 0, sz = cpuWeights.size(); i < sz; ++i)
    // For each kernel, randomly initialize all weights
    {
        double number = distribution(generator);
        double d = number * stdv2 - stdv;
        cpuWeights[i] = d;

    }

    weights.fromCPU(cpuWeights, false);
}


void TemporalConvolutionalLayerCuBlas::unwrapInput(const sgdtk::CudaTensor& dX)
{
    // FIXME!!!!

    sgdtk::Tensor x(dX.dims);
    dX.toCPU(x, false);
    const int oT = numFrames - kW + 1;

    dUnwrappedInput.resize({oT, kW * kL});


    sgdtk::Tensor unwrappedInput(dUnwrappedInput.dims);
    int n = 0;


    for (int k = 0; k < kL; ++k)
    {

        for (int m = 0; m < kW; ++m)
        {
            for (int i = 0; i < oT; ++i)
            {

                int offset = k * numFrames + i + m;
                unwrappedInput[n] = x[offset];
                ++n;
            }
        }
    }
    dUnwrappedInput.fromCPU(unwrappedInput, false);
}

void TemporalConvolutionalLayerCuBlas::wrapGrad(const sgdtk::CudaTensor& unwrapped)
{

    sgdtk::Tensor unwrappedT(unwrapped.dims);
    unwrapped.toCPU(unwrappedT, false);

    const int oT = unwrapped.dims[0];
    const int iT = oT + kW - 1;
    assert(iT == grads.dims[2]);
    const int kL = grads.dims[0];
    const int embedSz = grads.dims[1];
    assert(1 == embedSz);

    sgdtk::Tensor gradsT(grads.dims);

    // In the blas case, we need to write in column major, which means write down one lag, then move up to the next
    int n = 0;
    // 1 .. 100
    // n = (k * kW + m) * oT + i;
    for (int k = 0; k < kL; ++k)
    {
        // 7
        for (int m = 0; m < kW; ++m)
        {
            // 1 - 256
            for (int i = 0; i < oT; ++i)
            {
                int offset = k * iT + i + m;
                    // x(kL, iT, embedSz)
                gradsT[offset] += unwrappedT[n];
                n++;
            }
        }
    }

    grads.fromCPU(gradsT, false);

}

sgdtk::TensorI& TemporalConvolutionalLayerCuBlas::forward(const sgdtk::TensorI& z)
{
    const sgdtk::CudaTensor& zT = (const sgdtk::CudaTensor&)z;
    // For convolutions, we should assume that our VectorN is truly a matrix
    // and the usual math applies

    numFrames = z.size() / kL;
    grads.resize({kL, 1, numFrames});

    const int oT = numFrames - kW + 1;

    output.resize({nK, 1, oT});
    ///////unwrapInput(zT);

    dUnwrappedInput.resize({oT, kW * kL});
    n3rdgUnwrapInput(zT.d, dUnwrappedInput.d, kL, kW, numFrames);


    double alpha = 1.0;
    double beta = 0.0;
    TRY_CUBLAS(cublasDgemm_v2(sgdtk::Globals::gBlasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                   dUnwrappedInput.dims[0], weights.dims[1], dUnwrappedInput.dims[1], &alpha, dUnwrappedInput.d, dUnwrappedInput.dims[0], weights.d, weights.dims[0], &beta, output.d, oT));

    return output;

}

sgdtk::TensorI& TemporalConvolutionalLayerCuBlas::backward(sgdtk::TensorI &chainGrad, double y)
{
    const sgdtk::CudaTensor& chainGradT = (const sgdtk::CudaTensor&)chainGrad;

    const int oT = numFrames - kW + 1;
    std::vector<int> outputDims = { nK, 1, oT };
    chainGrad.reshape(outputDims);

    std::vector<int> unwrappedGradDims = {oT, kW * kL};
    //sgdtk::Tensor unwrappedGradInput(unwrappedGradDims);
    sgdtk::CudaTensor dUnwrappedGradInput(unwrappedGradDims);

    int m = oT;
    int k = nK;
    int n = weights.dims[0];

    double alpha = 1.0;
    double beta = 0.0;

    TRY_CUBLAS(cublasDgemm_v2(sgdtk::Globals::gBlasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
                chainGradT.d, m, weights.d, n, &beta,
                dUnwrappedGradInput.d, m));

    m = dUnwrappedInput.dims[1];
    k = dUnwrappedInput.dims[0];
    n = nK;

    TRY_CUBLAS(cublasDgemm_v2(sgdtk::Globals::gBlasHandle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, dUnwrappedInput.d, k, chainGradT.d, k, &beta, gradsW.d, m));

        // We need to update gradsW, which are (kL * embeddingSize) * kW x (nK * embeddingSize);
        /*
        int stride = convOutputSz * embedSz;
        for (int l = 0; l < nK; ++l)
        {
            for (int i = 0; i < stride; ++i)
            {
                this.biasGrads[l] += chainGradX[l * stride + i];
            }
            this.biasGrads[l] /= embedSz;
        }*/




    ////// VALIDATE
    //////wrapGrad(dUnwrappedGradInput);
    //////sgdtk::CudaTensor gg(grads.dims);

    n3rdgWrapGrad(dUnwrappedGradInput.d, grads.d, nK, kW, oT);


    return grads;
}