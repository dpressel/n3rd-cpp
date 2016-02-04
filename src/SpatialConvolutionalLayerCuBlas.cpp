#include "n3rd/SpatialConvolutionalLayerCuBlas.h"

using namespace n3rd;
using namespace sgdtk;

SpatialConvolutionalLayerCuBlas::SpatialConvolutionalLayerCuBlas(int nK, int kH, int kW, std::vector<int> inputDims)
{
    this->nK = nK;
    int inputDimsSz = inputDims.size();
    this->kL = inputDimsSz == 3 ? inputDims[0]: 1;
    this->kH = kH;
    this->kW = kW;
    this->iH = inputDimsSz == 3 ? inputDims[1]: inputDims[0];
    this->iW = inputDimsSz == 3 ? inputDims[2]: inputDims[1];
    // For each kernel, randomly initialize all weights
    output.resize({nK, iH - kH + 1, iW - kW + 1});
    grads.resize({kL, iH, iW});



    // The unwrapped input is tap-unrolled with a width that is kH * kW * nK, and a height that is the number of lags
    dUnwrappedInput.resize({output.dims[1]*output.dims[2], kH * kW * kL});
    dUnwrappedGradInput.resize(dUnwrappedInput.dims);
    weights.resize({kL * kH * kW, nK});
    weightAccum.resize({kL * kH * kW, nK});
    gradsW.resize({kL * kH * kW, nK});

    // For now, let this just be a pointer to input
    ////input.resize(inputDims);
    biases.resize({nK}, 0);
    biasGrads.resize({nK}, 0);
    grads.resize(inputDims);
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    double stdv = 1. / std::sqrt(grads.dims[1] * grads.dims[2]);

    double stdv2 = stdv * 2;

    // For each kernel, randomly initialize all weights
    Tensor cWeights(weights.dims);
    for (int i = 0; i < cWeights.size(); ++i)
    {

        double number = distribution(generator);
        double d = number * stdv2 - stdv;
        cWeights[i] = d;
    }
    weights.fromCPU(cWeights, false);

}

// TODO: replace with nerdg function
void SpatialConvolutionalLayerCuBlas::unwrapInput(const sgdtk::CudaTensor& x)
{

    Tensor xT;
    Tensor unwrappedInput(dUnwrappedInput.dims);

    x.toCPU(xT);
    int z = 0;

    const int oH = iH - kH + 1;
    const int oW = iW - kW + 1;

    for (int k = 0; k < kL; ++k)
    {

        for (int m = 0; m < kH; ++m)
        {
            for (int n = 0; n < kW; ++n)
            {
                // Cycle all taps at each kernel?
                for (int i = 0; i < oH; ++i)
                {
                    for (int j = 0; j < oW; ++j)
                    {
                        // This is then image(k, i + m, j + n)
                        int offset = (k * iH + i + m) * iW + j + n;
                        unwrappedInput[z] = xT[offset];
                        ++z;
                    }
                }
            }
        }
    }
    dUnwrappedInput.fromCPU(unwrappedInput);

}

// TODO: replace with nerdg
void SpatialConvolutionalLayerCuBlas::wrapGrad(sgdtk::CudaTensor& dUnwrapped)
{

    sgdtk::Tensor unwrapped;

    dUnwrapped.toCPU(unwrapped);

    sgdtk::Tensor cGrads(grads.dims);

    const int oH = iH - kH + 1;
    const int oW = iW - kW + 1;

    // In the blas case, we need to write in column major, which means write down one lag, then move up to the next
    int z = 0;
    for (int k = 0; k < kL; ++k)
    {
        for (int m = 0; m < kH; ++m)
        {
            for (int n = 0; n < kW; ++n)
            {
                for (int i = 0; i < oH; ++i)
                {
                    for (int j = 0; j < oW; ++j)
                    {
                        int offset = (k * iH + i + m) * iW + j + n;
                        cGrads[offset] += unwrapped[z];
                        z++;
                    }
                }

            }
        }
    }
    grads.fromCPU(cGrads);


}

sgdtk::TensorI& SpatialConvolutionalLayerCuBlas::forward(const sgdtk::TensorI& z)
{
    const sgdtk::CudaTensor& input = (const sgdtk::CudaTensor&)z;
    grads.constant(0.);

    n3rdgUnwrapInput2(input.d, dUnwrappedInput.d, kL, kH, kW, iH, iW);

    const int oH = iH - kH + 1;
    const int oW = iW - kW + 1;

    n3rdgAddBias2(output.d, biases.d, nK, oH, oW);

    double alpha = 1.0;
    double beta = 1.0;

    TRY_CUBLAS(cublasDgemm_v2(sgdtk::Globals::gBlasHandle, CUBLAS_OP_N, CUBLAS_OP_N, dUnwrappedInput.dims[0],
                weights.dims[1],
                dUnwrappedInput.dims[1], &alpha,
                dUnwrappedInput.d,
                dUnwrappedInput.dims[0],
                weights.d, weights.dims[0], &beta,
                output.d, dUnwrappedInput.dims[0]));

    return output;

}

sgdtk::TensorI& SpatialConvolutionalLayerCuBlas::backward(sgdtk::TensorI& chainGrad, double y)
{

    sgdtk::CudaTensor& chainGradT = (sgdtk::CudaTensor&) chainGrad;


    const int oH = iH - kH + 1;
    const int oW = iW - kW + 1;

    const int zpH = iH + kH - 1;
    const int zpW = iW + kW - 1;

    chainGradT.reshape({nK, 1, oH * oW});

    int m = chainGradT.dims[2];
    int k = nK;
    int n = weights.dims[0];

    double alpha = 1.0;
    double beta = 0.0;

    TRY_CUBLAS(cublasDgemm_v2(sgdtk::Globals::gBlasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, chainGradT.d, m, weights.d, n, &beta, dUnwrappedGradInput.d, m));

    m = dUnwrappedInput.dims[1];
    k = dUnwrappedInput.dims[0];
    n = nK;


    TRY_CUBLAS(cublasDgemm_v2(sgdtk::Globals::gBlasHandle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, dUnwrappedInput.d, k, chainGradT.d, k, &beta, gradsW.d, m));

    n3rdgBiasGrad2(biasGrads.d, chainGradT.d, nK, oH, oW);
    n3rdgWrapGrad2(dUnwrappedGradInput.d, grads.d, kL, kH, kW, iH, iW);

    return grads;

}