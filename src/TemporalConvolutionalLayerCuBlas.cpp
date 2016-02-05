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

sgdtk::TensorI& TemporalConvolutionalLayerCuBlas::forward(const sgdtk::TensorI& z)
{
    const sgdtk::CudaTensor& zT = (const sgdtk::CudaTensor&)z;

    numFrames = z.size() / kL;
    grads.resize({kL, 1, numFrames});
    grads.zeros();
    const int oT = numFrames - kW + 1;
    output.resize({nK, 1, oT});
    dUnwrappedInput.resize({oT, kW * kL});
    n3rdgUnwrapInput(zT.d, dUnwrappedInput.d, kL, kW, numFrames);

    double alpha = 1.0;
    double beta = 0.0;
    TRY_CUBLAS(cublasDgemm_v2(sgdtk::Globals::gBlasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                   dUnwrappedInput.dims[0], weights.dims[1], dUnwrappedInput.dims[1], &alpha, dUnwrappedInput.d, dUnwrappedInput.dims[0],
                              weights.d, weights.dims[0], &beta, output.d, oT));

    //output.add(biases);
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