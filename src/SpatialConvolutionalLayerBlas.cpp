#include "n3rd/SpatialConvolutionalLayerBlas.h"

using namespace n3rd;
using namespace sgdtk;

SpatialConvolutionalLayerBlas::SpatialConvolutionalLayerBlas(int nK, int kH, int kW, std::vector<int> inputDims)
{



    this->nK = nK;
    int inputDimsSz = inputDims.size();
    this->kL = inputDimsSz == 3? inputDims[0]: 1;
    this->kH = kH;
    this->kW = kW;
    this->iH = inputDimsSz == 3 ? inputDims[1]: inputDims[0];
    this->iW = inputDimsSz == 3 ? inputDims[2]: inputDims[1];
    // For each kernel, randomly initialize all weights
    output.resize({nK, iH - kH + 1, iW - kW + 1});
    grads.resize({kL, iH, iW});



    // The unwrapped input is tap-unrolled with a width that is kH * kW * nK, and a height that is the number of lags
    unwrappedInput.resize({output.dims[1]*output.dims[2], kH * kW * kL});
    unwrappedGradInput.resize(unwrappedInput.dims);
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
    for (int i = 0; i < weights.size(); ++i)
    {

        double number = distribution(generator);
        double d = number * stdv2 - stdv;
        weights[i] = d;
    }

}

void SpatialConvolutionalLayerBlas::unwrapInput(const sgdtk::Tensor& x)
{


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
                        unwrappedInput[z] = x[offset];
                        ++z;
                    }
                }
            }
        }
    }

}

void SpatialConvolutionalLayerBlas::wrapGrad(sgdtk::Tensor& unwrapped)
{

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
                        grads[offset] += unwrapped[z];
                        z++;
                    }
                }

            }
        }
    }



}

sgdtk::TensorI& SpatialConvolutionalLayerBlas::forward(const sgdtk::TensorI& z)
{

    grads.constant(0.);

    const sgdtk::Tensor& input = (const sgdtk::Tensor&)z;

    unwrapInput(input);


    const int oH = iH - kH + 1;
    const int oW = iW - kW + 1;
    for (int l = 0; l < nK; ++l)
    {
        for (int i = 0; i < oH; ++i)
        {
            for (int j = 0; j < oW; ++j)
            {
                output[(l * oH + i) * oW + j] = biases[l];
            }
        }
    }

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, unwrappedInput.dims[0],
                weights.dims[1],
                unwrappedInput.dims[1], 1.0,
                &(unwrappedInput.d[0]),
                unwrappedInput.dims[0],
                &(weights.d[0]), weights.dims[0], 1.0,
                &(output.d[0]), unwrappedInput.dims[0]);

    return output;

}


sgdtk::TensorI& SpatialConvolutionalLayerBlas::backward(sgdtk::TensorI& chainGrad, double y)
{

    sgdtk::Tensor& chainGradT = (sgdtk::Tensor&) chainGrad;


    const int oH = iH - kH + 1;
    const int oW = iW - kW + 1;

    const int zpH = iH + kH - 1;
    const int zpW = iW + kW - 1;

    chainGradT.reshape({nK, 1, oH * oW});

    int m = chainGradT.dims[2];
    int k = nK;
    int n = weights.dims[0];


    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0, &(chainGradT.d[0]), m,
                &(weights.d[0]), n, 0.0, &(unwrappedGradInput.d[0]), m);

    m = unwrappedInput.dims[1];
    k = unwrappedInput.dims[0];
    n = nK;

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0, &(unwrappedInput.d[0]), k,
                &(chainGradT.d[0]), k, 0.0, &(gradsW.d[0]), m);

    for (int l = 0; l < nK; ++l)
    {
        for (int i = 0; i < oH; ++i)
        {
            for (int j = 0; j < oW; ++j)
            {
                biasGrads[l] += chainGradT[(l * oH + i) * oW + j];
            }
        }

    }

    wrapGrad(unwrappedGradInput);
    return grads;

}