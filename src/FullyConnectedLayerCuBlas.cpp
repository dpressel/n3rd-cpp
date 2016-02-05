//
// Created by Daniel on 9/24/2015.
//
#include <cblas.h>
#include "n3rd/FullyConnectedLayerCuBlas.h"

using namespace n3rd;
using namespace sgdtk;

FullyConnectedLayerCuBlas::FullyConnectedLayerCuBlas(int outputLength, int inputLength)
{

    this->outputLength = outputLength;
    this->inputLength = inputLength;
    sgdtk::Tensor cpuWeights({outputLength, this->inputLength});
    weights.resize(cpuWeights.dims);
    gradsW.resize(weights.dims);

    // Only zero'd once
    weightAccum.resize(weights.dims, 0);
    grads.resize({this->inputLength});
    output.resize({outputLength});

    ///dWeights.resize({outputLength, this->inputLength});
    ///dWeightGrads.resize({outputLength, this->inputLength});
    ///dOutput.resize({outputLength});
    ///dGrads.resize({this->inputLength});

    ///dInput.resize({inputLength});

    biases.resize({outputLength}, 0);
    biasGrads.resize({outputLength}, 0);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double stdv = 1. / std::sqrt(inputLength);
    double stdv2 = stdv * 2;

    for (int i = 0, ibase = 0; i < outputLength; ++i, ibase += this->inputLength)
    {
        double number;
        double d;

        for (int j = 0; j < this->inputLength; ++j)
        {
//#ifdef DEBUG
///            number = RND[Current++ % RND.size()];
//#else
            number = distribution(generator);
//#endif
            d = number * stdv2 - stdv;

            cpuWeights[ibase + j] = d;
        }
        number = distribution(generator);
        d = number * stdv2 - stdv;
        ///biases[i] = 0;//d;
    }
    weights.fromCPU(cpuWeights, false);
}



sgdtk::TensorI& FullyConnectedLayerCuBlas::forward(const sgdtk::TensorI& input)
{

    dInput = &((const sgdtk::CudaTensor&)input);

    double one = 1.0;
    output.zeros();
    /// TODO: avoid constant() call above
    TRY_CUBLAS(cublasDgemv_v2(sgdtk::Globals::gBlasHandle, CUBLAS_OP_N, outputLength, inputLength, &one, weights.d, outputLength, dInput->d, 1, &one, output.d, 1));

    return output;

}


sgdtk::TensorI& FullyConnectedLayerCuBlas::backward(sgdtk::TensorI& chainGrad, double y)
{

    double one = 1.0;

    const sgdtk::CudaTensor& dChainGrad = (const sgdtk::CudaTensor&)chainGrad;

    grads.zeros();

    TRY_CUBLAS(cublasDgemv_v2(sgdtk::Globals::gBlasHandle, CUBLAS_OP_T, outputLength, inputLength, &one, weights.d, outputLength, dChainGrad.d, 1, &one, grads.d, 1));

    gradsW.zeros();
    TRY_CUBLAS(cublasDger_v2(sgdtk::Globals::gBlasHandle, outputLength, inputLength, &one, dChainGrad.d, 1, dInput->d, 1, gradsW.d, outputLength));

    biasGrads.zeros();
    biasGrads.add(dChainGrad);

    return grads;


}
