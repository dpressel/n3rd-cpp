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
    weights.resize({outputLength, this->inputLength});
    gradsW.resize({outputLength, this->inputLength});
    grads.resize({this->inputLength});
    output.resize({outputLength});

    dWeights.resize({outputLength, this->inputLength});
    dWeightGrads.resize({outputLength, this->inputLength});
    dOutput.resize({outputLength});
    dGrads.resize({this->inputLength});

    dInput.resize({inputLength});

    biases.resize(outputLength);
    biasGrads.resize(outputLength);

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

            weights[ibase + j] = d;
        }
        number = distribution(generator);
        d = number * stdv2 - stdv;
        biases[i] = 0;//d;
    }


}



sgdtk::Tensor& FullyConnectedLayerCuBlas::forward(const sgdtk::Tensor& input)
{

    dWeights.fromCPU(weights, false);
    dOutput.constant(0.);
    dInput.fromCPU(input, false);
    double one = 1.0;
    TRY_CUBLAS(cublasDgemv_v2(sgdtk::Globals::gBlasHandle, CUBLAS_OP_N, outputLength, inputLength, &one, dWeights.d, outputLength, dInput.d, 1, &one, dOutput.d, 1));

    dOutput.toCPU(output, false);
    return output;

}


sgdtk::Tensor& FullyConnectedLayerCuBlas::backward(sgdtk::Tensor& chainGrad, double y)
{

    double one = 1.0;
    sgdtk::CudaTensor dChainGrad(chainGrad);

    dGrads.constant(0.);

    TRY_CUBLAS(cublasDgemv_v2(sgdtk::Globals::gBlasHandle, CUBLAS_OP_T, outputLength, inputLength, &one, dWeights.d, outputLength, dChainGrad.d, 1, &one, dGrads.d, 1));
    //cblas_dgemv(CblasColMajor, CblasTrans, outputLength, inputLength, &one, &weights.d[0], outputLength,
    //            &chainGrad.d[0], 1, &one, &grads.d[0], 1);

    dGrads.toCPU(grads);

    dWeightGrads.constant(0.);
    TRY_CUBLAS(cublasDger_v2(sgdtk::Globals::gBlasHandle, outputLength, inputLength, &one, dChainGrad.d, 1, dInput.d, 1, dWeightGrads.d, outputLength));

    //cblas_dger(CblasColMajor, outputLength, inputLength, &one, dChainGrad.d, 1, &z.d[0], 1, &gradsW.d[0], outputLength);

    dWeightGrads.toCPU(gradsW);


    for (int i = 0; i < outputLength; ++i)
    {
        biasGrads[i] = chainGrad.d[i];
    }


    return grads;


}
