//
// Created by Daniel on 9/24/2015.
//
#include <cblas.h>
#include "n3rd/FullyConnectedLayerBlas.h"

using namespace n3rd;
using namespace sgdtk;

FullyConnectedLayerBlas::FullyConnectedLayerBlas(int outputLength, int inputLength)
{

    this->outputLength = outputLength;
    this->inputLength = inputLength;
    weights.resize({outputLength, this->inputLength});
    gradsW.resize({outputLength, this->inputLength});
    grads.resize({this->inputLength});
    output.resize({outputLength});
    z.resize({inputLength});
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

sgdtk::Tensor& FullyConnectedLayerBlas::forward(const sgdtk::Tensor& input)
{

    output.constant(0.);
    std::copy(input.d.begin(), input.d.end(), z.d.begin());
    cblas_dgemv(CblasColMajor, CblasNoTrans, outputLength, inputLength, 1.0, &weights.d[0], outputLength, &input.d[0], 1, 1.0, &output.d[0], 1);
    return output;

}


sgdtk::Tensor& FullyConnectedLayerBlas::backward(sgdtk::Tensor& chainGrad, double y)
{

    grads.constant(0.);

    cblas_dgemv(CblasColMajor, CblasTrans, outputLength, inputLength, 1.0, &weights.d[0], outputLength,
                &chainGrad.d[0], 1, 1.0, &grads.d[0], 1);
    cblas_dger(CblasColMajor, outputLength, inputLength, 1.0, &chainGrad.d[0], 1, &z.d[0], 1, &gradsW.d[0], outputLength);

    for (int i = 0; i < outputLength; ++i)
    {
        biasGrads[i] = chainGrad.d[i];
    }
    return grads;


}
