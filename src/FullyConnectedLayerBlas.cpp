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
    gradsW.resize(weights.dims);
    weightAccum.resize(weights.dims, 0);
    grads.resize({this->inputLength});
    output.resize({outputLength});
    z.resize({inputLength});
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

            weights[ibase + j] = d;
        }
        number = distribution(generator);
        d = number * stdv2 - stdv;
        biases[i] = 0;//d;
    }


}

sgdtk::TensorI& FullyConnectedLayerBlas::forward(const sgdtk::TensorI& input)
{

    output.zeros();
    const sgdtk::Tensor& inputT = (const sgdtk::Tensor&)input;

    for (int i = 0, sz = input.size(); i < sz; ++i)
    {
        z[i] = inputT[i];
    }

    cblas_dgemv(CblasColMajor, CblasNoTrans, outputLength, inputLength, 1.0, &weights.d[0], outputLength, &inputT.d[0], 1, 1.0, &output.d[0], 1);
    return output;

}


sgdtk::TensorI& FullyConnectedLayerBlas::backward(sgdtk::TensorI& chainGrad, double y)
{

    grads.zeros();

    const sgdtk::Tensor& chainGradT = (const sgdtk::Tensor&)chainGrad;
    cblas_dgemv(CblasColMajor, CblasTrans, outputLength, inputLength, 1.0, &weights.d[0], outputLength,
                &chainGradT.d[0], 1, 1.0, &grads.d[0], 1);
    cblas_dger(CblasColMajor, outputLength, inputLength, 1.0, &chainGradT.d[0], 1, &z.d[0], 1, &gradsW.d[0], outputLength);

    for (int i = 0; i < outputLength; ++i)
    {
        biasGrads[i] = chainGradT.d[i];
    }
    return grads;


}
