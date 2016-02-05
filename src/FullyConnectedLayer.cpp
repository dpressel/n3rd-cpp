//
// Created by Daniel on 9/24/2015.
//
#include "n3rd/FullyConnectedLayer.h"

using namespace n3rd;
using namespace sgdtk;
//#ifdef DEBUG

FullyConnectedLayer::FullyConnectedLayer(int outputLength, int inputLength)
{
    this->outputLength = outputLength;
    this->inputLength = inputLength;
    weights.resize({outputLength, this->inputLength});
    weightAccum.resize(weights.dims, 0);
    gradsW.resize(weights.dims);

    biases.resize({outputLength}, 0.);
    biasGrads.resize({outputLength}, 0.);
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

           number = distribution(generator);
            d = number * stdv2 - stdv;

            weights[ibase + j] = d;
        }
        number = distribution(generator);
        d = number * stdv2 - stdv;
        biases[i] = 0;//d;
    }
    grads.resize({this->inputLength});
    z.resize({this->inputLength});
    output.resize({outputLength});

}

sgdtk::Tensor& FullyConnectedLayer::fX(const Tensor& x, const Tensor& w)
{
    int zL = std::min(inputLength, (int)x.size());
    //System.out.println("zL " + zL + ", " + outputLength + " x " + inputLength);
    for (int i = 0, ibase = 0; i < outputLength; ++i, ibase += inputLength)
    {
        double acc = 0.;
        for (int j = 0; j < zL; ++j)
        {
            acc += w[ibase + j] * x[j];
        }
        output[i] = acc + this->biases[i];
    }
    return output;
}


sgdtk::TensorI& FullyConnectedLayer::backward(sgdtk::TensorI& chainGrad, double y)
{

    const sgdtk::Tensor& chainGradT = (const sgdtk::Tensor&)chainGrad;
    int zLength = z.size();
    int howLong = std::min(inputLength, zLength);

    grads.zeros();
    for (int i = 0, ibase = 0; i < outputLength; ++i, ibase += inputLength)
    {
        double g = chainGradT[i];
        for (int j = 0; j < howLong; ++j)
        {

            gradsW[ibase + j] += g * z[j];
            grads[j] += g * weights[ibase + j];

        }
        // push propagates through on a constant term
        biasGrads[i] += g;
    }

    return grads;

}
