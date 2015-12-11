//
// Created by dpressel on 10/28/15.
//

#include "n3rd/DropoutLayer.h"

using namespace n3rd;


sgdtk::TensorI& DropoutLayer::forward(const sgdtk::TensorI& x)
{
    const sgdtk::Tensor& xT = (const sgdtk::Tensor&)x;
    std::bernoulli_distribution bernoulli(probDrop);
    int sz = x.size();

    if (probDrop > 0.)
    {
        output.resize(xT.dims);

        double scale = 1. / (1. - probDrop);
        int sz = x.size();
        if (bits.size() < x.size())
        {
            bits.resize(sz);
        }
        else
        {
            bits.clear();
        }
        for (int i = 0; i < sz; ++i)
        {
            bool mask = bernoulli(generator);

            bits[i] = mask;
            output[i] = mask ? 0.: xT[i] * scale;

        }

    }
    else
    {
        output = xT;
    }

}

sgdtk::TensorI& DropoutLayer::backward(sgdtk::TensorI& chainGrad, double y)
{
    const sgdtk::Tensor& chainGradT = (const sgdtk::Tensor&)chainGrad;
    double scale = 1. / (1. - probDrop);
    int sz = chainGradT.size();
    grads.resize(chainGradT.dims);

    grads.constant(0.);

    for (int i = 0; i < sz; ++i)
    {
        grads[i] = bits[i] ? 0. : chainGradT[i] * scale;
    }
    return grads;
}
