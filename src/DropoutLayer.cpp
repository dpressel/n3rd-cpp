//
// Created by dpressel on 10/28/15.
//

#include "n3rd/DropoutLayer.h"

using namespace n3rd;


sgdtk::Tensor& DropoutLayer::forward(const sgdtk::Tensor& x)
{
    std::bernoulli_distribution bernoulli(probDrop);
    int sz = x.size();

    if (probDrop > 0.)
    {
        output.resize(x.dims);

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
            output[i] = mask ? 0.: x[i] * scale;

        }

    }
    else
    {
        output = x;
    }

}

sgdtk::Tensor& DropoutLayer::backward(sgdtk::Tensor& chainGrad, double y)
{
    double scale = 1. / (1. - probDrop);
    int sz = chainGrad.size();
    grads.resize(chainGrad.dims);

    grads.constant(0.);

    for (int i = 0; i < sz; ++i)
    {
        grads[i] = bits[i] ? 0. : chainGrad[i] * scale;
    }
    return grads;
}
