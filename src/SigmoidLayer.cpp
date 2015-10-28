#include "n3rd/SigmoidLayer.h"

using namespace n3rd;

sgdtk::Tensor& SigmoidLayer::forward(const sgdtk::Tensor& z)
{
    int sz = z.size();

    output.resize({sz});
    grads.resize({sz});

    for (int i = 0; i < sz; ++i)
    {
        output[i] = sigmoid(z[i]);
    }
    return output;
}

sgdtk::Tensor& SigmoidLayer::backward(sgdtk::Tensor& chainGrad, double y)
{
    int sz = chainGrad.size();

    for (int i = 0; i < sz; ++i)
    {
        grads[i] = chainGrad[i] * (1 - output[i]) * output[i];
    }
    return grads;
}