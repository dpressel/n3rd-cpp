#include "n3rd/ReLULayer.h"

using namespace n3rd;

sgdtk::Tensor& ReLULayer::forward(const sgdtk::Tensor& z)
{
    int sz = z.size();

    if (sz != output.size())
    {
        output.resize({sz});
        grads.resize({sz});
    }

    output.resize({sz});
    grads.resize({sz});

    for (int i = 0; i < sz; ++i)
    {
        output[i] = relu(z[i]);
    }
    return output;
}

sgdtk::Tensor& ReLULayer::backward(sgdtk::Tensor& chainGrad, double y)
{
    int sz = chainGrad.size();

    for (int i = 0; i < sz; ++i)
    {
        grads[i] = chainGrad[i] * drelu(output[i]);
    }
    return grads;
}