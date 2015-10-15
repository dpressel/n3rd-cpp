#include "n3rd/TanhLayer.h"

using namespace n3rd;

sgdtk::Tensor& TanhLayer::forward(const sgdtk::Tensor& z)
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
        output[i] = std::tanh(z[i]);
    }
    return output;
}

sgdtk::Tensor& TanhLayer::backward(const sgdtk::Tensor& chainGrad, double y)
{
    int sz = chainGrad.size();

    for (int i = 0; i < sz; ++i)
    {
        grads[i] = chainGrad[i] * (1 - output[i]*output[i]);
    }
    return grads;
}