#include "n3rd/TanhLayer.h"

using namespace n3rd;

sgdtk::TensorI& TanhLayer::forward(const sgdtk::TensorI& z)
{
    const sgdtk::Tensor& zT = (const sgdtk::Tensor&)z;
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
        output[i] = std::tanh(zT[i]);
    }
    return output;
}

sgdtk::TensorI& TanhLayer::backward(sgdtk::TensorI& chainGrad, double y)
{
    const sgdtk::Tensor& chainGradT = (const sgdtk::Tensor&)chainGrad;
    int sz = chainGrad.size();

    for (int i = 0; i < sz; ++i)
    {
        grads[i] = chainGradT[i] * (1 - output[i]*output[i]);
    }
    return grads;
}