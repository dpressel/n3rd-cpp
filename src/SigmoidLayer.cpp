#include "n3rd/SigmoidLayer.h"

using namespace n3rd;

sgdtk::TensorI& SigmoidLayer::forward(const sgdtk::TensorI& z)
{
    const sgdtk::Tensor& zT = (const sgdtk::Tensor&)z;
    int sz = z.size();

    output.resize({sz});
    grads.resize({sz});

    for (int i = 0; i < sz; ++i)
    {
        output[i] = sigmoid(zT[i]);
    }
    return output;
}

sgdtk::TensorI& SigmoidLayer::backward(sgdtk::TensorI& chainGrad, double y)
{
    const sgdtk::Tensor& chainGradT = (const sgdtk::Tensor&)chainGrad;
    int sz = chainGrad.size();

    for (int i = 0; i < sz; ++i)
    {
        grads[i] = chainGradT[i] * (1 - output[i]) * output[i];
    }
    return grads;
}