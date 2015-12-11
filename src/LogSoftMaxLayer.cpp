#include "n3rd/LogSoftMaxLayer.h"

using namespace n3rd;

template<typename Container_T> double logSum(const Container_T& v)
{
    double m = v[0];
    for (int i = 1, sz = v.size(); i < sz; ++i)
    {
        m = std::max(m, v[i]);
    }
    double s = 0.;
    for (int i = 0, sz = v.size(); i < sz; ++i)
    {
        s += std::exp(-(m - v[i]));
    }
    return m + std::log(s);
}

sgdtk::TensorI& LogSoftMaxLayer::forward(const sgdtk::TensorI& z)
{
    const sgdtk::Tensor& zT = (const sgdtk::Tensor&)z;
    int sz = z.size();
    if (sz != output.size())
    {
        output.resize({sz});
        grads.resize({sz});
    }

    auto logsum = logSum(zT);

    for (int i = 0; i < sz; ++i)
    {
        output[i] = zT[i] - logsum;
    }

    return output;
}


sgdtk::TensorI& LogSoftMaxLayer::backward(sgdtk::TensorI& chainGrad, double y)
{
    const sgdtk::Tensor& chainGradT = (const sgdtk::Tensor&)chainGrad;
    int sz = output.size();
    // Only will be one thing above us, a loss function
    auto sum = chainGradT[0];
    int yidx = (int)(y - 1);
    for (int i = 0; i < sz; ++i)
    {
        double indicator = yidx == i ? 1.0: 0.0;
        grads[i] = (indicator - std::exp(output[i]))*sum;
    }
    return grads;
}