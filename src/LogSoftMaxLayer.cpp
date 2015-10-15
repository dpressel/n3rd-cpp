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

sgdtk::Tensor& LogSoftMaxLayer::forward(const sgdtk::Tensor& z)
{
    int sz = z.size();
    if (sz != output.size())
    {
        output.resize({sz});
        grads.resize({sz});
    }

    auto logsum = logSum(z);

    for (int i = 0; i < sz; ++i)
    {
        output[i] = z[i] - logsum;
    }

    return output;
}


sgdtk::Tensor& LogSoftMaxLayer::backward(const sgdtk::Tensor& chainGrad, double y)
{
    int sz = output.size();
    // Only will be one thing above us, a loss function
    auto sum = chainGrad[0];
    int yidx = (int)(y - 1);
    for (int i = 0; i < sz; ++i)
    {
        double indicator = yidx == i ? 1.0: 0.0;
        grads[i] = (indicator - std::exp(output[i]))*sum;
    }
    return grads;
}