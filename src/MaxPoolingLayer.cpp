#include "n3rd/MaxPoolingLayer.h"
#include <cassert>
using namespace n3rd;
using namespace sgdtk;
#include <iostream>

sgdtk::TensorI& MaxPoolingLayer::forward(const sgdtk::TensorI& z)
{

    const sgdtk::Tensor& zT = (const sgdtk::Tensor&)z;

    int sz = origin.size();

    const int kL = inputDims[0];
    const int iH = inputDims[1];
    const int iW = inputDims[2];
    const int oH = output.dims[1];
    const int oW = output.dims[2];

    for (int i = 0; i < sz; ++i)
    {
        output[i] = DS_MIN;
        origin[i] = 0;
    }

    for (int l = 0; l < kL; ++l)
    {
        for (int i = 0; i < iH; ++i)
        {
            int oi = (int) std::floor(i / (double) dh);

            for (int j = 0; j < iW; ++j)
            {
                int oj = (int) std::floor(j / (double) dw);
                int outAddr = (l * oH + oi) * oW + oj;
                int inAddr = (l * iH + i) * iW + j;

                double zi = zT[inAddr];

                if (output[outAddr] < zi)
                {
                    output[outAddr] = zi;
                    origin[outAddr] = inAddr;
                }
            }
        }

    }
    return output;

}

// Since the output and input are the same for the max value, we can just apply the
// max-pool value from the output
sgdtk::TensorI& MaxPoolingLayer::backward(sgdtk::TensorI& chainGrad, double y)
{
    const sgdtk::Tensor& chainGradT = (const sgdtk::Tensor&)chainGrad;
    grads.zeros();

    const int kL = inputDims[0];
    const int iH = inputDims[1];
    const int iW = inputDims[2];
    const int oH = output.dims[1];
    const int oW = output.dims[2];


    for (int l = 0; l < kL; ++l)
    {
        for (int i = 0; i < iH; ++i)
        {
            int oi = (int)std::floor(i / (double) dh);

            for (int j = 0; j < iW; ++j)
            {
                int oj = (int)std::floor(j / (double) dw);
                int outAddr = (l *oH + oi) * oW + oj;
                int inAddr = (l * iH + i) * iW + j;
                grads[inAddr] = origin[outAddr] == inAddr ? chainGradT[outAddr] : 0.;
            }
        }
    }
    return grads;

}
