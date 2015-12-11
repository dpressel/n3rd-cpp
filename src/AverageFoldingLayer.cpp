//
// Created by Daniel on 9/24/2015.
//

#include "n3rd/AverageFoldingLayer.h"

using namespace n3rd;

sgdtk::TensorI& AverageFoldingLayer::forward(const sgdtk::TensorI& z)
{

    const sgdtk::Tensor& tensor = (const sgdtk::Tensor&)z;
    numFrames = z.size() / embeddingSz / featureMapSz;
    const int outEmbeddingSz = embeddingSz / k;

    //if (output.size() != z.size())
    //{
    output.resize({featureMapSz, outEmbeddingSz, numFrames});
    grads.resize({featureMapSz, embeddingSz, numFrames});
    //}
    auto div = 1.0 / k;
    for (int l = 0, lbase = 0, libase = 0; l < featureMapSz; ++l, lbase += outEmbeddingSz, libase += embeddingSz)
    {
        for (int j = 0, p = 0; j < embeddingSz; j += k, ++p)
        {
            int obase = (lbase + p) * numFrames;
            for (int i = 0; i < numFrames; ++i)
            {
                int oAddr = obase + i;

                output[oAddr] = 0.0;
                for (int m = 0; m < k; ++m)
                {
                    int iAddr = (libase + j + m) * numFrames + i;
                    output[oAddr] += tensor[iAddr];
                }
                output[oAddr] *= div;
            }
        }
    }
    return output;
}

// Since the output and input are the same for the max value, we can just apply the
// max-pool value from the output

sgdtk::TensorI& AverageFoldingLayer::backward(sgdtk::TensorI& chainGrad, double y)
{
    const sgdtk::Tensor& chainGradT = (const sgdtk::Tensor&)chainGrad;
    auto div = 1.0 / k;
    int outEmbeddingSz = embeddingSz / k;
    grads.constant(0.);

    for (int l = 0, lbase = 0, libase = 0; l < featureMapSz; ++l, lbase += outEmbeddingSz, libase += embeddingSz)
    {
        for (int j = 0, p = 0; j < embeddingSz; j += k, ++p)
        {
            int obase = (lbase + p) * numFrames;
            for (int i = 0; i < numFrames; ++i)
            {
                int oAddr = obase + i;
                double value = chainGradT[oAddr] * div;
                for (int m = 0; m < k; ++m)
                {
                    int iAddr = (libase + j + m) * numFrames + i;
                    grads[iAddr] = value;
                }
            }
        }
    }

    return grads;
}
