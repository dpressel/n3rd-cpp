//
// Created by Daniel on 9/24/2015.
//

#include "n3rd/KMaxPoolingLayer.h"
#include <cassert>
using namespace n3rd;
using namespace sgdtk;
#include <iostream>

sgdtk::Tensor& KMaxPoolingLayer::forward(const sgdtk::Tensor& z)
{

    numFrames = z.size() / embeddingSz / featureMapSz;
    const int sz = featureMapSz * k * embeddingSz;
    grads.resize({featureMapSz, embeddingSz, numFrames});
    grads.constant(0.);

    // If the numFrames is less than k, we need to make sure these are zero'd
    output.constant(0.);
    originDims = {featureMapSz, embeddingSz, k};

    for (int i = 0; i < sz; ++i)
    {
        output[i] = 0.;
        origin[i] = -100;
    }

    //}

    for (int l = 0, lbase = 0; l < featureMapSz; ++l, lbase += embeddingSz)
    {
        for (int j = 0; j < embeddingSz; ++j)
        {
            std::vector<sgdtk::Offset> offsets(numFrames);

            const int ibase = (lbase + j) * numFrames;
            const int obase = (lbase + j) * k;

            for (int i = 0; i < numFrames; ++i)
            {
                int inAddr = ibase + i;
                offsets[i] = sgdtk::Offset(inAddr, z[inAddr]);
            }

            std::sort(offsets.begin(), offsets.end(), maxValue);
            int mn = std::min(k, (int)offsets.size());
            offsets.resize(mn);

            std::sort(offsets.begin(), offsets.end(), minIndex);
            for (int i = 0, sz = offsets.size(); i < sz; ++i)
            {
                int outAddr = obase + i;
                origin[outAddr] = offsets[i].first;
                output[outAddr] = offsets[i].second;
            }
        }

    }
    return output;

}

// Since the output and input are the same for the max value, we can just apply the
// max-pool value from the output
sgdtk::Tensor& KMaxPoolingLayer::backward(const sgdtk::Tensor& chainGrad, double y)
{

    grads.constant(0.);
    for (int l = 0, lbase = 0; l < featureMapSz; ++l, lbase += embeddingSz)
    {
        for (int j = 0; j < embeddingSz; ++j)
        {
            int obase = (lbase + j) * k;
            for (int i = 0; i < k; ++i)
            {

                int outAddr = obase + i;
                int inAddr = origin[outAddr];

                if (origin[outAddr] < 0 && (origin[outAddr] != -100))
                {
                    std::cout << "fail " << origin[462] << std::endl;
                    std::cout << "outAddr " << outAddr << " l,k,i,j " << l << ',' << k << ',' << i << ',' << j << std::endl;
                    assert(false);
                }

                if (inAddr == -100)
                {
                    continue;
                }
                grads[inAddr] = chainGrad[outAddr];
            }
        }
    }
    return grads;
}
