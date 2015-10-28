//
// Created by Daniel on 9/24/2015.
//

#include "n3rd/MaxOverTimePoolingLayer.h"
#include <cassert>
using namespace n3rd;
using namespace sgdtk;
#include <iostream>

sgdtk::Tensor& MaxOverTimePoolingLayer::forward(const sgdtk::Tensor& z)
{

    numFrames = z.size() / featureMapSz;
    grads.resize({featureMapSz, 1, numFrames});
    int sz = output.size();

    for (int i = 0; i < sz; ++i)
    {
        output[i] = 0;
        origin[i] = -100;
    }

    for (int l = 0; l < featureMapSz; ++l)
    {

        int mxIndex = 0;
        double mxValue = -100;

        for (int i = 0; i < numFrames; ++i)
        {

            int inAddr = l * numFrames + i;
            double ati = z[inAddr];
            if (ati > mxValue)
            {
                mxIndex = inAddr;
                mxValue = ati;
            }

        }

        origin[l] = mxIndex;
        output[l] = mxValue;


    }
    return output;

}

// Since the output and input are the same for the max value, we can just apply the
// max-pool value from the output
sgdtk::Tensor& MaxOverTimePoolingLayer::backward(sgdtk::Tensor& chainGrad, double y)
{
    grads.constant(0.);

    for (int l = 0; l < featureMapSz; ++l)
    {

        int inAddr = origin[l];
        grads[inAddr] = chainGrad[l];

    }
    return grads;
}
