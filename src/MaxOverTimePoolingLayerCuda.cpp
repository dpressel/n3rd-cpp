//
// Created by Daniel on 9/24/2015.
//

#include "n3rd/MaxOverTimePoolingLayerCuda.h"
#include <cassert>
using namespace n3rd;
using namespace sgdtk;
#include <iostream>

sgdtk::TensorI& MaxOverTimePoolingLayerCuda::forward(const sgdtk::TensorI& z)
{
    const sgdtk::CudaTensor& dInput = (const sgdtk::CudaTensor&)z;

    numFrames = dInput.size() / featureMapSz;
    /////CudaTensor dInput(zT);

    grads.resize({featureMapSz, 1, numFrames});

    n3rdgMaxOverTimeForward(dInput.d, output.d, origin.d, featureMapSz, numFrames);


    ////dOutput.toCPU(output);
    ////dOrigin.toCPU(origin);
    //dOutput.toCPU(output, false);
    //dOrigin.toCPU(origin, false);

    return output;

}

// Since the output and input are the same for the max value, we can just apply the
// max-pool value from the output
sgdtk::TensorI& MaxOverTimePoolingLayerCuda::backward(sgdtk::TensorI& chainGrad, double y)
{
    const sgdtk::CudaTensor& chainGradT = (const sgdtk::CudaTensor&)chainGrad;
    grads.constant(0.);

///    for (int l = 0; l < featureMapSz; ++l)
///    {
///
///        int inAddr = origin[l];
///        grads[inAddr] = chainGradT[l];
///    }
    n3rdgMaxOverTimeBackward(chainGradT.d, origin.d, grads.d, featureMapSz);
    return grads;
}
