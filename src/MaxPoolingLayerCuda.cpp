#include "n3rd/MaxPoolingLayerCuda.h"
#include <cassert>
#include "n3rd/MaxPoolingLayer.h"

using namespace n3rd;
using namespace sgdtk;

sgdtk::TensorI& MaxPoolingLayerCuda::forward(const sgdtk::TensorI& z)
{

    const sgdtk::CudaTensor& zT = (const sgdtk::CudaTensor&)z;

    int sz = origin.size();

    const int kL = inputDims[0];
    const int iH = inputDims[1];
    const int iW = inputDims[2];
    const int oH = output.dims[1];
    const int oW = output.dims[2];

    origin.zeros();


    n3rdgMaxPooling2Forward(zT.d, origin.d, output.d, kL, iH, iW, oH, oW, dh, dw);
    return output;

}

// Since the output and input are the same for the max value, we can just apply the
// max-pool value from the output
sgdtk::TensorI& MaxPoolingLayerCuda::backward(sgdtk::TensorI& chainGrad, double y)
{
    const sgdtk::CudaTensor& chainGradT = (const sgdtk::CudaTensor&)chainGrad;
    grads.zeros();

    const int kL = inputDims[0];
    const int iH = inputDims[1];
    const int iW = inputDims[2];
    const int oH = output.dims[1];
    const int oW = output.dims[2];

    n3rdgMaxPooling2Backward(grads.d, origin.d, chainGradT.d, kL, iH, iW, oH, oW, dh, dw);
    return grads;

}
