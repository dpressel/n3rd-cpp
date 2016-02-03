#ifndef __N3RD_CPP_SPATIALCONVOLUTIONALLAYERCUBLAS_H__
#define __N3RD_CPP_SPATIALCONVOLUTIONALLAYERCUBLAS_H__

#include <sgdtk/Tensor.h>
#include <sgdtk/CudaTensor.h>
#include "n3rd/AbstractLayer.h"
#include "sgdtk/DenseVectorN.h"
#include <cmath>
#include <cstdlib>
#include "n3rd/FilterOps.h"
#include "n3rd/GPUOps.h"
namespace n3rd
{

    class SpatialConvolutionalLayerCuBlas : public AbstractLayer<sgdtk::CudaTensor>
    {
        void unwrapInput(const sgdtk::CudaTensor& x);
        void wrapGrad(sgdtk::CudaTensor& unwrapped);
    public:

        int nK;
        int kL;
        int kH;
        int kW;
        int iH;
        int iW;
        sgdtk::CudaTensor dUnwrappedInput;
        sgdtk::CudaTensor dUnwrappedGradInput;

        SpatialConvolutionalLayerCuBlas()
        {

        }

        SpatialConvolutionalLayerCuBlas(int nK, int kH, int kW, std::vector<int> inputDims);

        sgdtk::TensorI& forward(const sgdtk::TensorI& z);


        sgdtk::TensorI& backward(sgdtk::TensorI& chainGrad, double y);

        std::string getType() const
        { return "SpatialConvolutionalLayerCuBlas"; }
    };


}
#endif