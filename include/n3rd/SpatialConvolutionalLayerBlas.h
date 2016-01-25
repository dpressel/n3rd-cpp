#ifndef __N3RD_CPP_SPATIALCONVOLUTIONALLAYERBLAS_H__
#define __N3RD_CPP_SPATIALCONVOLUTIONALLAYERBLAS_H__

#include <sgdtk/Tensor.h>
#include "n3rd/AbstractLayer.h"
#include "sgdtk/DenseVectorN.h"
#include <cmath>
#include <cstdlib>
#include "n3rd/FilterOps.h"

namespace n3rd
{

    class SpatialConvolutionalLayerBlas : public AbstractLayer<>
    {
        void unwrapInput(const sgdtk::Tensor& x);
        void wrapGrad(sgdtk::Tensor& unwrapped);
    public:

        int nK;
        int kL;
        int kH;
        int kW;
        int iH;
        int iW;
        sgdtk::Tensor unwrappedInput;
        sgdtk::Tensor unwrappedGradInput;

        SpatialConvolutionalLayerBlas()
        {

        }

        SpatialConvolutionalLayerBlas(int nK, int kH, int kW, std::vector<int> inputDims);

        sgdtk::TensorI& forward(const sgdtk::TensorI& z);


        sgdtk::TensorI& backward(sgdtk::TensorI& chainGrad, double y);

        std::string getType() const
        { return "SpatialConvolutionalLayerBlas"; }
    };


}
#endif