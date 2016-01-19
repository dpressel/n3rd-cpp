#ifndef __N3RD_CPP_SPATIALCONVOLUTIONALLAYER_H__
#define __N3RD_CPP_SPATIALCONVOLUTIONALLAYER_H__

#include <sgdtk/Tensor.h>
#include "n3rd/AbstractLayer.h"
#include "sgdtk/DenseVectorN.h"
#include <cmath>
#include <cstdlib>
#include "n3rd/FilterOps.h"

namespace n3rd
{

    class SpatialConvolutionalLayer : public AbstractLayer<>
    {
    public:


        sgdtk::Tensor input;

        SpatialConvolutionalLayer()
        {

        }

        SpatialConvolutionalLayer(int nK, int kH, int kW, std::vector<int> inputDims);

        sgdtk::TensorI& forward(const sgdtk::TensorI& z);


        sgdtk::TensorI& backward(sgdtk::TensorI& chainGrad, double y);

        std::string getType() const
        { return "SpatialConvolutionalLayer"; }
    };


}
#endif