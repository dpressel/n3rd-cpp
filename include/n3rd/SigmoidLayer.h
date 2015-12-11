//
// Created by Daniel on 9/24/2015.
//

#ifndef __N3RD_CPP_SIGMOIDLAYER_H__
#define __N3RD_CPP_SIGMOIDLAYER_H__

#include "n3rd/AbstractLayer.h"
#include <sgdtk/VectorN.h>
#include <sgdtk/DenseVectorN.h>
#include <cmath>
#include <sgdtk/TensorI.h>
#include <sgdtk/Tensor.h>
namespace n3rd
{
    class SigmoidLayer : public AbstractLayer<>
    {
        double sigmoid(double x)
        {
            return 1.0 / (1.0 + std::exp(-x));
        }
    public:

        SigmoidLayer() {}
        ~SigmoidLayer() {}

        sgdtk::TensorI& forward(const sgdtk::TensorI& z);

        sgdtk::TensorI& backward(sgdtk::TensorI& chainGrad, double y);

        std::string getType() const { return "SigmoidLayer"; }
    };

}
#endif
