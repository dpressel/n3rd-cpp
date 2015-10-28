//
// Created by Daniel on 9/24/2015.
//

#ifndef __N3RD_CPP_SIGMOIDLAYER_H__
#define __N3RD_CPP_SIGMOIDLAYER_H__

#include "n3rd/Layer.h"
#include <sgdtk/VectorN.h>
#include <sgdtk/DenseVectorN.h>
#include <cmath>
#include <sgdtk/Tensor.h>

namespace n3rd
{
    class SigmoidLayer : public Layer
    {
        double sigmoid(double x)
        {
            return 1.0 / (1.0 + std::exp(-x));
        }
    public:

        SigmoidLayer() {}
        ~SigmoidLayer() {}

        sgdtk::Tensor& forward(const sgdtk::Tensor& z);

        sgdtk::Tensor& backward(sgdtk::Tensor& chainGrad, double y);

        std::string getType() const { return "SigmoidLayer"; }
    };

}
#endif
