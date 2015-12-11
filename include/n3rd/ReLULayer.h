//
// Created by Daniel on 9/24/2015.
//

#ifndef __N3RD_CPP_RELULAYER_H__
#define __N3RD_CPP_RELULAYER_H__

#include "n3rd/AbstractLayer.h"
#include <sgdtk/VectorN.h>
#include <sgdtk/DenseVectorN.h>
#include <cmath>
#include <sgdtk/Tensor.h>

namespace n3rd
{
    class ReLULayer : public AbstractLayer<>
    {

    public:

        ReLULayer() {}
        ~ReLULayer() {}

        double relu(double d)
        {
            return std::max<double>(0., d);
        }
        double drelu(double d)
        {
            return d > 0. ? 1.: 0.;
        }

        sgdtk::TensorI& forward(const sgdtk::TensorI& input);

        sgdtk::TensorI& backward(sgdtk::TensorI& chainGrad, double y);

        std::string getType() const { return "ReLULayer"; }
    };

}
#endif
