//
// Created by Daniel on 9/24/2015.
//

#ifndef __N3RD_CPP_TANHLAYER_H__
#define __N3RD_CPP_TANHLAYER_H__

#include "n3rd/AbstractLayer.h"
#include <sgdtk/VectorN.h>
#include <sgdtk/DenseVectorN.h>
#include <cmath>
#include <sgdtk/TensorI.h>
#include <sgdtk/Tensor.h>
namespace n3rd
{
    class TanhLayer : public AbstractLayer<>
    {
    public:

        TanhLayer() {}
        ~TanhLayer() {}

        sgdtk::TensorI& forward(const sgdtk::TensorI& input);

        sgdtk::TensorI& backward(sgdtk::TensorI& chainGrad, double y);

        std::string getType() const { return "TanhLayer"; }
    };

}
#endif
