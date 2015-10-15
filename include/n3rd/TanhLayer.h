//
// Created by Daniel on 9/24/2015.
//

#ifndef __N3RD_CPP_TANHLAYER_H__
#define __N3RD_CPP_TANHLAYER_H__

#include "n3rd/Layer.h"
#include <sgdtk/VectorN.h>
#include <sgdtk/DenseVectorN.h>
#include <cmath>
#include <sgdtk/Tensor.h>

namespace n3rd
{
    class TanhLayer : public Layer
    {
    public:

        TanhLayer() {}
        ~TanhLayer() {}

        sgdtk::Tensor& forward(const sgdtk::Tensor& input);

        sgdtk::Tensor& backward(const sgdtk::Tensor& chainGrad, double y);

        std::string getType() const { return "TanhLayer"; }
    };

}
#endif
