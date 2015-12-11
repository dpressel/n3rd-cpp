//
// Created by dpressel on 10/28/15.
//

#ifndef __N3RD_CPP_DROPOUT_H__
#define __N3RD_CPP_DROPOUT_H__

#include "n3rd/AbstractLayer.h"
#include <random>
#include <vector>

namespace n3rd
{
    class DropoutLayer : public AbstractLayer<>
    {
        std::vector<bool> bits;
        double probDrop;

        std::default_random_engine generator;
    public:
        DropoutLayer(double pDrop = 0.) : probDrop(pDrop)
        {

            bits.resize(1024);
        }

        sgdtk::TensorI& forward(const sgdtk::TensorI& x);

        sgdtk::TensorI& backward(sgdtk::TensorI& chainGrad, double y);

        std::string getType() const { return "DropoutLayer"; }
    };
}
#endif
