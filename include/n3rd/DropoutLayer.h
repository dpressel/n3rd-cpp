//
// Created by dpressel on 10/28/15.
//

#ifndef __N3RD_CPP_DROPOUT_H__
#define __N3RD_CPP_DROPOUT_H__

#include "n3rd/Layer.h"
#include <random>
#include <vector>

namespace n3rd
{
    class DropoutLayer : public Layer
    {
        std::vector<bool> bits;
        double probDrop;

        std::default_random_engine generator;
    public:
        DropoutLayer(double pDrop = 0.) : probDrop(pDrop)
        {

            bits.resize(1024);
        }

        sgdtk::Tensor& forward(const sgdtk::Tensor& x);

        sgdtk::Tensor& backward(sgdtk::Tensor& chainGrad, double y);

        std::string getType() const { return "DropoutLayer"; }
    };
}
#endif
