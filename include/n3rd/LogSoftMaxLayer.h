//
// Created by Daniel on 9/24/2015.
//

#ifndef __N3RD_CPP_LOGSOFTMAXLAYER_H__
#define __N3RD_CPP_LOGSOFTMAXLAYER_H__

#include "n3rd/Layer.h"
#include <sgdtk/DenseVectorN.h>
#include <cmath>
namespace n3rd
{

/**
 * LogSoftMaxLayer returns outputs in log soft max space
 *
 * @author dpressel
 */
    class LogSoftMaxLayer : public Layer
    {
    public:

        sgdtk::Tensor& forward(const sgdtk::Tensor& z);

        sgdtk::Tensor& backward(sgdtk::Tensor& chainGrad, double y);

        std::string getType() const { return "LogSoftMaxLayer"; }
    };
}
#endif
