//
// Created by Daniel on 9/24/2015.
//

#ifndef __N3RD_CPP_LOGSOFTMAXLAYER_H__
#define __N3RD_CPP_LOGSOFTMAXLAYER_H__

#include "n3rd/AbstractLayer.h"
#include <sgdtk/DenseVectorN.h>
#include <cmath>
namespace n3rd
{

/**
 * LogSoftMaxLayer returns outputs in log soft max space
 *
 * @author dpressel
 */
    class LogSoftMaxLayer : public AbstractLayer<>
    {
    public:

        sgdtk::TensorI& forward(const sgdtk::TensorI& z);

        sgdtk::TensorI& backward(sgdtk::TensorI& chainGrad, double y);

        std::string getType() const { return "LogSoftMaxLayer"; }
    };
}
#endif
