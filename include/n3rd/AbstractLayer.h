#ifndef __N3RD_CPP_ABSTRACT_LAYER_H__
#define __N3RD_CPP_ABSTRACT_LAYER_H__

#include <sgdtk/VectorN.h>
#include <sgdtk/Tensor.h>
#include "n3rd/Layer.h"
#include <vector>
namespace n3rd
{
    /**
     * Contract for a layer
     *
     * Not all layers actually support learning parameters/biases, in which case they simply return null.
     * Forward and backward prop implementations are expected to be implemented and work as intended
     */
    template<typename TensorT = sgdtk::Tensor> class AbstractLayer : public Layer
    {
    protected:
        TensorT biasGrads;
        TensorT biases;

        TensorT weights;
        TensorT grads;
        TensorT gradsW;
        TensorT output;
        TensorT weightAccum;

    public:
        /**
         * Take the input and produce some outputs
         * for a pre-activation layer, this will perform a dot product on each output, and produces a layer that
         * is number of pre-activation units long.  for an activation layer, this will produce exactly the same number of
         * outputs as inputs
         *
         * @param x previous layer inputs or actual inputs
         * @return this layer's outputs
         */

        AbstractLayer() {}
        virtual ~AbstractLayer() {}
        virtual sgdtk::TensorI& forward(const sgdtk::TensorI& x) = 0;

        /**
         * Implement back prop
         *
         * @param chainGrad Deltas from the layer above
         * @param y deltas from this layer
         * @return
         */
        virtual sgdtk::TensorI& backward(sgdtk::TensorI& chainGrad, double y) = 0;

        virtual sgdtk::TensorI& getOutput()
        {
            return output;
        }

        virtual const sgdtk::TensorI& getOutput() const
        {
            return output;
        }

        virtual sgdtk::TensorI& getParamGrads()
        {
            return gradsW;
        }
        virtual sgdtk::TensorI& getParams()
        {
            return weights;
        }
        virtual sgdtk::TensorI& getBiasGrads()
        {
            return biasGrads;
        }
        virtual sgdtk::TensorI& getBiasParams()
        {
            return biases;
        }
        virtual sgdtk::TensorI& getWeightAccum()
        {
            return weightAccum;
        }

        virtual std::string getType() const = 0;



    };
}

#endif
