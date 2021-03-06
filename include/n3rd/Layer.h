#ifndef __N3RD_CPP_LAYER_H__
#define __N3RD_CPP_LAYER_H__

#include <sgdtk/VectorN.h>
#include <sgdtk/TensorI.h>
#include <vector>
namespace n3rd
{
    /**
     * Contract for a layer
     *
     * Not all layers actually support learning parameters/biases, in which case they simply return null.
     * Forward and backward prop implementations are expected to be implemented and work as intended
     */
    class Layer
    {
    protected:

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

        Layer() {}
        virtual ~Layer() {}
        virtual sgdtk::TensorI& forward(const sgdtk::TensorI& x) = 0;

        /**
         * Implement back prop
         *
         * @param chainGrad Deltas from the layer above
         * @param y deltas from this layer
         * @return
         */
        virtual sgdtk::TensorI& backward(sgdtk::TensorI& chainGrad, double y) = 0;

        virtual sgdtk::TensorI& getOutput() = 0;

        virtual const sgdtk::TensorI& getOutput() const = 0;

        virtual sgdtk::TensorI& getParamGrads() = 0;

        virtual sgdtk::TensorI& getParams() = 0;

        virtual sgdtk::TensorI& getBiasGrads() = 0;

        virtual sgdtk::TensorI& getBiasParams() = 0;

        virtual std::string getType() const = 0;

        // If an algorithm like Adagrad is used, we can use this to maintain a local weight accumulation
        virtual sgdtk::TensorI& getWeightAccum() = 0;


    };
}

#endif
