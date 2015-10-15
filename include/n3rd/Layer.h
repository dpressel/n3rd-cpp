#ifndef __N3RD_CPP_LAYER_H__
#define __N3RD_CPP_LAYER_H__

#include <sgdtk/VectorN.h>
#include <sgdtk/Tensor.h>
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
        std::vector<double> biasGrads;
        std::vector<double> biases;

        sgdtk::Tensor weights;
        sgdtk::Tensor grads;
        sgdtk::Tensor gradsW;
        sgdtk::Tensor output;
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
        virtual sgdtk::Tensor& forward(const sgdtk::Tensor& x) = 0;

        /**
         * Implement back prop
         *
         * @param chainGrad Deltas from the layer above
         * @param y deltas from this layer
         * @return
         */
        virtual sgdtk::Tensor& backward(const sgdtk::Tensor& chainGrad, double y) = 0;

        virtual sgdtk::Tensor& getOutput()
        {
            return output;
        }

        virtual const sgdtk::Tensor& getOutput() const
        {
            return output;
        }

        virtual sgdtk::Tensor& getParamGrads()
        {
            return gradsW;
        }
        virtual sgdtk::Tensor& getParams()
        {
            return weights;
        }
        virtual std::vector<double>& getBiasGrads()
        {
            return biasGrads;
        }
        virtual std::vector<double>& getBiasParams()
        {
            return biases;
        }


        virtual std::string getType() const = 0;



    };
}

#endif
