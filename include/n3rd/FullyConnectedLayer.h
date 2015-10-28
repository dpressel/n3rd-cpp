#ifndef __N3RD_CPP_FULLYCONNECTEDLAYER_H__
#define __N3RD_CPP_FULLYCONNECTEDLAYER_H__

#include "n3rd/Layer.h"
#include <sgdtk/Tensor.h>
#include <cmath>
#include <cstdlib>
#include <sgdtk/DenseVectorN.h>


namespace n3rd
{
    class FullyConnectedLayer : public Layer
    {

        sgdtk::Tensor z;

        ///Tensor gradsW;

        int outputLength;
        int inputLength;

    public:
        /**
         * Empty constructor (for reincarnating models)
         */

        FullyConnectedLayer()
        {

        }

        /**
         * Constructor, with given outputLength and input length
         * @param outputLength Output length
         * @param inputLength Input length
         */

        FullyConnectedLayer(int outputLength, int inputLength);

        /**
         * Forward prop
         * @param x
         * @return
         */
        sgdtk::Tensor& forward(const sgdtk::Tensor& input)
        {

            for (int i = 0, sz = input.size(); i < sz; ++i)
            {
                z[i] = input[i];
            }
            return fX(z, weights);
        }

        sgdtk::Tensor& fX(const sgdtk::Tensor& x, const sgdtk::Tensor& w);

        /**
         * Do backprop
         * @param outputLayerGrad layers above's deltas
         * @param y Label
         * @return The deltas for this layer
         */
        sgdtk::Tensor& backward(sgdtk::Tensor& chainGrad, double y);

        int getOutputLength() const
        {
            return outputLength;
        }


        void setOutputLength(int outputLength)
        {
            this->outputLength = outputLength;
        }


        int getInputLength() const
        {
            return inputLength;
        }


        void setInputLength(int inputLength)
        {
            this->inputLength = inputLength;
        }
        std::string getType() const { return "FullyConnectedLayer"; }

    };
}

#endif