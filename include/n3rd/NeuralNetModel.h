#ifndef __N3RD_CPP_NEURALNETMODEL_H__
#define __N3RD_CPP_NEURALNETMODEL_H__

#include "n3rd/Layer.h"
#include <sgdtk/Model.h>
#include <sgdtk/LinearModel.h>
#include <sgdtk/FeatureVector.h>

#include <cmath>
#include <iostream>

namespace n3rd
{


    /**
     * Flexible plug-n-play Layered Neural network model trained with Adagrad
     *
     * This model is basically a wrapper around a set of Layers as building blocks, similar to Torch, and trained
     * using Adagrad (for now anyway)
     *
     * For simplicity, serialization is done using Jackson to JSON, though we could use the fast serialization provided
     * in SGDTk in the future if its helpful.
     *
     * @author dpressel
     */
    class NeuralNetModel : public sgdtk::LinearModel
    {

        std::vector<Layer *> layers;
        bool scaleOutput = true;
        std::vector<std::vector<double>> gg;

        // We could allow these as tuning parameters to control the weighting on Adagrad, but for now, just do 1, 1
        const double ALPHA = 1.;
        const double BETA = 1.;
        const double EPS = 1e-8;
    public:
        /**
         * Default constructor, needed to reincarnate models
         */
        NeuralNetModel()
        {
        }

        /**
         * Constructor supporting a stack of layers, and an argument of whether to center the output at zero and scale
         * it between -1 and 1.  This detail is encapsulated with the NeuralNetModelFactory is the preferred way to create a
         * NeuralNetModel
         *
         * @param layers A stack of network layers
         * @param scaleOutput Scale and center data?
         */
        NeuralNetModel(std::vector<Layer *> layers, bool scaleOutput)
        {
            this->layers = layers;
            this->scaleOutput = scaleOutput;
            gg.resize(layers.size());

        }
        virtual ~NeuralNetModel()
        {
            for (auto layer : layers)
            {
                delete layer;
            }
        }

        // Here ya go!
        std::vector<Layer *> getLayers()
        {
            return layers;
        }


        double mag() const
        {
            return 0;
        }

        /**
         * Function to update the weights for a model.  I realize in retrospect that this may have been better encapsulated
         * in the actual Trainer, but this wouldve caused a coupling between the Trainer and the Model which we have
         * managed to avoid, which is sort of convenient for SGDTk, since, in the case of flatter models,
         * this manifests in a fairly straightforward implementation.  However, for a Neural Net, the result isnt
         * that desirable -- we end up jamming the backprop guts in here, and the type of update itself as well,
         * which means we are doing Adagrad details within here for now.
         *
         * @param vector
         * @param eta
         * @param lambda
         * @param dLoss
         * @param y
         */
        void updateWeights(const sgdtk::VectorN *vector, double eta, double lambda, double dLoss, double y);

        /**
         * Override the SGDTk base model's fit() function to predict.  This gives back a binary prediction centered around
         * 0.  Values that are positive are 'true', values that are negative are 'false'.  Note that for non-binary cases,
         * you should use NeuralNetModel.score(), not predict().  In the case of predict() for softmax output, it will
         * simply give you the (log) probability under the best class (but wont tell you which -- not very useful
         * outside the library, so use score()!!)
         *
         * @param fv A feature vector
         * @return A value centered at zero, where, in the case of classification, negative values indicate 'false'
         * and positive indicate 'true'
         */

        double predict(const sgdtk::FeatureVector *fv);

        /**
         * Forward prop
         * @param x A feature vector
         * @return A result
         */
        sgdtk::Tensor forward(const sgdtk::Tensor& x);

        /**
         * Give back the the output layer of the network, scaling if required
         *
         * @param fv Feature vector
         * @return Return an array of scores.  In the simple binary case, we get a single value back.  In the case of a
         * log softmax type output, we get an array of values, where the index into the array is (label - 1).  The highest
         * score is then the prediction
         */

        std::vector<double> score(const sgdtk::FeatureVector* fv);

        /**
         * Not doing this right now, so dont make us select the learning rate in SGDTk unless you desire abject failure.
         * @return Nothing!
         */

        Model* prototype()
        {
            return nullptr;
        }
    };

}

#endif
