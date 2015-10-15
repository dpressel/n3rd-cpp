#include "n3rd/NeuralNetModel.h"

using namespace n3rd;
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
void NeuralNetModel::updateWeights(const sgdtk::VectorN *vector, double eta, double lambda, double dLoss, double y)
{
    // Allocate Adagrad vector

    sgdtk::Tensor out({dLoss}, {1});

    auto& chainGrad = out;

    int numLayers = layers.size();
    for (int k = numLayers - 1; k >= 0; --k)
    {
        Layer *layer = layers[k];
#ifdef DEBUG_OUT
        std::cout << layer->getType() << "... ";
#endif
        // This updates the entire chain back, which handles our deltas, so now we have the backward delta
        // during this step, the weight params, if they exist should have also been computed
        chainGrad = layer->backward(chainGrad, y);
#ifdef DEBUG_OUT
        std::cout << "DONE" << std::endl;
#endif
        // Now we need to update each layer's weights
        sgdtk::Tensor& weights = layer->getParams();

        //std::cout << layer->getType() << std::endl;
        // Sometimes weights can be NULL in layers without parameters, dont touch them!
        if (!weights.empty())
        {
#ifdef DEBUG_OUTPUT
            std::cout << "\tUpdating weights" << std::endl;
#endif
            // Initialize Adagrad for layer k
            if (gg[k].empty())
            {
                gg[k].resize(weights.size(), 0);
            }
            auto& weightGrads = layer->getParamGrads();

            for (int i = 0, sz = weights.size(); i < sz; ++i)
            {
                if (weightGrads[i] == 0.0)
                    continue;

                // Adagrad update
                gg[k][i] = ALPHA * gg[k][i] + BETA * weightGrads[i] * weightGrads[i];
                auto etaThis = eta / std::sqrt(gg[k][i] + EPS);
                auto delta = -etaThis * weightGrads[i];
                weights[i] *= (1 - eta * lambda);
                weights[i] += delta;
                weightGrads[i] = 0;

            }

        }

        auto& biasParams = layer->getBiasParams();
        // Same story for biasParams, can be NULL
        if (!biasParams.empty())
        {
#ifdef DEBUG_OUTPUT
            std::cout << "\tUpdating biases" << std::endl;
#endif
            auto& biasGrads = layer->getBiasGrads();
            for (int i = 0, sz = biasParams.size(); i < sz; ++i)
            {
                // Dont bother to regularize
                auto delta = -(biasGrads[i] * eta);// * 0.01; // last number is total fudge
                biasParams[i] += delta;
                biasGrads[i] = 0;
            }
        }
    }
}


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

double NeuralNetModel::predict(const sgdtk::FeatureVector *fv)
{
    std::vector<double> scores = score(fv);
    double mx = scores[0];
    for (int i = 1, sz = scores.size(); i < sz; ++i)
    {
        mx = std::max(scores[i], mx);
    }
    return mx;
}

/**
 * Forward prop
 * @param x A feature vector
 * @return A result
 */
sgdtk::Tensor NeuralNetModel::forward(const sgdtk::Tensor& x)
{

    const auto* in = &x;
    for (int i = 0, sz = layers.size(); i < sz; ++i)
    {
        Layer* layer = layers[i];
        layer->forward(*in);
        const auto& ref = layer->getOutput();
        in = &ref;
    }
    return *in;
}

/**
 * Give back the the output layer of the network, scaling if required
 *
 * @param fv Feature vector
 * @return Return an array of scores.  In the simple binary case, we get a single value back.  In the case of a
 * log softmax type output, we get an array of values, where the index into the array is (label - 1).  The highest
 * score is then the prediction
 */

std::vector<double> NeuralNetModel::score(const sgdtk::FeatureVector* fv)
{

    sgdtk::DenseVectorN* dvn = (sgdtk::DenseVectorN*)fv->getX();
    auto x = forward(dvn->getX());

    // Assuming a probability distribution, we are going to want to shift and scale
    if (scaleOutput)
    {
        for (int i = 0, sz = x.size(); i < sz; ++i)
        {
            x[i] = (x[i] - 0.5) * 2.0;
        }
    }
    return x.d;
}
