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

    // THIS *DOES NOT* WORK

    sgdtk::TensorI* start = makeLossTensor(dLoss);

    sgdtk::TensorI* chainGrad = start;

    int numLayers = layers.size();
    for (int k = numLayers - 1; k >= 0; --k)
    {
        Layer *layer = layers[k];

        // This updates the entire chain back, which handles our deltas, so now we have the backward delta
        // during this step, the weight params, if they exist should have also been computed
        sgdtk::TensorI& output = layer->backward(*chainGrad, y);
        chainGrad = &output;
        // Now we need to update each layer's weights

        updateLayerWeights(layer, eta, lambda);



    }
    delete start;
}

void NeuralNetModel::updateLayerWeights(Layer *layer, double eta, double lambda)
{
    sgdtk::TensorI &weights = (sgdtk::Tensor &) layer->getParams();
    if (weights.empty())
        return;

    sgdtk::TensorI &biasParams = layer->getParams();

    adagradUpdate(layer, eta, lambda);


    if (!biasParams.empty())
    {
        updateBiasWeights(layer, eta);
    }

}
void NeuralNetModel::updateBiasWeights(Layer *layer, double eta) const
{


    sgdtk::Tensor &biasParamsT = (sgdtk::Tensor&)layer->getBiasParams();
    sgdtk::Tensor &biasGradsT = (sgdtk::Tensor &) layer->getBiasGrads();

    int bSz = biasParamsT.size();

    for (int i = 0; i < bSz; ++i)
    {
        auto delta = -(biasGradsT[i] * eta);// * 0.01; // last number is total fudge
        biasParamsT[i] += delta;
        biasGradsT[i] = 0;
    }
}

void NeuralNetModel::adagradUpdate(Layer *layer, double eta, double lambda) const
{
    sgdtk::Tensor& gg = (sgdtk::Tensor&)layer->getWeightAccum();
    sgdtk::Tensor& weightsT = (sgdtk::Tensor&)layer->getParams();
    int wSz = weightsT.size();

    sgdtk::Tensor &weightGradsT = (sgdtk::Tensor &) layer->getParamGrads();

    for (int i = 0; i < wSz; ++i)
    {
        // No harm in skipping this on GU, no divergence
        if (weightGradsT[i] == 0.0)
            continue;

        // Adagrad update
        //addSquare(gg[k], weightGradsT);
        //gg[k].addSquare(weightGradsT);
        gg[i] = gg[i] + weightGradsT[i] * weightGradsT[i];
        auto etaThis = eta / sqrt(gg[i] + EPS);
        auto delta = -etaThis * weightGradsT[i];
        weightsT[i] *= (1 - eta * lambda);
        weightsT[i] += delta;
        weightGradsT[i] = 0;

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
sgdtk::TensorI& NeuralNetModel::forward(sgdtk::TensorI& x)
{
    const sgdtk::TensorI* in = &x;

    for (int i = 0, sz = layers.size(); i < sz; ++i)
    {
        Layer* layer = layers[i];
        layer->forward(*in);
        const auto& ref = layer->getOutput();
        in = &ref;
    }

    return layers[layers.size() - 1]->getOutput();
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
    sgdtk::Tensor& x = dvn->getX();
    auto& fwd = forward(x);

    const sgdtk::Tensor& outT = (const sgdtk::Tensor&) fwd;

    std::vector<double> output(outT.d);
    // Assuming a probability distribution, we are going to want to shift and scale
    if (scaleOutput)
    {
        for (int i = 0, sz = x.size(); i < sz; ++i)
        {
            output[i] = (output[i] - 0.5) * 2.0;
        }
    }
    return output;
}
