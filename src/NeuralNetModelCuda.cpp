#include "n3rd/NeuralNetModelCuda.h"

using namespace n3rd;

#include <cassert>
void NeuralNetModelCuda::updateBiasWeights(Layer *layer, double eta) const
{


    sgdtk::CudaTensor &biasParamsT = (sgdtk::CudaTensor&)layer->getBiasParams();
    sgdtk::CudaTensor &biasGradsT = (sgdtk::CudaTensor &) layer->getBiasGrads();

    int bSz = biasParamsT.size();
    n3rdgBiasUpdates(biasParamsT.d, biasGradsT.d, eta, bSz);

}

void NeuralNetModelCuda::adagradUpdate(Layer *layer, double eta, double lambda) const
{
    sgdtk::CudaTensor& gg = (sgdtk::CudaTensor&)layer->getWeightAccum();
    sgdtk::CudaTensor& weightsT = (sgdtk::CudaTensor&)layer->getParams();
    sgdtk::CudaTensor &weightGradsT = (sgdtk::CudaTensor &) layer->getParamGrads();
    int wSz = weightsT.size();

    assert(wSz);
    assert(weightGradsT.size() == wSz);
    assert(wSz == gg.size());

    n3rdgAdagradWeightUpdates(weightsT.d, weightGradsT.d, gg.d, eta, lambda, wSz);

/*
    sgdtk::Tensor cpuDbgWeights(weightGradsT.dims);
    weightGradsT.toCPU(cpuDbgWeights, false);

    for (int i = 0; i < wSz; ++i)
    {
        assert(cpuDbgWeights[i] == 0);
    }

    weightsT.toCPU(cpuDbgWeights, false);

    std::cout << cpuDbgWeights[0] << std::endl;
*/

}


/**
 * Give back the the output layer of the network, scaling if required
 *
 * @param fv Feature vector
 * @return Return an array of scores.  In the simple binary case, we get a single value back.  In the case of a
 * log softmax type output, we get an array of values, where the index into the array is (label - 1).  The highest
 * score is then the prediction
 */

std::vector<double> NeuralNetModelCuda::score(const sgdtk::FeatureVector* fv)
{

    sgdtk::DenseVectorN* dvn = (sgdtk::DenseVectorN*)fv->getX();

    sgdtk::CudaTensor x(dvn->getX());
    auto& fwd = forward(x);

    const sgdtk::CudaTensor& outT = (const sgdtk::CudaTensor&) fwd;

    std::vector<double> output(outT.size());
    sgdtk::copyArrayFromGPU(outT.d, output);

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
