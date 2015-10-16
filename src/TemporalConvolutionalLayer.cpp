//
// Created by Daniel on 9/24/2015.
//

#include "n3rd/TemporalConvolutionalLayer.h"

using namespace n3rd;

TemporalConvolutionalLayer::TemporalConvolutionalLayer(int nK, int kL, int kW, int embeddingSize)
{

    weights.resize({nK, kL, embeddingSize, kW});
    gradsW.resize({nK, kL, embeddingSize, kW});
    biases.resize(nK);
    biasGrads.resize(nK);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double stdv = 1. / std::sqrt(6. / 28.);
    double stdv2 = stdv * 2;

    // For each kernel, randomly initialize all weights
    for (int i = 0; i < weights.size(); ++i)
    {

        double number = distribution(generator);
        double d = number * stdv2 - stdv;
        weights[i] = d;
    }

}

sgdtk::Tensor& TemporalConvolutionalLayer::forward(const sgdtk::Tensor& z)
{

    const int nK = weights.dims[0];
    const int kL = weights.dims[1];
    const int embeddingSz = weights.dims[2];
    const int kW = weights.dims[3];

    const int numFrames = z.size() / embeddingSz / kL;
    const int oT = numFrames - kW + 1;

    input = z;
    input.reshape({kL, embeddingSz, numFrames});
    //z.constant(input.d, {inputFeatureMapSz, numFrames, embeddingSz});
    grads.resize({kL, embeddingSz, numFrames});
    output.resize({nK, embeddingSz, oT});
    FilterOps::corr1(input, weights, biases, output); // biases

    return output;

}


sgdtk::Tensor& TemporalConvolutionalLayer::backward(const sgdtk::Tensor& chainGrad, double y)
{
    const int featureMapSz = weights.dims[0];
    const int embeddingSz = weights.dims[2];
    const int kW = weights.dims[3];
    const int numFrames = input.dims[2];
    const int convOutputSz = numFrames - kW + 1;
    // The desired dims going backwards is going to be

    //std::vector<int> outputDims({ featureMapSz, convOutputSz, embeddingSz });

    int stride = convOutputSz * embeddingSz;

    for (int l = 0; l < featureMapSz; ++l)
    {
        for (int i = 0; i < stride; ++i)
        {
            this->biasGrads[l] += chainGrad[l * stride + i];
        }
        this->biasGrads[l] /= embeddingSz;
    }

    int zpFrameSize = numFrames + kW - 1;
    int zp = zpFrameSize - convOutputSz;
    grads.constant(1.0);
    sgdtk::Tensor zpChainGrad;

    embed(chainGrad, 0, 0, zp, zpChainGrad);
    sgdtk::Tensor tWeights;

    transposeWeight4D(weights, tWeights);

    std::vector<double> empty;

    FilterOps::conv1(zpChainGrad, tWeights, empty, grads);
    FilterOps::corr1Weights(input, chainGrad, gradsW);

    //// GRADIENT CHECK
    ////gradCheck(chainGradTensor);
    ////gradCheckX(chainGradTensor);
    return grads;
}