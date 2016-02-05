//
// Created by Daniel on 9/24/2015.
//

#include "n3rd/TemporalConvolutionalLayerFFT.h"

using namespace n3rd;



TemporalConvolutionalLayerFFT::TemporalConvolutionalLayerFFT(int nK, int kL, int kW, int embeddingSize)
{

    weights.resize({nK, kL, embeddingSize, kW});
    gradsW.resize(weights.dims);
    weightAccum.resize(weights.dims, 0);
    biases.resize({nK}, 0);
    biasGrads.resize({nK}, 0);

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

sgdtk::TensorI& TemporalConvolutionalLayerFFT::forward(const sgdtk::TensorI& z)
{

    const int nK = weights.dims[0];
    const int kL = weights.dims[1];
    const int embeddingSz = weights.dims[2];
    const int kW = weights.dims[3];

    const int numFrames = z.size() / embeddingSz / kL;
    const int oT = numFrames - kW + 1;

    const sgdtk::Tensor& zT = (const sgdtk::Tensor&)z;
    input = zT;
    input.reshape({kL, embeddingSz, numFrames});
    //z.constant(input.d, {inputFeatureMapSz, numFrames, embeddingSz});
    grads.resize({kL, embeddingSz, numFrames});
    output.resize({nK, embeddingSz, oT});
    conv.fftfilt1(input, weights, biases, output);

    return output;

}


sgdtk::TensorI& TemporalConvolutionalLayerFFT::backward(sgdtk::TensorI& chainGrad, double y)
{
    const sgdtk::Tensor& chainGradT = (const sgdtk::Tensor&)chainGrad;
    const int featureMapSz = weights.dims[0];
    const int embeddingSz = weights.dims[2];
    const int kW = weights.dims[3];
    const int numFrames = input.dims[2];
    const int convOutputSz = numFrames - kW + 1;
    // The desired dims going backwards is going to be
    chainGrad.reshape({featureMapSz, embeddingSz, convOutputSz});
    //std::vector<int> outputDims({ featureMapSz, convOutputSz, embeddingSz });

    int stride = convOutputSz * embeddingSz;

    for (int l = 0; l < featureMapSz; ++l)
    {
        for (int i = 0; i < stride; ++i)
        {
            this->biasGrads[l] += chainGradT[l * stride + i];
        }
        this->biasGrads[l] /= embeddingSz;
    }

    int zpFrameSize = numFrames + kW - 1;
    int zp = zpFrameSize - convOutputSz;
    grads.zeros();
    sgdtk::Tensor zpChainGrad;

    embed(chainGradT, 0, 0, zp, zpChainGrad);
    sgdtk::Tensor tWeights;

    transposeWeight4D(weights, tWeights);

    sgdtk::Tensor empty;

    //conv.fftfilt1(zpChainGrad, tWeights, empty, grads, false);
    FilterOps::conv1(zpChainGrad, tWeights, empty, grads);
    FilterOps::corr1Weights(input, chainGradT, gradsW);

    //// GRADIENT CHECK
    ////gradCheck(chainGradTensor);
    ////gradCheckX(chainGradTensor);
    return grads;
}