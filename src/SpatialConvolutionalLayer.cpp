#include "n3rd/SpatialConvolutionalLayer.h"

using namespace n3rd;
using namespace sgdtk;

SpatialConvolutionalLayer::SpatialConvolutionalLayer(int nK, int kH, int kW, std::vector<int> inputDims)
{


    const int iL = inputDims.size() == 3 ? inputDims[0]: 1;
    const int iH = inputDims[1];
    const int iW = inputDims[2];

    // For now, let this just be a pointer to input
    input.resize(inputDims);

    weights.resize({nK, iL, kH, kW});
    weightAccum.resize({nK, iL, kH, kW});
    gradsW.resize({nK, iL, kH, kW});
    biases.resize({nK}, 0);
    biasGrads.resize({nK}, 0);
    grads.resize(inputDims);

    // For each kernel, randomly initialize all weights
    output.resize({nK, iH - kH + 1, iW - kW + 1});

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    double stdv = 1. / std::sqrt(grads.dims[1] * grads.dims[2]);

    double stdv2 = stdv * 2;

    // For each kernel, randomly initialize all weights
    for (int i = 0; i < weights.size(); ++i)
    {

        double number = distribution(generator);
        double d = number * stdv2 - stdv;
        weights[i] = d;
    }

}

sgdtk::TensorI& SpatialConvolutionalLayer::forward(const sgdtk::TensorI& z)
{

    const int nK = weights.dims[0];
    const int kL = weights.dims[1];
    const int embeddingSz = weights.dims[2];
    const int kW = weights.dims[3];

    const int numFrames = z.size() / embeddingSz / kL;
    const int oT = numFrames - kW + 1;
    const sgdtk::Tensor& zT = (const sgdtk::Tensor&)z;


    input = zT;
    input.reshape(grads.dims);
    FilterOps::corr2(input, weights, biases, output);

    return output;

}


sgdtk::TensorI& SpatialConvolutionalLayer::backward(sgdtk::TensorI& chainGrad, double y)
{

    sgdtk::Tensor& chainGradT = (sgdtk::Tensor&) chainGrad;
    //final int iL = input.dims[0];
    const int iH = input.dims[1];
    const int iW = input.dims[2];

    const int oH = output.dims[1];
    const int oW = output.dims[2];
    //final int kL = weights.dims[1];
    const int kH = weights.dims[2];
    const int kW = weights.dims[3];

    const int zpH = iH + kH - 1;
    const int zpW = iW + kW - 1;

    const int nK = weights.dims[0];

    chainGradT.reshape({nK, oH, oW});

    Tensor zpChainGradCube;
    embed(chainGradT, 0, zpH - oH, zpW - oW, zpChainGradCube);

    Tensor tWeights;

    transposeWeight4D(weights, tWeights);

    // This should NOT be required
    grads.constant(0.);
    // This is actually what is failing.  Why?  Probably a bug in transpose weight 4D?
    FilterOps::conv2(zpChainGradCube, tWeights, {}, grads);

    // This is correct, we know that the gradient of the weights is checking out
    FilterOps::corr2Weights(input, chainGradT, gradsW);

    return grads;
}