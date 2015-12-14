#include "n3rd/TemporalConvolutionalLayerBlas.h"
#include <cassert>
#include <cblas.h>

using namespace n3rd;

TemporalConvolutionalLayerBlas::TemporalConvolutionalLayerBlas(int nK, int kL, int kW)
{
    this->nK = nK;
    this->kL = kL;
    this->kW = kW;

    weights.resize({kL * kW, nK});
    gradsW.resize(weights.dims);
    weightAccum.resize(weights.dims, 0);
    biases.resize({nK}, 0);
    biasGrads.resize({nK}, 0);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double stdv = 1. / std::sqrt(6. / 28.);
    double stdv2 = stdv * 2;

    for (int i = 0, sz = weights.size(); i < sz; ++i)
    // For each kernel, randomly initialize all weights
    {
        double number = distribution(generator);
        double d = number * stdv2 - stdv;
        weights[i] = d;

    }


}

void TemporalConvolutionalLayerBlas::unwrapInput(const sgdtk::Tensor& x)
{
    const int oT = numFrames - kW + 1;
    unwrappedInput.resize({oT, kW * kL});
    int n = 0;


    for (int k = 0; k < kL; ++k)
    {

        for (int m = 0; m < kW; ++m)
        {
            for (int i = 0; i < oT; ++i)
            {

                int offset = k * numFrames + i + m;
                unwrappedInput[n] = x[offset];
                ++n;
            }
        }
    }
}

void TemporalConvolutionalLayerBlas::wrapGrad(const sgdtk::Tensor& unwrapped)
{

    const int oT = unwrapped.dims[0];
    const int iT = oT + kW - 1;
    assert(iT == grads.dims[2]);
    const int kL = grads.dims[0];
    const int embedSz = grads.dims[1];
    assert(1 == embedSz);


    // In the blas case, we need to write in column major, which means write down one lag, then move up to the next
    int n = 0;

    for (int k = 0; k < kL; ++k)
    {
        for (int m = 0; m < kW; ++m)
        {
            for (int i = 0; i < oT; ++i)
            {
                int offset = k * iT + i + m;
                    // x(kL, iT, embedSz)
                grads[offset] += unwrapped[n];
                n++;
            }
        }
    }
}

sgdtk::TensorI& TemporalConvolutionalLayerBlas::forward(const sgdtk::TensorI& z)
{
    // For convolutions, we should assume that our VectorN is truly a matrix
    // and the usual math applies

    const sgdtk::Tensor& zT = (const sgdtk::Tensor&)z;
    numFrames = z.size() / kL;
    input = zT;
    grads.resize({kL, 1, numFrames});
    grads.constant(0.);
    const int oT = numFrames - kW + 1;
    output.resize({nK, 1, oT});

    unwrapInput(input);
    output.constant(0.);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, unwrappedInput.dims[0],
                weights.dims[1],
                unwrappedInput.dims[1], 1.0,
                &(unwrappedInput.d[0]),
                unwrappedInput.dims[0],
                &(weights.d[0]), weights.dims[0], 0,
                &(output.d[0]), oT);

    //reorderOutput(output);
    return output;

}


sgdtk::TensorI& TemporalConvolutionalLayerBlas::backward(sgdtk::TensorI &chainGrad, double y)
{
    const int oT = numFrames - kW + 1;

    std::vector<int> outputDims = { nK, 1, oT };

    sgdtk::Tensor& chainGradT = (sgdtk::Tensor&)chainGrad;

    //sgdtk::Tensor unwrappedChainGrad({oT, nK});
    //unwrapGradFromNextLayer(chainGradT, unwrappedChainGrad);
    std::vector<int> unwrappedGradDims = {oT, kW * kL};
    sgdtk::Tensor unwrappedGradInput(unwrappedGradDims);

    int m = oT;
    int k = nK;
    int n = weights.dims[0];

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0,
                &(chainGradT.d[0]), m, &(weights.d[0]), n, 0,
                &(unwrappedGradInput.d[0]), m);



    m = unwrappedInput.dims[1];
    k = unwrappedInput.dims[0];
    n = nK;


    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0,
                &(unwrappedInput.d[0]), k,
                &(chainGradT.d[0]), k, 0, &(gradsW.d[0]), m);



        // We need to update gradsW, which are (kL * embeddingSize) * kW x (nK * embeddingSize);
        /*
        int stride = convOutputSz * embedSz;
        for (int l = 0; l < nK; ++l)
        {
            for (int i = 0; i < stride; ++i)
            {
                this.biasGrads[l] += chainGradX[l * stride + i];
            }
            this.biasGrads[l] /= embedSz;
        }*/


    wrapGrad(unwrappedGradInput);

    return grads;
}
