#include "n3rd/TemporalConvolutionalLayerBlas.h"
#include <cassert>
#include <cblas.h>

using namespace n3rd;

TemporalConvolutionalLayerBlas::TemporalConvolutionalLayerBlas(int nK, int kL, int kW, int embeddingSize)
{
    this->nK = nK;
    this->kL = kL;
    this->kW = kW;
    this->embedSz = embeddingSize;

    weights.resize({kL * embeddingSize * kW, nK * embeddingSize});
    gradsW.resize({kL * embeddingSize * kW, nK * embeddingSize});
    biases.resize(nK);
    biasGrads.resize(nK);

    int pitch = weights.dims[0];


    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double stdv = 1. / std::sqrt(6. / 28.);
    double stdv2 = stdv * 2;

    // For each kernel, randomly initialize all weights
    for (int j = 0; j < embedSz; ++j) {

        for (int i = 0; i < kL; ++i) {
            for (int k = 0; k < nK; ++k) {
                for (int m = 0; m < kW; ++m) {
                    // we need to get eSz * j + i row and eSz * j + k col
                    //
                    int row = (j * kL + i) * kW + m;
                    int col = k * embedSz + j;
                    int addr = col * pitch + row;
                    double number = distribution(generator);
                    double d = number * stdv2 - stdv;

                    weights.d[addr] = d;
                }
            }

        }
    }


}

void TemporalConvolutionalLayerBlas::reorderOutput(sgdtk::Tensor& unwrapped, int nK, int embedSz)
{
    //Tensor output = new Tensor(oT, nK * embedSz);
    // We have committed to unwrapping our output matrix to the form
    int oT = unwrapped.dims[0];
    unwrapped.reshape({nK, oT, embedSz});
    std::vector<double> out(nK * oT * embedSz);

    for (int k = 0; k < nK; ++k) {
        for (int i = 0; i < oT; ++i) {
            for (int j = 0; j < embedSz; ++j) {
                int nIdx = (k * oT + i) * embedSz + j;
                int cIdx = (k * embedSz + j) * oT + i;
                //int cIdx = (j * nK + k) * oT + i;
                double old = unwrapped.d[cIdx];
                //unwrapped.d[cIdx] = unwrapped.d[nIdx];
                //unwrapped.d[nIdx] = old;
                out[nIdx] = old;
            }
        }
    }
    unwrapped.d = out;
}


void TemporalConvolutionalLayerBlas::unwrapGrad(const sgdtk::Tensor& chainGrad, int nK, int embedSz, sgdtk::Tensor& unwrapped)
{
    const int oT = chainGrad.dims[1];
    unwrapped.resize({oT, nK * embedSz});

    // You could also do nIdx++ I think
    for (int k = 0; k < nK; ++k)
    {
        for (int i = 0; i < oT; ++i)
        {
            for (int j = 0; j < embedSz; ++j)
            {
                int nIdx = (k * oT + i) * embedSz + j;
                int cIdx = (k * embedSz + j) * oT + i;

                unwrapped.d[cIdx] = chainGrad.d[nIdx];
            }
        }
    }
}

void TemporalConvolutionalLayerBlas::unwrapX(const sgdtk::Tensor& x, int kW, sgdtk::Tensor& unwrapped)
{
    const int kL = x.dims[0];
    const int iT = x.dims[1];
    const int embedSz = x.dims[2];
    const int oT = iT - kW + 1;
    unwrapped.resize({oT, kW * kL * embedSz});

    // In the blas case, we need to write in column major, which means write down one lag, then move up to the next
    int n = 0;
    for (int j = 0; j < embedSz; ++j)
    {
        for (int k = 0; k < kL; ++k)
        {
            for (int m = 0; m < kW; ++m)
            {
                for (int i = 0; i < oT; ++i)
                {
                    int offset = (k * iT + i + m) * embedSz + j;
                    // x(kL, iT, embedSz)
                    unwrapped.d[n++] = x.d[offset];
                }
            }
        }
    }
}

void TemporalConvolutionalLayerBlas::wrapX(const sgdtk::Tensor& unwrapped, sgdtk::Tensor& grads, int kW)
{

    const int oT = unwrapped.dims[0];
    const int iT = oT + kW - 1;
    assert(iT == grads.dims[1]);
    const int kL = grads.dims[0];
    const int embedSz = grads.dims[2];
    assert(unwrapped.dims[1] / kW / kL == embedSz);


    // In the blas case, we need to write in column major, which means write down one lag, then move up to the next
    int n = 0;
    for (int j = 0; j < embedSz; ++j)
    {
        for (int k = 0; k < kL; ++k)
        {
            for (int m = 0; m < kW; ++m)
            {
                for (int i = 0; i < oT; ++i)
                {
                    int offset = (k * iT + i + m) * embedSz + j;
                    // x(kL, iT, embedSz)
                    grads.d[offset] += unwrapped.d[n++];
                }
            }
        }
    }

}

sgdtk::Tensor& TemporalConvolutionalLayerBlas::forward(const sgdtk::Tensor& z)
{
    // For convolutions, we should assume that our VectorN is truly a matrix
    // and the usual math applies

    numFrames = z.size() / embedSz / kL;
    input.reset(z.d, {kL, numFrames, embedSz});
    grads.resize({kL, numFrames, embedSz});
    grads.constant(0.);
    const int oT = numFrames - kW + 1;
    output.resize({oT, nK*embedSz});
    output.constant(0.);
    unwrapX(input, kW, unwrappedInput);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, unwrappedInput.dims[0], weights.dims[1],
                unwrappedInput.dims[1], 1.0, &unwrappedInput.d[0], unwrappedInput.dims[0],
                &weights.d[0], weights.dims[0], 0, &output.d[0], output.dims[0]);

    reorderOutput(output, nK, embedSz);
    return output;

}


sgdtk::Tensor& TemporalConvolutionalLayerBlas::backward(sgdtk::Tensor &chainGrad, double y)
{
    const int oT = numFrames - kW + 1;

    std::vector<int> outputDims = { nK, oT, embedSz };

    sgdtk::Tensor unwrappedChainGrad;
    unwrapGrad(chainGrad, nK, embedSz, unwrappedChainGrad);
    std::vector<int> unwrappedGradDims = {oT, kW * kL * embedSz};
    sgdtk::Tensor unwrappedGradInput(unwrappedGradDims);

    int m = unwrappedChainGrad.dims[0];
    int k = unwrappedChainGrad.dims[1];
    int n = weights.dims[0];

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0,
                &unwrappedChainGrad.d[0], m, &weights.d[0], n, 0,
                &unwrappedGradInput.d[0], m);

    m = unwrappedInput.dims[1];
    k = unwrappedInput.dims[0];
    n = unwrappedChainGrad.dims[1];

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0, &unwrappedInput.d[0], k,
                &unwrappedChainGrad.d[0], k, 0, &gradsW.d[0], m);

    // Because of the way we unrolled the embeddings matrix, we actually are allowing the previous computation
    // to calculate values that can't and should'nt be applied to the weight
    for (int i = 0, sz = weights.size(); i < sz; ++i)
    {
        if (weights.d[i] == 0.0)
        {
            gradsW.d[i] = 0.;
        }
    }

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


    wrapX(unwrappedGradInput, grads, kW);

    return grads;
}