#include "n3rd/TemporalConvolutionalLayerCuBlas.h"
#include <cassert>

using namespace n3rd;

TemporalConvolutionalLayerCuBlas::TemporalConvolutionalLayerCuBlas(int nK, int kL, int kW)
{
    this->nK = nK;
    this->kL = kL;
    this->kW = kW;

    weights.resize({kL * kW, nK});
    dWeights.resize({kL * kW, nK});
    dWeightGrads.resize({kL * kW, nK});
    gradsW.resize({kL * kW, nK});
    biases.resize(nK);
    biasGrads.resize(nK);

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

void TemporalConvolutionalLayerCuBlas::reorderOutput(sgdtk::Tensor& unwrapped)
{
    //Tensor output = new Tensor(oT, nK * embedSz);
    // We have committed to unwrapping our output matrix to the form
    int oT =  numFrames - kW + 1;
    for (int k = 0; k < nK; ++k)
    {
        for (int i = 0; i < oT; ++i)
        {

            int nIdx = k * oT + i;
            int cIdx = i * nK + k;
            double tmp = unwrapped[nIdx];
            unwrapped[nIdx] = unwrapped[cIdx];
            unwrapped[cIdx] = tmp;
        }
    }
}


void TemporalConvolutionalLayerCuBlas::unwrapGradFromNextLayer(const sgdtk::Tensor& chainGrad, sgdtk::Tensor& unwrapped)
{
    const int oT = numFrames - kW + 1;

    // You could also do nIdx++ I think
    for (int k = 0; k < nK; ++k)
    {
        for (int i = 0; i < oT; ++i)
        {
            int nIdx = k * oT + i;
            int cIdx = i * nK + k;

            unwrapped[nIdx] = chainGrad[cIdx];
        }
    }
}

void TemporalConvolutionalLayerCuBlas::unwrapInput(const sgdtk::Tensor& x)
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
    dUnwrappedInput = unwrappedInput;
}

void TemporalConvolutionalLayerCuBlas::wrapGrad(const sgdtk::Tensor& unwrapped)
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

sgdtk::Tensor& TemporalConvolutionalLayerCuBlas::forward(const sgdtk::Tensor& z)
{
    // For convolutions, we should assume that our VectorN is truly a matrix
    // and the usual math applies

    //dWeights = weights;
    //dWeights.toCPU(weights, false);

    dWeights.fromCPU(weights, false);

    numFrames = z.size() / kL;
    //input = z;
    grads.resize({kL, 1, numFrames});

    //dGrads.resize(grads.dims);
    //dGrads.constant(0.);

    const int oT = numFrames - kW + 1;

    sgdtk::CudaTensor dOutput({nK, 1, oT});

    unwrapInput(z);

    double alpha = 1.0;
    double beta = 0.0;
    TRY_CUBLAS(cublasDgemm_v2(sgdtk::Globals::gBlasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                   dUnwrappedInput.dims[0], dWeights.dims[1], dUnwrappedInput.dims[1], &alpha, dUnwrappedInput.d, dUnwrappedInput.dims[0], dWeights.d, dWeights.dims[0], &beta, dOutput.d, oT));
    //cublasDgemm('N', 'N', dUnwrappedInput.dims[0], dWeights.dims[1], dUnwrappedInput.dims[1], 1.0, dUnwrappedInput.d, dUnwrappedInput.dims[0], dWeights.d, dWeights.dims[0], 0, dOutput.d, oT);
    //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, unwrappedInput.dims[0],
    //            weights.dims[1],
    //            unwrappedInput.dims[1], 1.0,
    //            &(unwrappedInput.d[0]),
    //            unwrappedInput.dims[0],
    //            &(weights.d[0]), weights.dims[0], 0,
    //            &(output.d[0]), oT);
    //TRY_CUBLAS(cublasGetError());
    dOutput.toCPU(output);
    reorderOutput(output);
    return output;

}


sgdtk::Tensor& TemporalConvolutionalLayerCuBlas::backward(sgdtk::Tensor &chainGrad, double y)
{
    const int oT = numFrames - kW + 1;

    std::vector<int> outputDims = { nK, 1, oT };
    dWeightGrads.constant(0.);
    sgdtk::Tensor unwrappedChainGrad({oT, nK});
    unwrapGradFromNextLayer(chainGrad, unwrappedChainGrad);
    sgdtk::CudaTensor dUnwrappedChainGrad(unwrappedChainGrad);

    std::vector<int> unwrappedGradDims = {oT, kW * kL};
    //sgdtk::Tensor unwrappedGradInput(unwrappedGradDims);
    sgdtk::CudaTensor dUnwrappedGradInput(unwrappedGradDims);


    int m = unwrappedChainGrad.dims[0];
    int k = unwrappedChainGrad.dims[1];
    int n = weights.dims[0];
    double alpha = 1.0;
    double beta = 0.0;

    TRY_CUBLAS(cublasDgemm_v2(sgdtk::Globals::gBlasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
                dUnwrappedChainGrad.d, m, dWeights.d, n, &beta,
                dUnwrappedGradInput.d, m));


    //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0,
    //            &(unwrappedChainGrad.d[0]), m, &(weights.d[0]), n, 0,
    //            &(unwrappedGradInput.d[0]), m);


    //sgdtk::copyArrayFromGPU(dUnwrappedGradInput.d, unwrappedGradInput);

    sgdtk::Tensor unwrappedGradInput;
    dUnwrappedGradInput.toCPU(unwrappedGradInput);

    m = unwrappedInput.dims[1];
    k = unwrappedInput.dims[0];
    n = unwrappedChainGrad.dims[1];

    TRY_CUBLAS(cublasDgemm_v2(sgdtk::Globals::gBlasHandle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, dUnwrappedInput.d, k, dUnwrappedChainGrad.d, k, &beta, dWeightGrads.d, m));

    dWeightGrads.toCPU(gradsW, false);


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


    wrapGrad(unwrappedGradInput);

    return grads;
}