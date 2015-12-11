//
// Created by Daniel on 9/24/2015.
//

#include "n3rd/FFTOps.h"
#include <iostream>
#include <cassert>
#include <cstring>

using namespace sgdtk;
using namespace n3rd;

int FFTOps::nextPowerOf2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}
void FFTOps::fftfilt1(const Tensor& x, const Tensor& y, const Tensor &biases, Tensor& z, bool corr)
{

    // For Tensor x, we have {featureMaps, embeddings, iT}
    // For Tensor y, we have {outputFeatureMaps, featureMaps, embeddings, iT}
    const int xsz = x.dims[2];
    const int embedSz = x.dims[1];
    const int nK = y.dims[0];
    const int kL = y.dims[1];
    const int ysz = y.dims[3];

    // fft size
    const int wide = nextPowerOf2(xsz + ysz - 1);

    // valid-region convolution/cross-correlation size
    const int narrow = xsz - ysz + 1;

    assert(z.dims[0] == nK);
    assert(z.dims[1] == embedSz);
    assert(z.dims[2] == narrow);

    fftw_plan p, pinv;

    // Buffers for FFTs
    int doubleWide = 2 * sizeof(fftw_complex) * wide;
    if (xy == nullptr || currentWidth < doubleWide)
    {

        std::cout << "Expanding fft buffer to " << doubleWide << std::endl;
        xy = (fftw_complex*) fftw_malloc(doubleWide);
        assert(xy);
        currentWidth = doubleWide;
        destroyAllPlans();

    }

    const auto& plan = plans.find(wide);

    if (plan == plans.end()) {


        int rank = 1;
        int n[] = {wide};
        int howmany = 2;
        int istride = 1;
        int ostride = 1;
        int idist = wide;
        int odist = wide;
        int *inembed = n;
        int *onembed = n;

        p = fftw_plan_many_dft(rank, n, howmany,
                               xy, inembed,
                               istride, idist,
                               xy, onembed,
                               ostride, odist,
                               FFTW_FORWARD, FFTW_ESTIMATE);

        howmany = 1;

        pinv = fftw_plan_many_dft(rank, n, howmany,
                                  xy, inembed,
                                  istride, idist,
                                  xy, onembed,
                                  ostride, odist,
                                  FFTW_BACKWARD, FFTW_ESTIMATE);
        std::cout << "Creating plan of size " << wide << std::endl;
        plans[wide] = std::make_pair(p, pinv);
    }
    else
    {
        //std::cout << "Reusing plan" << plan->first << std::endl;
        p = plan->second.first;
        pinv = plan->second.second;
    }

    for (int k = 0, kbase = 0; k < nK; ++k, kbase += embedSz) {
        const double bias = biases.empty() ? 0.0 : biases[k];


        for (int ei = 0; ei < embedSz; ++ei) {
            const int outAddr0 = (kbase + ei) * narrow;
            for (int i = 0; i < narrow; ++i) {
                z[outAddr0 + i] = bias;
            }

            for (int l = 0, lbase = 0; l < kL; ++l, lbase += embedSz) {
                const int dataAddr0 = (lbase + ei) * xsz;
                const int kernAddr0 = ((k * kL + l) * embedSz + ei) * ysz;

                memset(xy, 0, doubleWide);

                // copy x into a complex array
                for (int i = 0; i < xsz; ++i) {
                    xy[i][0] = x[dataAddr0 + i];
                }

                // copy y into a complex array
                if (corr) {
                    for (int i = 0, j = wide; i < ysz; ++i, ++j) {
                        xy[j][0] = y[kernAddr0 + i];
                    }
                }
                else {
                    for (int i = 0, j = wide; i < ysz; ++i, ++j) {
                        xy[j][0] = y[kernAddr0 + ysz - i - 1];
                    }
                }

                fftw_execute(p);

                // conj, followed by complex multiply
                for (int i = 0, j = wide; i < wide; i++, j++) {

                    xy[j][1] = -xy[j][1];
                    double xwr = xy[i][0];
                    double xwi = xy[i][1];
                    xy[i][0] = xwr * xy[j][0] - xwi * xy[j][1];
                    xy[i][1] = xwr * xy[j][1] + xwi * xy[j][0];
                }

                // IFFT
                //p = fftw_plan_dft_1d(wide, xy, xy, FFTW_BACKWARD, FFTW_ESTIMATE);
                fftw_execute(pinv);

                // Copy to output
                for (int i = 0; i < narrow; ++i) {
                    double re = xy[i][0] / wide;
                    z[outAddr0 + i] += re;

                }
            }
        }
    }
    // Cleanup
    //if (xy != nullptr)
    //{
    //    fftw_free(xy);
    //    xy = nullptr;
    //}

    // Go home

}

void FFTOps::destroyAllPlans() {
    for (auto& plan : plans) {
        fftw_destroy_plan((plan.second.first));
        fftw_destroy_plan((plan.second.second));

    }
    plans.clear();
}
