#include "sgdtk/sgdtk.h"
#include <fftw3.h>
#include <vector>
#include <iostream>
#include <map>
#include "n3rd/FullyConnectedLayer.h"


using namespace sgdtk;
using namespace n3rd;

#include <cassert>
#include <n3rd/FFTOps.h>

#define assertEquals(X, Y) assert(X == Y)

#define assertEqualsF(X, Y, EPS) assert(std::abs(X - Y) < EPS)

#define EVAL(X) std::cout << #X << std::endl; X

std::vector<double> D =
        {
                1, 2,
                3, 4,
                5, 6,
                7, 8,
                9, 10,
                11, 12
        };

std::vector<double> D2 = { 1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12 };

std::vector<double> DOFF = { 115, 126, 1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12, -42, -86 };

std::vector<double> K = { 1, 4, 2, 5, 3, 6 };
std::vector<double> KOFF = { -64, -102, 1, 4, 2, 5, 3, 6, -100, -10, -1, 0.9 };

std::vector<double> K2 = { 5, 3, 6 };


std::vector<double> O1 = { 22, 34, 46, 58 };
std::vector<double> O1NEG = {-35,  -58,  -79, -100, -115};

std::vector<double> xcorr(const std::vector<double>& x, const std::vector<double>& y)
{
    int xsz = x.size();
    int ysz = y.size();
    int narrow = xsz - ysz + 1;
    std::vector<double> z(narrow);

    for (int i = 0; i < narrow; ++i)
    {
        for (int j = 0; j < ysz; ++j)
        {
            z[i] += x[i + j] * y[j];
        }
    }
    return z;

}


std::vector<double> fftfilt(const std::vector<double>& x, const std::vector<double> y, bool corr)
{

    int xsz = x.size();
    int ysz = y.size();

    int wide = FFTOps::nextPowerOf2(xsz + ysz - 1);
    int narrow = xsz - ysz + 1;
    std::vector<double> z(narrow);

    fftw_plan p;

    fftw_complex* xwide = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * wide);
    fftw_complex* ywide = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * wide);


    for (int i = 0 ; i < wide; ++i)
    {
        xwide[i][0] = xwide[i][1] = ywide[i][0] = ywide[i][1] = 0.;
    }
    for (int i = 0; i < xsz; ++i)
    {
        xwide[i][0] = x[i];
        xwide[i][1] = 0;
    }

    if (corr)
    {
        for (int i = 0; i < ysz; ++i)
        {
            ywide[i][0] = y[i];
            ywide[i][1] = 0;
        }
    }
    else
    {
        for (int i = 0; i < ysz; ++i)
        {
            ywide[i][0] = y[ysz - i - 1];
        }
    }


    p = fftw_plan_dft_1d(wide, xwide, xwide, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(p);

    //p = fftw_plan_dft_1d(wide, ywide, ywide, FFTW_FORWARD, FFTW_MEASURE);

    //fftw_execute(p);
    fftw_execute_dft(p, ywide, ywide);
    for (int i = 0; i < wide; i++)
    {
        ywide[i][1] = -ywide[i][1];
        double xwr = xwide[i][0];
        double xwi = xwide[i][1];
        xwide[i][0] = xwr * ywide[i][0] - xwi * ywide[i][1];
        xwide[i][1] = xwr * ywide[i][1] + xwi * ywide[i][0];
    }

    p = fftw_plan_dft_1d(wide, xwide, xwide, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    for (int i = 0, j = 0; i < narrow; ++i, j += 2)
    {
        double re = xwide[i][0] / wide;
        z[i] = re;
    }

    fftw_destroy_plan(p);
    fftw_free(xwide); fftw_free(ywide);


    return z;
}


void testFFT1() throw(Exception)
{
    std::vector<double> x = { 1, 3, 5, 7, 9, 11 };
    std::vector<double> y = { 1, 2, 3 };

    std::vector<double> z = fftfilt(x, y, true);

    for (int i = 0; i < z.size(); ++i)
    {
        std::cout << z[i] << std::endl;
        assertEqualsF(z[i], O1[i], 1e-6);
    }

}

int main(int argc, char **argv)
{

    try
    {

        EVAL(testFFT1());
        std::cout << "SUCCESS!" << std::endl;
    }
    catch (Exception &ex)
    {
        std::cout << "FAILURE: " << ex.getMessage() << std::endl;
    }

    return 0;
}