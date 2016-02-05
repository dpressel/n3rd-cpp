#include "sgdtk/sgdtk.h"

#include <iostream>
#include <map>


#include "n3rd/MaxPoolingLayer.h"
#include "n3rd/MaxPoolingLayerCuda.h"
using namespace sgdtk;
using namespace n3rd;

#include <cassert>

#define assertEquals(X, Y) assert(X == Y)

#define assertEqualsF(X, Y, EPS) assert(std::abs(X - Y) < EPS)

#define EVAL(X) std::cout << #X << std::endl; X

// embeddings ares
std::vector<double> D126 =
        {
                1,3,5,7,9,11,
                2,4,6,8,10,12
        };

std::vector<double> D126_13DS_MAX = {
        5, 11,
        6, 12
};
std::vector<int> D126_13DS_OR = {
        2, 5,
        8, 11
};

// 4, 1, 2
std::vector<double> D412 = {34,22,
                           46,58,
                           94,64,
                           124,154};

// 4, 1, 2     dh=1, dw=2
std::vector<double> D412_12DS_MAX = { 34, 58, 94, 154 };

std::vector<int> D412_12DS_OR = {0, 3, 4, 7};

std::vector<double> D133 = { 78.5, 120.5, -162.5,
                            -204.5, 252.5, 366.5,
                            -480.5, -594.5, 0};

std::vector<int> D313_22DS_OR = {4, 5, 6, 8};

std::vector<double> D313_22DS_MAX = {252.5, 366.5, -480.5, 0};


bool testIfOriginContains(int i, std::vector<int> origin, const Tensor& back)
{
    for (int x : origin)
    {
        if (x == i)
        {
            assertEquals(-86, back[i]);
            return true;
        }
    }
    return false;
}

void testOnCPU(const std::vector<double>& array, int k, int m, int n, int dh, int dw, const std::vector<double>& truth, const std::vector<int>& indices)
{

    std::unique_ptr<MaxPoolingLayer> l(new MaxPoolingLayer(dh, dw, {k, m, n}));

    std::cout << "Testing " << k << " x " << m << " x " << n << std::endl;
    Tensor input(array, {k, m, n});
    Tensor& output = (Tensor&)l->forward(input);
    assertEquals(output.size(), truth.size());
    for (int i = 0; i < output.size(); ++i)
    {
        assertEqualsF(output[i], truth[i], 0.00001);
    }

    std::vector<int> origin = l->getOrigin();

    assertEquals(origin.size(), indices.size());
    for (int i = 0; i < origin.size(); ++i)
    {
        assertEqualsF(origin[i], indices[i], 0.00001);
    }

    std::cout << " (forward)" << std::endl;
    std::cout << "\tPASS" << std::endl;

    Tensor chainGrad(output.dims);
    chainGrad.constant(-86);

    Tensor& back = (Tensor&)l->backward(chainGrad, 1);

    for (int i = 0; i < back.size(); ++i)
    {

        if (!testIfOriginContains(i, origin ,back))
        {
            assertEquals(0, back[i]);
        }

    }
    std::cout << " (backward)" << std::endl;
    std::cout << "\tPASS" << std::endl;

}


void testOnGPU(const std::vector<double>& array, int k, int m, int n, int dh, int dw, const std::vector<double>& truth, const std::vector<int>& indices) {
    std::unique_ptr<MaxPoolingLayerCuda> l(new MaxPoolingLayerCuda(dh, dw, {k, m, n}));

    std::cout << "Testing " << k << " x " << m << " x " << n << std::endl;
    Tensor input(array, {k, m, n});
    CudaTensor dInput(input);
    CudaTensor &dOutput = (CudaTensor &) l->forward(dInput);
    Tensor output;
    dOutput.toCPU(output);

    assertEquals(output.size(), truth.size());
    for (int i = 0; i < output.size(); ++i) {
        assertEqualsF(output[i], truth[i], 0.00001);
    }

    const CudaArray<int> &dOrigin = l->getOrigin();

    std::vector<int> origin;
    dOrigin.toCPU(origin);

    assertEquals(origin.size(), indices.size());
    for (int i = 0; i < m; ++i) {
        assertEqualsF(origin[i], indices[i], 0.00001);
    }
    std::cout << " (forward)" << std::endl;
    std::cout << "\tPASS" << std::endl;


    CudaTensor dChainGrad(output.dims);
    // Totally broken!!!
    dChainGrad.constant(-86);

    Tensor blah;
    dChainGrad.toCPU(blah);
    CudaTensor &dBack = (CudaTensor &) l->backward(dChainGrad, 1);

    Tensor back;

    dBack.toCPU(back);

    for (int i = 0; i < back.size(); ++i)
    {

        if (!testIfOriginContains(i, origin, back))
        {
            assertEquals(0, back[i]);
        }

    }
    std::cout << " (backward)" << std::endl;
    std::cout << "\tPASS" << std::endl;

}

void testCPU() throw(Exception)
{
    testOnCPU(D126, 1, 2, 6, 1, 3, D126_13DS_MAX, D126_13DS_OR);
    testOnCPU(D412, 4, 1, 2, 1, 2, D412_12DS_MAX, D412_12DS_OR);
    testOnCPU(D133, 1, 3, 3, 2, 2, D313_22DS_MAX, D313_22DS_OR);
}

void testGPU() throw(Exception)
{
    testOnGPU(D126, 1, 2, 6, 1, 3, D126_13DS_MAX, D126_13DS_OR);
    testOnGPU(D412, 4, 1, 2, 1, 2, D412_12DS_MAX, D412_12DS_OR);
    testOnGPU(D133, 1, 3, 3, 2, 2, D313_22DS_MAX, D313_22DS_OR);
}

int main(int argc, char **argv)
{

    try
    {

        EVAL(testCPU());
        EVAL(testGPU());

        std::cout << "SUCCESS!" << std::endl;
    }
    catch (Exception &ex)
    {
        std::cout << "FAILURE: " << ex.getMessage() << std::endl;
    }

    return 0;
}