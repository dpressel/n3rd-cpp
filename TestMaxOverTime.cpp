#include "sgdtk/sgdtk.h"

#include <iostream>
#include <map>


#include "n3rd/MaxOverTimePoolingLayer.h"
#include "n3rd/MaxOverTimePoolingLayerCuda.h"
using namespace sgdtk;
using namespace n3rd;

#include <cassert>

#define assertEquals(X, Y) assert(X == Y)

#define assertEqualsF(X, Y, EPS) assert(std::abs(X - Y) < EPS)

#define EVAL(X) std::cout << #X << std::endl; X

// embeddings ares
std::vector<double> D26 =
        {
                1,3,5,7,9,11,
                2,4,6,8,10,12
        };

std::vector<double> D26_MAX = {
        11,
        12
};
std::vector<int> D26_OR = {
        5,
        11
};

std::vector<double> D42 = {34,22,
                           46,58,
                           94,64,
                           124,154};

std::vector<double> D42_MAX = { 34, 58, 94, 154 };

std::vector<int> D42_OR = {0, 3, 4, 7};

std::vector<double> D33 = { 78.5, 120.5, -162.5,
                            -204.5, 252.5, 366.5,
                            -480.5, -594.5, 0};

std::vector<int> D33_OR = {1,5,8};

std::vector<double> D33_MAX = {120.5, 366.5, 0};




void testNOnCPU(const std::vector<double>& array, int m, const std::vector<double>& truth, const std::vector<int>& indices)
{

    std::unique_ptr<MaxOverTimePoolingLayer> l(new MaxOverTimePoolingLayer(m));
    int n = array.size()/m;
    std::cout << "Testing " << m << " x " << n << std::endl;
    Tensor d(array, {m, n});
    Tensor& output = (Tensor&)l->forward(d);
    assertEquals(output.size(), m);
    for (int i = 0; i < m; ++i)
    {
        assertEqualsF(output[i], truth[i], 0.00001);
    }
    std::vector<int> origin = l->getOrigin();
    assertEquals(origin.size(), m);
    for (int i = 0; i < m; ++i)
    {
        assertEqualsF(origin[i], indices[i], 0.00001);
    }
    std::cout << "\tPASS" << std::endl;

    Tensor mxn({m, n});
    for (int i = 0; i < m; ++i)
    {
        mxn[origin[i]] = 1;
    }

    Tensor chainGrad({m,n});
    chainGrad.constant(1);
    Tensor& back = (Tensor&)l->backward(chainGrad, 1);

    assertEquals(back.size(), chainGrad.size());
    for (int i = 0; i < back.size(); ++i)
    {
        assertEquals(mxn[i], back[i]);
    }


}


void testNOnGPU(const std::vector<double>& array, int m, const std::vector<double>& truth, const std::vector<int>& indices)
{

    std::unique_ptr<MaxOverTimePoolingLayerCuda> l(new MaxOverTimePoolingLayerCuda(m));
    int n = array.size()/m;
    std::cout << "Testing " << m << " x " << n << std::endl;
    Tensor d(array, {m, n});
    CudaTensor dD(d);

    CudaTensor& dOutput = (CudaTensor&)l->forward(dD);

    Tensor output;
    dOutput.toCPU(output);

    assertEquals(output.size(), m);
    for (int i = 0; i < m; ++i)
    {
        assertEqualsF(output[i], truth[i], 0.00001);
    }
    const CudaArray<int>& dOrigin = l->getOrigin();
    std::vector<int> origin;
    dOrigin.toCPU(origin);

    assertEquals(origin.size(), m);
    for (int i = 0; i < m; ++i)
    {
        assertEqualsF(origin[i], indices[i], 0.00001);
    }
    std::cout << "\tPASS" << std::endl;

    Tensor mxn({m, n});
    for (int i = 0; i < m; ++i)
    {
        mxn[origin[i]] = 1;
    }

    Tensor chainGrad({m,n});
    chainGrad.constant(1);
    CudaTensor dChainGrad(chainGrad);
    CudaTensor& dBack = (CudaTensor&)l->backward(dChainGrad, 1);

    Tensor back;
    dBack.toCPU(back);
    assertEquals(back.size(), chainGrad.size());
    for (int i = 0; i < back.size(); ++i)
    {
        assertEquals(mxn[i], back[i]);
    }


}

void testCPU() throw(Exception)
{
    testNOnCPU(D26, 2, D26_MAX, D26_OR);
    testNOnCPU(D42, 4, D42_MAX, D42_OR);
    testNOnCPU(D33, 3, D33_MAX, D33_OR);



}
void testGPU() throw(Exception)
{
    testNOnGPU(D26, 2, D26_MAX, D26_OR);
    testNOnGPU(D42, 4, D42_MAX, D42_OR);
    testNOnGPU(D33, 3, D33_MAX, D33_OR);
}

int main(int argc, char **argv)
{

    try
    {

        EVAL(testGPU());
        EVAL(testCPU());
        std::cout << "SUCCESS!" << std::endl;
    }
    catch (Exception &ex)
    {
        std::cout << "FAILURE: " << ex.getMessage() << std::endl;
    }

    return 0;
}