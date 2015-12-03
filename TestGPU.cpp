#include "sgdtk/sgdtk.h"

#include <iostream>
#include <map>
#include "n3rd/FullyConnectedLayer.h"
#include "sgdtk/CudaTensor.h"
#include <cassert>
using namespace sgdtk;
using namespace n3rd;


#define assertEquals(X, Y) assert(X == Y)

#define assertEqualsF(X, Y, EPS) assert(std::abs(X - Y) < EPS)

#define EVAL(X) std::cout << #X << std::endl; X
std::vector<double> K = {1, 2, 3, 4, 5, 6, 7, 8};
const int M = 2;
const int N = 4;
std::vector<double> V_4 = {0.4, 0.3, 0.2, 0.1};
std::vector<double> V_2 = {2, 6};
std::vector<double> D4_X = {32.0, 40.0, 48.0, 56.0};
std::vector<double> W_G = {0.8, 0.6, 0.4, 0.2,
                           2.4, 1.8, 1.2, 0.6};

std::vector<double> W_G2 = {0.8, 2.4, 0.6, 1.8, 0.4, 1.2, 0.2, 0.6};
std::vector<double> D2_X = {3, 4};



void testCudaScale() throw(Exception)
{
   // Tensor t2(V_2, {1, 1, 2});
 //   Tensor t4(V_4, {1, 1, 4});



    CudaTensor ct2(V_2, {1,1,2});
    CudaTensor ct4(V_4, {1,1,4});

    ct2.scale(4);
    Tensor t2({1,1,2});
    Tensor t4({1,1,4});
    ct2.toCPU(t2);
    for (int i = 0; i < t2.size(); ++i)
    {
        assertEqualsF(t2[i], 4*V_2[i], 0.00001);
    }

    ct4.scale(0.3);
    ct4.toCPU(t4);
    for (int i = 0; i < t4.size(); ++i)
    {
        assertEqualsF(t4[i], 0.3*V_4[i], 0.00001);
    }


}


void testCudaAdd() throw(Exception)
{

    Tensor ht4(V_4, {1,4});
    Tensor ht4x(D4_X, {1,4});
    CudaTensor t4(ht4);

    CudaTensor t4x(ht4x);

    t4.add(t4x);

    t4.toCPU(ht4);
    for (int i = 0; i < t4.size(); ++i)
    {
        assertEqualsF(ht4[i]-D4_X[i], V_4[i], 0.00001);
    }

}




int main(int argc, char **argv)
{

    try
    {
        initCuBlas();

        EVAL(testCudaScale());
        EVAL(testCudaAdd());
        std::cout << "SUCCESS!" << std::endl;
    }
    catch (Exception &ex)
    {
        std::cout << "FAILURE: " << ex.getMessage() << std::endl;
    }

    doneCuBlas();
    return 0;
}