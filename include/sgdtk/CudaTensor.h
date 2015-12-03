#ifndef __SGDTK_CPP_CUDA_TENSOR_H__
#define __SGDTK_CPP_CUDA_TENSOR_H__

#include <vector>
#include <sgdtk/Exception.h>
#include <sgdtk/GPU.h>
#include <sgdtk/Tensor.h>


namespace sgdtk
{

    typedef double Real;


    template<typename T> struct CudaArray
    {
        int sz;
        T* array;
        CudaArray() : sz(0), array(NULL)
        {

        }
        CudaArray(int length) : sz(length)
        {
            array = sgdtk::createGPUArray<T>(sz * sizeof(T));
        }

        ~CudaArray()
        {
            if (array)
            {
                cudaFree(array);
            }
        }
        void resize(int length)
        {
            sz = length;
            if (array)
            {
                TRY_CUDA(cudaFree(array));
            }
            array = sgdtk::createGPUArray<T>(sz * sizeof(T));
        }

    };

    class CudaTensor
    {

    public:

        std::vector<int> dims;
        Real* d;

        CudaTensor() : d(NULL)
        {
        }

        explicit CudaTensor(const Tensor& t)
        {
            d = createGPUArrayFromArray<Real>(t);
            dims = t.dims;
        }
        /*explicit CudaTensor(const std::vector<int>& dims)
        {
            this->dims = dims;
            int sz = size() * sizeof(Real);
            TRY_CUDA(cudaMalloc(&d, sz));
        }*/
        ~CudaTensor()
        {
            if (d != NULL)
            {
                //std::cout << "Deleting tensor" << std::endl;
                cudaFree(d);
            }
        }

        CudaTensor& operator=(const Tensor& t)
        {
            if (d)
            {
                //std::cout << "Deleting tensor 2" << std::endl;
                cudaFree(d);
            }
            d = createGPUArrayFromArray<Real>(t);
            dims = t.dims;
            return *this;
        }
        // No need for varargs in C++ 11, just pass initializer lists
        CudaTensor(const std::vector<Real>& x, const std::vector<int>& dimensions)
        {
            d = createGPUArrayFromArray<Real>(x);
            dims = dimensions;

        }


        CudaTensor(const std::vector<int>& dimensions) : d(NULL)
        {
            resize(dimensions);
        }


        void resize(const std::vector<int>& dimensions, Real cv = 0.)
        {

            dims = dimensions;

            int length = 1;
            for (int dim : dims)
            {
                length *= dim;
            }
            if (d != NULL)
            {
                TRY_CUDA(cudaFree(d));
            }
            d = createGPUArray<Real>(length);

        }

        void reshape(const std::vector<int>& newDimensions) throw(Exception)
        {
            int length = 1;
            int oldLength = size();
            for (int dim : newDimensions)
            {
                length *= dim;
            }

            dims = newDimensions;
            if (length != oldLength)
            {
                throw new sgdtk::Exception("Invalid shape!");
            }
        }

        void constant(Real x)
        {
            TRY_CUDA(cudaMemset(d, (int)x, sizeof(Real) * size()));
        }
        //void reset(const std::vector<Real>& x, const std::vector<int>& dimensions)
        //{
        //    dims = dimensions;
        //    d = x;
        //}

        int size() const
        {
            int length = 1;
            for (int dim : dims)
            {
                length *= dim;
            }
            return length;
        }

        void scale(Real x)
        {
            int n = size();
            cublasDscal(n, x, d, 1);
        }

        void add(CudaTensor& t)
        {


            double one = 1.;
            TRY_CUBLAS(
                    cublasDgeam(Globals::gBlasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1, size(), &one, d, 1, &one, t.d, 1, d, 1)
            );

        }

        void fromCPU(const Tensor& cpuTensor, bool doResize = true)
        {
            if (doResize)
            {
                resize(dims);
            }
            else
            {
                int sz = size();
                int csz = cpuTensor.size();
                if (csz != sz)
                {
                    throw Exception("Sizes must match when resize = false");
                }
            }
            copyArrayToGPU(d, cpuTensor);

        }
        void toCPU(Tensor& cpuTensor, bool doResize = true)
        {
            if (doResize)
            {
                cpuTensor.resize(dims);
            }
            else
            {
                int sz = size();
                int csz = cpuTensor.size();
                if (csz != sz)
                {
                    throw Exception("Sizes must match when resize = false");
                }
            }
            copyArrayFromGPU(d, cpuTensor);
        }
        //bool empty() const
        //{
        //    return d.empty();
        //}
    };


}

#endif