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
        T* d;
        CudaArray() : sz(0), d(NULL)
        {

        }
        CudaArray(int length) : sz(length)
        {
            d = sgdtk::createGPUArray<T>(sz * sizeof(T));
        }

        ~CudaArray()
        {
            if (d)
            {
                cudaFree(d);
            }
        }

        void zeros()
        {
            TRY_CUDA(cudaMemset(d, 0, sizeof(Real) * size()));
        }

        void constant(T x)
        {
            if (x == 0.0)
            {
                zeros();
                return;
            }
            std::vector<T> copy(size(), x);
            fromCPU(copy, false);
        }
        void resize(int length)
        {
            sz = length;
            if (d)
            {
                TRY_CUDA(cudaFree(d));
            }
            d = sgdtk::createGPUArray<T>(sz * sizeof(T));
        }

        int size() const
        {
            return sz;
        }
        void fromCPU(const std::vector<T>& cpuArray, bool doResize = true)
        {
            if (doResize)
            {
                resize(cpuArray.size());
            }
            else
            {
                int sz = size();
                int csz = cpuArray.size();
                if (csz != sz)
                {
                    throw Exception("Sizes must match when resize = false");
                }
            }
            copyArrayToGPU(d, cpuArray);

        }
        void toCPU(std::vector<T>& cpuArray, bool doResize = true) const
        {
            if (doResize)
            {
                cpuArray.resize(size());
            }
            else
            {
                int sz = size();
                int csz = cpuArray.size();
                if (csz != sz)
                {
                    throw Exception("Sizes must match when resize = false");
                }
            }
            copyArrayFromGPU(d, cpuArray);
        }

    };

    class CudaTensor : public TensorI
    {

    public:

        std::vector<int> dims;
        Real* d;

        CudaTensor() : d(NULL)
        {
            dims = {0};
        }

        explicit CudaTensor(const Tensor& t)
        {
            d = createGPUArrayFromArray<Real>(t);
            dims = t.dims;
        }
        explicit CudaTensor(const CudaTensor& t)
        {
            resize(t.dims);
            copyGPUToGPU(d, t.d, t.size());

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
                ///std::cout << "Deleting tensor" << std::endl;
                cudaFree(d);
            }
        }

        CudaTensor& operator=(const Tensor& t)
        {
            if (d)
            {
                ///std::cout << "Deleting tensor 2" << std::endl;
                cudaFree(d);
            }
            d = createGPUArrayFromArray<Real>(t);
            dims = t.dims;
            return *this;
        }

        CudaTensor& operator=(const CudaTensor& t)
        {
            if (this != &t)
            {
                resize(t.dims);
                copyGPUToGPU(d, t.d, t.size());
            }
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


        void resize(const std::vector<int>& dimensions)
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
        void resize(const std::vector<int>& dimensions, Real cv)
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
            constant(cv);

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

        void zeros()
        {
            TRY_CUDA(cudaMemset(d, 0, sizeof(Real) * size()));
        }

        // Inefficient except for 0!
        void constant(Real x)
        {
            if (x == 0.0)
            {
                zeros();
                return;
            }
            Tensor copy(dims);
            copy.constant(x);
            fromCPU(copy, false);
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

        void add(const TensorI& t)
        {
            const CudaTensor& cudaTensor = (const CudaTensor&)t;

            double one = 1.;
            TRY_CUBLAS(
                    cublasDgeam(Globals::gBlasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1, size(), &one, d, 1, &one, cudaTensor.d, 1, d, 1)
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
        void toCPU(Tensor& cpuTensor, bool doResize = true) const
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

        std::string getStorageType() const
        {
            return "CUDA";
        }
        bool empty() const
        {
            return size() < 1;
        }
    };


}

#endif