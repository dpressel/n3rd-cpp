#ifndef __SGDTK_CPP_CUDNN_TENSOR_H__
#define __SGDTK_CPP_CUDNN_TENSOR_H__

#include <vector>
#include <sgdtk/Exception.h>
#include <sgdtk/GPU.h>
#include <sgdtk/Tensor.h>
#include <sgdtk/CudaTensor.h>

namespace sgdtk
{

    typedef double Real;


    class CuDNNTensor
    {

    public:

        cudnnTensorDescriptor_t descriptor;
        std::vector<int> dims;
        Real* d;

        CuDNNTensor() : d(NULL), descriptor(NULL)
        {
            TRY_CUDNN(cudnnCreateTensorDescriptor(&descriptor));
        }

        explicit CuDNNTensor(const Tensor& t)
        {
            TRY_CUDNN(cudnnCreateTensorDescriptor(&descriptor));
            int sz = t.size() * sizeof(Real);

            TRY_CUDA(cudaMalloc(&d, sz));
            //TRY_CUDA(cudaMemcpy(gpuPtr, &t[0], sz, cudaMemcpyHostToDevice));
            sgdtk::copyArrayToGPU(d, t);
            dims = t.dims;
            sgdtk::setTensorDescriptor(descriptor, sizeof(Real), dims);
        }
        ~CuDNNTensor()
        {
            if (d)
            {
                cudaFree(d);
            }
            if (descriptor)
            {
                cudnnDestroyTensorDescriptor(descriptor);
            }
        }

        CuDNNTensor& operator=(const Tensor& t)
        {
            if (d)
            {
                cudaFree(d);
            }
            d = createGPUArrayFromArray<Real>(t);
            dims = t.dims;
            sgdtk::setTensorDescriptor(descriptor, sizeof(Real), dims);

            return *this;
        }
        // No need for varargs in C++ 11, just pass initializer lists
        CuDNNTensor(const std::vector<Real>& x, const std::vector<int>& dimensions)
        {
            TRY_CUDNN(cudnnCreateTensorDescriptor(&descriptor));
            dims = dimensions;
            d = createGPUArrayFromArray<Real>(x);
            sgdtk::setTensorDescriptor(descriptor, sizeof(Real), dims);
        }


        CuDNNTensor(const std::vector<int>& dimensions) : d(NULL)
        {
            TRY_CUDNN(cudnnCreateTensorDescriptor(&descriptor));
            resize(dimensions);
        }


        void resize(const std::vector<int>& dimensions, Real cv = 0.)
        {
            dims.resize(std::max<size_t>(dimensions.size(), 3));
            int diff = dims.size() - dimensions.size();

            if (diff)
            {
                for (int i = 0; i < diff; ++i)
                {
                    dims[i] = 1;
                }
                for (int i = 0, j = diff; i < dims.size(); ++i, ++j)
                {
                    dims[j] = dimensions[i];
                }
            }
            else
            {
                dims = dimensions;
            }

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

            sgdtk::setTensorDescriptor(descriptor, sizeof(Real), dims);

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
            TRY_CUDNN(cudnnSetTensor(Globals::gHandle, descriptor, d, &x));
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
            TRY_CUDNN(cudnnScaleTensor(Globals::gHandle, descriptor, d, &x));
        }

        void add(CuDNNTensor& t)
        {
            int alpha = 1;
            int beta = 1;
            TRY_CUDNN(cudnnAddTensor_v3(Globals::gHandle, &alpha, t.descriptor, t.d, &beta, descriptor, d));
        }

        void toCPU(Tensor& cpuTensor)
        {
            cpuTensor.resize(dims);
            sgdtk::copyArrayFromGPU(d, cpuTensor);
        }
        //bool empty() const
        //{
        //    return d.empty();
        //}
    };


}

#endif