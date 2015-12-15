#ifndef __SGDTK_CPP_TENSOR_H__
#define __SGDTK_CPP_TENSOR_H__

#include <vector>
#include <sgdtk/Exception.h>
#include <cblas.h>
#include <sgdtk/TensorI.h>
namespace sgdtk
{

    typedef double Real;
    class Tensor : public TensorI
    {
    public:

        std::vector<int> dims;
        std::vector<Real> d;

        Tensor()
        {}

        Tensor(const Tensor& t)
        {
            d = t.d;
            dims = t.dims;
        }

        Tensor& operator=(const Tensor& t)
        {
            if (&t != this)
            {
                d = t.d;
                dims = t.dims;
            }
            return *this;
        }
        // No need for varargs in C++ 11, just pass initializer lists
        Tensor(const std::vector<Real>& x, const std::vector<int>& dimensions)
        {
            dims = dimensions;
            d = x;
        }


        Tensor(const std::vector<int>& dimensions)
        {
            resize(dimensions);
        }

        double mag() const;

        void scale(double scalar);

        void resize(const std::vector<int>& dimensions)
        {
            dims = dimensions;
            int length = 1;
            for (int dim : dims)
            {
                length *= dim;
            }
            d.resize(length, 0);
        }


        void resize(const std::vector<int>& dimensions, Real cv)
        {
            dims = dimensions;
            int length = 1;
            for (int dim : dims)
            {
                length *= dim;
            }
            d.resize(length, cv);
        }


        void reshape(const std::vector<int>& newDimensions) throw(Exception)
        {
            int length = 1;
            dims = newDimensions;
            for (int dim : newDimensions)
            {
                length *= dim;
            }
            if (length != d.size())
            {
                throw new sgdtk::Exception("Invalid shape!");
            }
        }

        void constant(Real x)
        {
            for (int i = 0, sz = d.size(); i < sz; ++i)
            {
                d[i] = x;
            }
        }
        void reset(const std::vector<Real>& x, const std::vector<int>& dimensions)
        {
            dims = dimensions;
            d = x;
        }

        const double& operator[](int i) const
        {
            return d[i];
        }
        double& operator[](int i)
        {
            return d[i];
        }
        int size() const
        {
            return d.size();
        }

        bool empty() const
        {
            return d.empty();
        }

        //void scale(Real x)
        //{
        //    for (int i = 0, sz = d.size(); i < sz; ++i)
        //    {
        //        d[i] *= x;
        //    }
        //}
        void add(const TensorI& x)
        {
            const Tensor& tensor = (const Tensor&)x;
            int sz = x.size();
            if (sz != d.size())
            {
                throw new sgdtk::Exception("Invalid shape!");
            }
            for (int i = 0; i < sz; ++i)
            {
                d[i] += tensor[i];
            }
        }

        std::string getStorageType() const
        {
            return "vector";
        }
    };

    inline void transposeWeight4D(const Tensor& weight, Tensor& weightCopy)
    {

        std::vector<int> newDims(weight.dims.size());

        newDims[0] = weight.dims[1];
        newDims[1] = weight.dims[0];
        newDims[2] = weight.dims[2];
        newDims[3] = weight.dims[3];

        // If either feature map is one, no need to copy, memory is the same
        if (weight.dims[0] == 1 || weight.dims[1] == 1)
        {
            weightCopy = weight;
            weightCopy.reshape(newDims);
            return;
        }
        weightCopy.resize(newDims);


        int sz = newDims[2] * newDims[3];
        for (int i = 0; i < newDims[0]; ++i)
        {
            for (int j = 0; j < newDims[1]; ++j)
            {
                for (int k = 0; k < sz; ++k)
                {
                    weightCopy.d[(i * newDims[1] + j) * sz + k] = weight.d[(j * weight.dims[1] + i) * sz + k];
                }
            }
        }
    }

    //Tensor* embed(const Tensor* tensor, int h, int w);
    inline void embed(const Tensor& tensor, int l, int h, int w, Tensor& zpCube)
    {
        const int tL = tensor.dims[0];
        const int tH = tensor.dims[1];
        const int tW = tensor.dims[2];
        const int oH = tH + h;
        const int oW = tW + w;
        const int oL = tL + l;
        zpCube.resize({oL, oH, oW});
        int lStart = l / 2;
        int hStart = h / 2;
        int wStart = w / 2;
        for (int k = 0; k < tL; ++k)
        {
            for (int i = 0; i < tH; ++i)
            {
                for (int j = 0; j < tW; ++j)
                {
                    zpCube.d[((k + lStart) * oH + i + hStart) * oW + j + wStart] = tensor.d[(k * tH + i) * tW + j];
                }
            }
        }

    }


}

#endif