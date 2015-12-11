#ifndef __SGDTK_CPP_TENSORI_H__
#define __SGDTK_CPP_TENSORI_H__

#include <vector>
#include <sgdtk/Exception.h>
#include <cblas.h>

namespace sgdtk
{

    typedef double Real;
    class TensorI {
    public:

        TensorI() { }

        virtual ~TensorI() { }

        //virtual double mag() const = 0;

        virtual void scale(double scalar) = 0;

        virtual void resize(const std::vector<int> &dimensions) = 0;

        virtual void resize(const std::vector<int> &dimensions, Real cv) = 0;

        virtual void reshape(const std::vector<int> &newDimensions) throw(Exception) = 0;

        virtual void constant(Real x) = 0;

        //virtual void reset(const std::vector<Real> &x, const std::vector<int> &dimensions) = 0;

        virtual int size() const = 0;

        virtual bool empty() const = 0;

        //void scale(Real x)
        //{
        //    for (int i = 0, sz = d.size(); i < sz; ++i)
        //    {
        //        d[i] *= x;
        //    }
        //}
        virtual void add(const TensorI &x) = 0;

        virtual std::string getStorageType() const = 0;
    };
}

#endif