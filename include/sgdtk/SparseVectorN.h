#ifndef __SGDTK_CPP_SPARSEVECTORN_H__
#define __SGDTK_CPP_SPARSEVECTORN_H__

#include "sgdtk/Types.h"
#include "sgdtk/VectorN.h"

namespace sgdtk
{
    class SparseVectorN : public VectorN
    {
        Offsets offsets;
    public:

        SparseVectorN();
        SparseVectorN(const VectorN &source);

        SparseVectorN& operator=(const VectorN &v);
        SparseVectorN& operator=(const SparseVectorN &v);
        ~SparseVectorN();

        int length() const
        {
            int sz = offsets.size();
            return sz == 0 ? 0 : (offsets[sz - 1].first + 1);
        }

        int realIndex(int i) const;

        void add(Offset offset)
        {
            offsets.push_back(offset);
        }

        void set(int i, double v);


        Offsets getNonZeroOffsets() const
        {
            return offsets;
        }

        double at(int i) const
        {
            int j = realIndex(i);
            return j < 0 ? 0. : offsets[j].second;
        }

        void from(const VectorN &source);

        void organize();

        double dot(const VectorN &vec);
    };
}
#endif
