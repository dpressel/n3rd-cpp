#include "sgdtk/SparseVectorN.h"

using namespace sgdtk;


SparseVectorN::SparseVectorN()
{

}

SparseVectorN::SparseVectorN(const VectorN &source)
{
    this->from(source);
}


SparseVectorN::~SparseVectorN()
{ }


SparseVectorN& SparseVectorN::operator=(const VectorN &v)
{
    if (&v != this)
    {
        from(v);
    }
    return *this;
}

SparseVectorN& SparseVectorN::operator=(const SparseVectorN &v)
{
    if (&v != this)
    {
        offsets = v.offsets;
    }
    return *this;
}


int  SparseVectorN::realIndex(int i) const
{
    for (int j = 0, sz = offsets.size(); j < sz; ++j)
    {
        Offset offset = offsets[j];
        if (offset.first > i)
        {
            return -1;
        }
        else if (offset.first == i)
        {
            return j;
        }
    }
    return -1;
}
void SparseVectorN::set(int i, double v)
{
    int j = realIndex(i);
    if (j < 0)
    {
        add(Offset(i, v));
        //Collections.sort(offsets);
    }
    else
    {
        offsets[j] = Offset(i, v);
    }
}
void SparseVectorN::organize() {
    /*
     * Collections.sort(offsets);
        Set<Integer> seen = new HashSet<Integer>();
        List<Offset> clean = new ArrayList<Offset>(offsets.size());
        for (Offset offset: offsets)
        {
            if (!seen.contains(offset.index))
            {
                seen.add(offset.index);
                clean.add(offset);
            }
        }
        this.offsets = new ArrayList<Offset>(clean);
     */
}

double SparseVectorN::dot(const VectorN &vec)
{
    double acc = 0.;

    Offsets offsets = getNonZeroOffsets();

    for (int i = 0, sz = offsets.size(); i < sz; ++i)
    {
        acc += offsets[i].second * vec.at(offsets[i].first);
    }
    return acc;
}

void SparseVectorN::from(const VectorN& v)
{
    Offsets voff = v.getNonZeroOffsets();
    offsets = voff;

}