#include "sgdtk/LinearModel.h"

using namespace sgdtk;


// TODO
void LinearModel::load(String file)
{

}

// TODO
void LinearModel::save(String file)
{

}

double LinearModel::predict(const FeatureVector* fv) const
{
    double dot = 0.;
    const Offsets& sv = fv->getNonZeroOffsets();

    for (Offsets::const_iterator p = sv.begin(); p != sv.end(); ++p)
    {
        dot += this->weights[p->first] * p->second;
    }
    return dot / wdiv + wbias;
}

Model* LinearModel::prototype() const
{
    return new LinearModel(*this);
}

double LinearModel::mag() const
{
    double dot = 0.;
    for (size_t i = 0, sz = weights.size(); i < sz; ++i)
    {
        dot += weights[i] * weights[i];
    }
    return dot / wdiv / wdiv;
}

void LinearModel::scaleInplace(double scalar)
{
    for (size_t i = 0, sz = weights.size(); i < sz; ++i)
    {
        weights[i] *= scalar;
    }
}

void LinearModel::addInplace(int i, double update)
{
    weights[i] += update;
}