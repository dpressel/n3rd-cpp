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
