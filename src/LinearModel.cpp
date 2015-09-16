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
    auto dot = weights.mag();
    return dot / wdiv / wdiv;
}

void LinearModel::scaleInplace(double scalar)
{
    weights.scale(scalar);

}

double LinearModel::predict(const FeatureVector *fv) const
{
    double dot = 0.;
    const Offsets &sv = fv->getNonZeroOffsets();

    for (Offset p : sv)
    {
        dot += weights.x[p.first] * p.second;
    }
    return dot / wdiv + wbias;
}

