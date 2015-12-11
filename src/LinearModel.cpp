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

double LinearModel::predict(const FeatureVector *fv)
{
    /*
     * THIS IS WHAT I WANT TO DO
    auto& vecN = fv->getX();
    auto dot = weights.dot(vecN);
    */
    const Offsets &sv = fv->getNonZeroOffsets();
    double dot = weights.sparseDot(sv);
    return dot / wdiv + wbias;
}

std::vector<double> LinearModel::score(const sgdtk::FeatureVector *fv)
{
    return {
            predict(fv)
    };
}