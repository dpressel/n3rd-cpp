#ifndef __SGDTK_ADAGRAD_LINEAR_MODEL_H__
#define __SGDTK_ADAGRAD_LINEAR_MODEL_H__

#include "sgdtk/Model.h"
#include "sgdtk/Types.h"
#include "sgdtk/VectorN.h"
#include "sgdtk/DenseVectorN.h"
#include <cmath>

namespace sgdtk
{

/**
 * Linear model for classification
 *
 * @author dpressel
 */
    class AdagradLinearModel : public LinearModel
    {
        DenseVectorN gg;

        double sumEta;

        const double ALPHA = 1.;
        const double BETA = 1.;
        const double EPS = 1e-8;

        AdagradLinearModel(const AdagradLinearModel &alm)
        {
            weights = alm.weights;
            wdiv = alm.wdiv;
            wbias = alm.wbias;
            gg = alm.gg;
        }

    public:

        AdagradLinearModel(long wl, double wd = 1.0, double wb = 0.0)
                : LinearModel(wl, wd, wb)
        {
            gg.resize(wl);
        }

        AdagradLinearModel()
        { }

        virtual ~AdagradLinearModel()
        {
        }

        Model *prototype() const
        {
            return new AdagradLinearModel(*this);
        }

        virtual void scaleWeights(double eta, double lambda)
        {
            if (sumEta != 0)
            {
                eta = sumEta / weights.length();
            }
            sumEta = 0.;
            LinearModel::scaleWeights(eta, lambda);
        }

        /// UPDATE!!!!
        virtual double perWeightUpdate(int index, double grad, double eta)
        {

            gg.x[index] = ALPHA * gg.x[index] + BETA * grad * grad;
            double etaThis = eta / std::sqrt(gg.x[index] + EPS);
            sumEta += etaThis;
            return etaThis;
        }



    };
}

#endif