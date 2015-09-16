#ifndef __SGDTK_LINEAR_MODEL_H__
#define __SGDTK_LINEAR_MODEL_H__

#include "sgdtk/Model.h"
#include "sgdtk/Types.h"
#include "sgdtk/VectorN.h"
#include "sgdtk/DenseVectorN.h"

namespace sgdtk
{

/**
 * Linear model for classification
 *
 * @author dpressel
 */
    class LinearModel : public Model
    {
    protected:
        DenseVectorN weights;

        double wdiv;

        double wbias;

        LinearModel(const LinearModel &lm)
        {
            weights = lm.weights;
            wdiv = lm.wdiv;
            wbias = lm.wbias;
        }

    public:

        LinearModel(long wl, double wd = 1.0, double wb = 0.0)
                : wdiv(wd), wbias(wb)
        {
            weights.resize(wl);
        }

        LinearModel()
        { }

        virtual ~LinearModel()
        {
        }

        /**
         * Load the model from the stream
         * @param file Model file to load from
         * @throws Exception
         */
        void load(String file);

        /**
         * Save the model to a stream
         * @param file model to save to
         * @throws Exception
         */
        void save(String file);

        void add(const FeatureVector *fv, double disp)
        {
            const auto &sv = fv->getX().getNonZeroOffsets();
            for (auto p : sv)
            {
                weights.x[p.first] += p.second * disp;
            }
        }

        double predict(const FeatureVector *fv) const;

        Model *prototype() const
        {
            return new LinearModel(*this);
        }

        double getWdiv() const
        {
            return wdiv;
        }

        void setWdiv(double wdiv)
        {
            this->wdiv = wdiv;
        }

        double getWbias() const
        {
            return wbias;
        }

        void setWbias(double wbias)
        {
            this->wbias = wbias;
        }

        double mag() const;

        void scaleInplace(double scalar);

        virtual void scaleWeights(double eta, double lambda)
        {
            wdiv /= (1 - eta * lambda);

            if (wdiv > 1e5)
            {
                auto sf = 1.0 / wdiv;
                weights.scale(sf);

                wdiv = 1.;


            }
        }

        void updateWeights(const VectorN& vectorN, double eta, double lambda, double dLoss, double y)
        {
            scaleWeights(eta, lambda);

            for (auto offset : vectorN.getNonZeroOffsets())
            {
                auto grad = dLoss * offset.second;
                auto thisEta = perWeightUpdate(offset.first, grad, eta);
                weights.x[offset.first] += offset.second * -thisEta * dLoss * wdiv;
            }
            // This is referenced on Leon Bottou's SGD page
            auto etab = eta * 0.01;
            wbias += -etab * dLoss;
        }

        virtual double perWeightUpdate(int index, double grad, double eta)
        {
            return eta;
        }



    };
}

#endif