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
        /// THIS SHOULD BE CUDA BACKED!
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
        virtual void load(String file);

        /**
         * Save the model to a stream
         * @param file model to save to
         * @throws Exception
         */
        virtual void save(String file);

        /// Extract to a (D)AXPY!
        virtual void add(const FeatureVector *fv, double disp)
        {
            const auto &sv = fv->getX()->getNonZeroOffsets();

            ////weights.axpy(fv->getX(), disp);
            for (auto p : sv)
            {
                weights.x[p.first] += p.second * disp;
            }
        }

        virtual double predict(const FeatureVector *fv);

        virtual std::vector<double> score(const sgdtk::FeatureVector *fv);
        virtual Model *prototype() const
        {
            return new LinearModel(*this);
        }

        virtual double getWdiv() const
        {
            return wdiv;
        }

        virtual void setWdiv(double wdiv)
        {
            this->wdiv = wdiv;
        }

        virtual double getWbias() const
        {
            return wbias;
        }

        virtual void setWbias(double wbias)
        {
            this->wbias = wbias;
        }

        virtual double mag() const;

        virtual void scaleInplace(double scalar);

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

        virtual void updateWeights(const VectorN* vectorN, double eta, double lambda, double dLoss, double y)
        {
            scaleWeights(eta, lambda);

            /// ANOTHER (D)AXPY, except, a is determined on each

            for (auto offset : vectorN->getNonZeroOffsets())
            {
                auto grad = dLoss * offset.second;
                auto thisEta = perWeightUpdate(offset.first, grad, eta);
                weights.x[offset.first] += offset.second * -thisEta * dLoss * wdiv;
            }
            // This is referenced on Leon Bottou's SGD page
            auto etab = eta * 0.01;
            wbias += -etab * dLoss;
        }

        // Not really adding too much value, just makes it possible for adagrad to happen without a separate updateWeights
        virtual double perWeightUpdate(int index, double grad, double eta)
        {
            return eta;
        }



    };
}

#endif