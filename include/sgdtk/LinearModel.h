#ifndef __SGDTK_LINEAR_MODEL_H__
#define __SGDTK_LINEAR_MODEL_H__

#include "sgdtk/Model.h"
#include "sgdtk/Types.h"

namespace sgdtk
{

/**
 * Linear model for classification
 *
 * @author dpressel
 */
class LinearModel : public Model
{
    std::vector<double> weights;
    
    double wdiv;
 
    double wbias;

    LinearModel(const LinearModel& lm)
    {
        weights = lm.weights;
        wdiv = lm.wdiv;
        wbias = lm.wbias;
    }
public:

    LinearModel(int wl, double wd, double wb)
        : wdiv(wd), wbias(wb) 
    {
        weights.resize(wl, 0);
    }
    LinearModel() {}
    ~LinearModel() {}

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

    double predict(const FeatureVector* fv) const;
    
    Model* prototype() const;

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

    void addInplace(int i, double update);

};
}

#endif