#include "sgdtk/SGDLearner.h"
#include <iostream>
using namespace sgdtk;

Model* SGDLearner::create(int wlength)
{
    numSeenTotal = 0;
    LinearModel* lm = new LinearModel(wlength, 1., 0.);
    return lm;
}

void SGDLearner::trainOne(Model* model, const FeatureVector* fv)
{
    LinearModel* lm = (LinearModel*)model;
    double eta = eta0 / (1 + lambda * eta0 * numSeenTotal);

    trainOneWithEta(lm, fv, eta);

    ++numSeenTotal;
}

void SGDLearner::trainOneWithEta(LinearModel* lm, const FeatureVector* fv, double eta)
{
    double y = fv->getY();
    double fx = lm->predict(fv);
    double wdiv = lm->getWdiv();
    wdiv /= (1 - eta * lambda);
    if (wdiv > 1e5)
    {
        const double sf = 1.0 / wdiv;
        lm->scaleInplace(sf);
        wdiv = 1.;
    }
    lm->setWdiv(wdiv);

    double d = lossFunction->dLoss(fx, y);
    double disp = -eta * d * wdiv;

    //const Offsets& sv = fv->getNonZeroOffsets();

    lm->add(fv, disp);
    //for (Offsets::const_iterator p = sv.begin(); p != sv.end(); ++p)
    //{
    //    lm->addInplace(p->first, p->second * disp);
    //}
  
    double etab = eta * 0.01;
    double wbias = lm->getWbias();

    wbias += -etab * d;
    lm->setWbias(wbias);
 
}
void SGDLearner::preprocess(Model* model, const std::vector<FeatureVector*>& sample)
{
    double lowEta = LOW_ETA_0;
    double lowCost = evalEta(model, sample, lowEta);
    double highEta = lowEta * ETA_FACTOR;
    double highCost = evalEta(model, sample, highEta);
    if (lowCost < highCost)
    {
        while (lowCost < highCost)
        {
            highEta = lowEta;
            highCost = lowCost;
            lowEta = highEta / ETA_FACTOR;
            lowCost = evalEta(model, sample, lowEta);
        }
    }
    else if (highCost < lowCost)
    {
        while (highCost < lowCost)
        {
            lowEta = highEta;
            lowCost = highCost;
            highEta = lowEta * ETA_FACTOR;
            highCost = evalEta(model, sample, highEta);
        }
    }
    eta0 = lowEta;
    std::cout << "picked: " << eta0 << " in eta0" << std::endl;
}


double SGDLearner::evalEta(Model* model, const std::vector<FeatureVector*>& sample, double eta)
{
    LinearModel* clone = (LinearModel*)model->prototype();

    for (size_t i = 0, sz = sample.size(); i < sz; ++i)
    {
        trainOneWithEta(clone, sample[i], eta);
    }
    
    Metrics metrics;
    eval(clone, sample, metrics);
    return metrics.getCost();
}

Model* SGDLearner::trainEpoch(Model* model,
        const std::vector<FeatureVector*>& trainingExamples)
{
    if (eta0 <= 0)
    {
        
        // get samples from trainingExamples
        size_t nSamples = std::min<size_t>(1000, trainingExamples.size());
        
        std::vector<FeatureVector*> samples(nSamples);
        for (size_t i = 0; i < nSamples; ++i)
        {
            samples[i] = trainingExamples[i];
        }
        preprocess(model, samples);
    }
    
    
    for (size_t i = 0, sz = trainingExamples.size(); i < sz; ++i)
    {
        //double eta = eta0 / (1 + lambda * eta0 * numSeenTotal);
        trainOne(model, trainingExamples[i]);
        //++numSeenTotal;
    }

    LinearModel* lm = (LinearModel*)model;
    
    std::cout << "wNorm="  << lm->mag() << std::endl;
    return model;

}

double SGDLearner::evalOne(Model* model, const FeatureVector* fv, Metrics& metrics)
{
    double y = fv->getY();
    double fx = model->predict(fv);
    double loss = lossFunction->loss(fx, y);
    double error = (fx * y <= 0) ? 1 : 0;
    metrics.add(loss, error);
    return fx;
}

void SGDLearner::eval(Model* model,
        const std::vector<FeatureVector*>& testingExamples,
        Metrics& metrics)
{
    int seen = testingExamples.size();

    for (int i = 0; i < seen; ++i)
    {
        const FeatureVector* fv = testingExamples[i];
        evalOne(model, fv, metrics);

    }

    LinearModel* lm = (LinearModel*)model;
    double normW = lm->mag();
    double cost = metrics.getLoss() + 0.5 * lambda * normW;
    metrics.setCost(cost);
}