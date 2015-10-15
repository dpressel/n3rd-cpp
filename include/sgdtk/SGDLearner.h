#ifndef __SGDTK_SGD_LEARNER_H__
#define __SGDTK_SGD_LEARNER_H__

#include "sgdtk/Learner.h"
#include "sgdtk/LinearModel.h"
#include "sgdtk/FeatureVector.h"
#include "sgdtk/Loss.h"
#include "sgdtk/LearningRateSchedule.h"
#include "sgdtk/ModelFactory.h"
#include "sgdtk/RobbinsMonroUpdateSchedule.h"
#include "sgdtk/LinearModelFactory.h"

namespace sgdtk
{
    const double ETA_FACTOR = 2.;
    const double LOW_ETA_0 = 1.;

    class SGDLearner : public Learner
    {

        const Loss *lossFunction;
        const ModelFactory* modelFactory;
        LearningRateSchedule* learningRateSchedule;

        double lambda;
        double eta0;
        double numSeenTotal;

        double evalEta(Model *model,
                       const std::vector<FeatureVector *> &sample,
                       double eta);

        //void trainOneWithEta(LinearModel *lm, const FeatureVector *fv, double eta);

    public:
        SGDLearner(const Loss *loss, double l = 1e-5, double kEta = -1., const ModelFactory* factory = nullptr, LearningRateSchedule* schedule = nullptr) :
                lossFunction(loss), lambda(l), eta0(kEta), numSeenTotal(0.)
        {
            modelFactory = (factory == NULL) ? (new LinearModelFactory()): (factory);
            learningRateSchedule = (schedule == nullptr) ? (new RobbinsMonroUpdateSchedule()): (schedule);
        }

        ~SGDLearner()
        { }

        void trainOne(Model *lm, const FeatureVector *fv);

        double evalOne(Model *model, const FeatureVector *fv, Metrics &metrics);

        void preprocess(Model *model, const std::vector<FeatureVector *> &sample);

        Model *create(void* p);

        Model *trainEpoch(Model *model,
                          const std::vector<FeatureVector *> &trainingExamples);

        void eval(Model *model,
                  const std::vector<FeatureVector *> &testingExamples,
                  Metrics &metrics);

    };
}

#endif