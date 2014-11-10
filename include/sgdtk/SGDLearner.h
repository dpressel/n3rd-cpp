#ifndef __SGDTK_SGD_LEARNER_H__
#define __SGDTK_SGD_LEARNER_H__

#include "sgdtk/Learner.h"
#include "sgdtk/LinearModel.h"
#include "sgdtk/FeatureVector.h"
#include "sgdtk/Loss.h"
namespace sgdtk
{
const double ETA_FACTOR = 2.;
const double LOW_ETA_0 = 1.;

class SGDLearner : public Learner
{

	const Loss* lossFunction;
	double lambda;
	double eta0;
	double numSeenTotal;

	double evalEta(Model* model,
		const std::vector<FeatureVector*>& sample,
		double eta);

	void trainOneWithEta(LinearModel* lm, const FeatureVector* fv, double eta);
public:
	SGDLearner(const Loss* loss, double l = 1e-5, double kEta = -1.) :
		lossFunction(loss), lambda(l), eta0(kEta), numSeenTotal(0.)
	{
	}
	~SGDLearner() {}

	void trainOne(Model* lm, const FeatureVector* fv);
	double evalOne(Model* model, const FeatureVector* fv, Metrics& metrics);
	void preprocess(Model* model, const std::vector<FeatureVector*>& sample);

    Model* create(int wlength);

    Model* trainEpoch(Model* model,
    	const std::vector<FeatureVector*>& trainingExamples);

    void eval(Model* model,
    	const std::vector<FeatureVector*>& testingExamples,
    	Metrics& metrics);

};
}

#endif