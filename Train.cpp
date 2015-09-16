#include "sgdtk/sgdtk.h"

#include <iostream>
#include <map>
using namespace sgdtk;


void showMetrics(const Metrics& metrics, String pre)
{

    std::cout << "========================================================" << std::endl;
    std::cout << pre << std::endl;
    std::cout << "========================================================" << std::endl;

    std::cout << "\tLoss = " << metrics.getLoss() << std::endl;
    std::cout << "\tCost = " << metrics.getCost() << std::endl;
    std::cout << "\tError = " << 100. * metrics.getError() << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

}

int main(int argc, char** argv)
{
    try
    {
        Params params(argc, argv);

        String trainFile = params("train");
        std::cout << "Train: " << trainFile << std::endl;

        double l0 = currentTimeSeconds();
        SVMLightFileFeatureProvider reader;
        
        std::vector<FeatureVector*> trainingSet = reader.load(trainFile);
        double elapsed = currentTimeSeconds() - l0;

        std::vector<FeatureVector*> evalSet;
        String evalFile = params("eval");
        if (evalFile != "")
        {
            reader.open(evalFile);
            evalSet = reader.load(evalFile);
        }



        std::cout << "Training data loaded in " << elapsed << "s" << std::endl;
        Loss* lossFunction;
        String loss = params("loss");
        if (loss == "log")
        {
            std::cout << "Using log loss" << std::endl;
            lossFunction = new LogLoss();
        }
        else if (loss == "sqh")
        {
            std::cout << "Using squared hinge loss" << std::endl;
            lossFunction = new SquaredHingeLoss();
        }
        else if (loss == "sq")
        {
            std::cout << "Using squared loss" << std::endl;
            lossFunction = new SquaredLoss();
        }
        else
        {
            std::cout << "Using hinge loss" << std::endl;
            lossFunction = new HingeLoss();
        }

        double eta0 = valueOf<double>(params("eta0", "-1"));
        if (eta0 != -1)
        {
            std::cout << "Using eta0=" << eta0 << std::endl;
        }
        double lambda = valueOf<double>(params("lambda", "1e-5"));
        std::cout << "Using lambda=" << lambda << std::endl;

        String type = params("optim", "sgd");

        LinearModelFactory mf(type);

        SGDLearner learner(lossFunction, lambda, eta0, &mf);

        long vSz = (long)reader.getLargestVectorSeen();
        std::cout << "Creating model with vector of size " << vSz << std::endl;
        Model* model = learner.create((void*)vSz);
        
        int epochs = valueOf<int>(params("epochs", "5"));
        double totalTrainingElapsed = 0.;
        for (int i = 0; i < epochs; ++i)
        {
            std::cout << 
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" <<
            std::endl;

            std::cout << "EPOCH: " << (i + 1) << std::endl;
            Metrics metrics;

            double t0 = currentTimeSeconds();

            learner.trainEpoch(model, trainingSet);
            double elapsedThisEpoch = currentTimeSeconds() - t0;
            std::cout << "Epoch training time " << elapsedThisEpoch << "s" << std::endl;
            totalTrainingElapsed += elapsedThisEpoch;

            learner.eval(model, trainingSet, metrics);
            showMetrics(metrics, "Training Set Eval Metrics");
            metrics.clear();

            if (evalSet.size())
            {
                learner.eval(model, evalSet, metrics);
                showMetrics(metrics, "Test Set Eval Metrics");
            }
        }

        std::cout << "Total training time " << totalTrainingElapsed << "s" << std::endl;
        String modelFile = params("model");
        if (modelFile != "")
        {
            model->save(modelFile);
        }

        delete model;
        delete lossFunction;
        for (size_t i = 0, sz = trainingSet.size(); i < sz; ++i)
        {
            delete trainingSet[i];
        }

        for (size_t i = 0, sz = evalSet.size(); i < sz; ++i)
        {
            delete evalSet[i];
        }
    }
    catch (Exception& ex)
    {
        std::cout << "Exception: " << ex.getMessage() << std::endl;
    }

    return 0;
}