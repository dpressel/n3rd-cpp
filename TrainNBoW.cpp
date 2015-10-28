#include "sgdtk/sgdtk.h"

#include "n3rd/Word2VecModel.h"
#include <iostream>
#include <map>
#include <cassert>
#include <n3rd/SigmoidLayer.h>

#include "n3rd/NeuralNetModelFactory.h"
#include "n3rd/TanhLayer.h"
#include "n3rd/TemporalConvolutionalLayer.h"
#include "n3rd/FullyConnectedLayer.h"
#include "n3rd/AverageFoldingLayer.h"
#include "n3rd/KMaxPoolingLayer.h"
#include "n3rd/SumEmbeddedDatasetReader.h"


using namespace sgdtk;
using namespace n3rd;



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

        String modelFile = params("embed");
        String trainFile = params("train");
        String evalFile = params("eval");
        const double Lambda = 1e-4;
        const double Eta = 0.1;

        int epochs = valueOf<int>(params("epochs", "5"));

        std::cout << "Model file: " << modelFile << std::endl;

        SumEmbeddedDatasetReader reader(modelFile);

        long l0 = sgdtk::currentTimeSeconds();

        auto trainingSet = reader.load(trainFile);

#ifdef __DEBUG
        // Shuffle deterministically
        WeightHacks::shuffle(trainingSet);
#else
        std::random_shuffle(trainingSet.begin(), trainingSet.end());
#endif
        double elapsed = sgdtk::currentTimeSeconds() - l0;
        std::cout << "Training data ("  << trainingSet.size() << " examples) + loaded in " <<  elapsed <<  "s" << std::endl;

        trainingSet.resize(30000);

        std::vector<FeatureVector*> evalSet;

        if (!evalFile.empty())
        {

            evalSet = reader.load(evalFile);

#ifdef __DEBUG
            // Shuffle deterministically
            WeightHacks::shuffle(evalSet);
#else
            std::random_shuffle(evalSet.begin(), evalSet.end());
#endif
            evalSet.resize(3590);
        }

        int vSz = reader.getLargestVectorSeen();
        Learner* learner = nullptr;
        if (params("type", "nbow") == "nbow")
        {
            learner =
                    new SGDLearner(new LogLoss, Lambda, Eta, new NeuralNetModelFactory({
                            new FullyConnectedLayer(100, vSz), new TanhLayer(), new FullyConnectedLayer(1, 100), new TanhLayer()
                    }));
        }

        else
        {
            learner =
                    new SGDLearner(new LogLoss(), Lambda, Eta, new LinearModelFactory());
        }

        Model* model = learner->create(&vSz);
        double totalTrainingElapsed = 0.;

        for (int i = 0; i < epochs; ++i)
        {
            std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            std::cout << "EPOCH: " << (i + 1) << std::endl;

            Metrics metrics;
            double t0 = sgdtk::currentTimeSeconds();

            learner->trainEpoch(model, trainingSet);//trainingSet.subList(0, 1000));
            double elapsedThisEpoch = sgdtk::currentTimeSeconds() - t0;
            std::cout << "Epoch training time " << elapsedThisEpoch << "s" << std::endl;
            totalTrainingElapsed += elapsedThisEpoch;

            learner->eval(model, trainingSet, metrics);
            showMetrics(metrics, "Training Set Eval Metrics");
            metrics.clear();

            if (!evalSet.empty())
            {
                //evaluate(evalSet, model);
                learner->eval(model, evalSet, metrics);
                showMetrics(metrics, "Test Set Eval Metrics");
                metrics.clear();
            }

        }

        std::cout << "Total training time " << totalTrainingElapsed << "s" << std::endl;

        for (auto x : trainingSet)
        {
            delete x;
        }
        for (auto x : evalSet)
        {
            delete x;
        }
        delete learner;
        delete model;
    }


    catch (Exception& ex)
    {
        std::cout << "Exception: " << ex.getMessage() << std::endl;
    }

    return 0;
}