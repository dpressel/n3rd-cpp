#include "sgdtk/sgdtk.h"

#include <iostream>
#include <map>
#include <cassert>
#include <n3rd/LogSoftMaxLayer.h>
#include "n3rd/NeuralNetModelFactory.h"
#include "n3rd/TanhLayer.h"
#include "n3rd/SpatialConvolutionalLayer.h"
#include "n3rd/SpatialConvolutionalLayerBlas.h"
#include "n3rd/FullyConnectedLayer.h"
#include "n3rd/FullyConnectedLayerBlas.h"
#include "n3rd/FullyConnectedLayerCuBlas.h"
#include "n3rd/MaxPoolingLayer.h"
#include "n3rd/ReLULayer.h"
#include "n3rd/TanhLayerCuda.h"
#include "n3rd/NeuralNetModelCuda.h"
#include "n3rd/ClassNLLLoss.h"
#include "n3rd/MNISTReader.h"
using namespace sgdtk;
using namespace n3rd;

const int ZERO_PAD = 4;

void showMetrics(Metrics metrics, String pre)
{
    std::cout << "========================================================" << std::endl;
    std::cout << pre << std::endl;
    std::cout << "========================================================" << std::endl;

    std::cout << "\tLoss = " << metrics.getLoss() << std::endl;
    std::cout << "\tCost = " << metrics.getCost() << std::endl;
    std::cout << "\tError = " << 100 * metrics.getError() << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
}

Learner* createTrainer(double lambda, double eta)//, cudnnHandle_t handle)
{
    NeuralNetModelFactory<>* factory = new NeuralNetModelFactory<>;

    factory->addLayer(new SpatialConvolutionalLayerBlas(6, 5, 5, {1,32,32}));
    factory->addLayer(new MaxPoolingLayer(2, 2, {6, 28, 28}));
    factory->addLayer(new TanhLayer());
    factory->addLayer(new SpatialConvolutionalLayerBlas(16, 5, 5, {6,14,14}));
    factory->addLayer(new MaxPoolingLayer(2, 2, {16, 10, 10}));
    factory->addLayer(new TanhLayer());
    factory->addLayer(new SpatialConvolutionalLayerBlas(128, 5, 5, {16,5,5}));
    factory->addLayer(new TanhLayer());
    factory->addLayer(new FullyConnectedLayerBlas(84, 128));
    factory->addLayer(new TanhLayer());
    factory->addLayer(new FullyConnectedLayerBlas(10, 84));
    factory->addLayer(new LogSoftMaxLayer());

    Learner* learner = new SGDLearner(new ClassNLLLoss(), lambda, eta, factory, new FixedLearningRateSchedule());
    return learner;
}

int main(int argc, char** argv)
{
    //cudnnHandle_t handle = NULL;

    try
    {
        initCuBlas();
        //cudnnCreate(&handle);
        Params params(argc, argv);

        String trainImageFile = params("trainx");
        String trainLabelFile = params("trainy");
        String evalImageFile = params("evalx");
        String evalLabelFile = params("evaly");

        const double Lambda = 1e-4;
        const double Eta = 0.01;

        int epochs = valueOf<int>(params("epochs", "5"));



        MNISTReader reader(ZERO_PAD);

        long l0 = currentTimeSeconds();

        auto trainingSet = reader.load(trainImageFile, trainLabelFile);
        ///auto trainingSet = reader.load(trainFile);

        double elapsed = currentTimeSeconds() - l0;
        std::cout << "Training data ("  << trainingSet.size() << " examples) + loaded in " <<  elapsed <<  "s" << std::endl;


        std::vector<FeatureVector*> evalSet;
        if (!evalLabelFile.empty())
        {
            evalSet = reader.load(evalImageFile, evalLabelFile);
#ifdef __DEBUG
            // Shuffle deterministically
            WeightHacks::shuffle(evalSet);
#else
            std::random_shuffle(evalSet.begin(), evalSet.end());
#endif
        }

        Learner* learner = createTrainer(Lambda, Eta);
        auto model = learner->create(nullptr);
        double totalTrainingElapsed = 0.;


        for (int i = 0; i < epochs; ++i)
        {
            std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
            std::cout << "EPOCH: " << (i + 1) << std::endl;


            Metrics metrics;
            double t0 = currentTimeSeconds();


            auto subset = trainingSet;
            std::random_shuffle(subset.begin(), subset.end());
            subset.resize(5000);
            learner->trainEpoch(model, subset);
            double elapsedThisEpoch = sgdtk::currentTimeSeconds() - t0;
            std::cout << "Epoch training time " << elapsedThisEpoch << "s" << std::endl;
            totalTrainingElapsed += elapsedThisEpoch;

            learner->eval(model, subset, metrics);
            showMetrics(metrics, "Training Set Eval Metrics");
            metrics.clear();

            if (!evalSet.empty())
            {
                //evaluate(evalSet, model);
                //std::random_shuffle(evalSet.begin(), evalSet.end());
                learner->eval(model, evalSet, metrics);
                showMetrics(metrics, "Test Set Eval Metrics");
                metrics.clear();
            }

        }

        std::cout << "Total training time " << totalTrainingElapsed << "s" << std::endl;


    }
    catch (Exception& ex)
    {

        std::cout << "Exception: " << ex.getMessage() << std::endl;
    }
    //if (handle) cudnnDestroy(handle);
    doneCuBlas();
    return 0;
}