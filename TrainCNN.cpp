#include "sgdtk/sgdtk.h"

#include "n3rd/Word2VecModel.h"
#include <iostream>
#include <map>
#include <cassert>
#include "n3rd/OrderedEmbeddedDatasetReader.h"
#include "n3rd/NeuralNetModelFactory.h"
#include "n3rd/TanhLayer.h"
#include "n3rd/TemporalConvolutionalLayer.h"
#include "n3rd/TemporalConvolutionalLayerBlas.h"
#include "n3rd/FullyConnectedLayer.h"
#include "n3rd/FullyConnectedLayerBlas.h"
#include "n3rd/FullyConnectedLayerCuBlas.h"
#include "n3rd/AverageFoldingLayer.h"
#include "n3rd/KMaxPoolingLayer.h"
#include "n3rd/ReLULayer.h"
#include "n3rd/MaxOverTimePoolingLayer.h"
#include "n3rd/WeightHacks.h"
//#include "n3rd/TemporalConvolutionalLayerFFT.h"
#include "n3rd/TemporalConvolutionalLayerCuBlas.h"
#include "n3rd/MaxOverTimePoolingLayerCuda.h"
#include "n3rd/TanhLayerCuda.h"
#include "n3rd/NeuralNetModelCuda.h"
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

Learner* createTrainer(double lambda, double eta, int embeddingsAsChannels)//, cudnnHandle_t handle)
{
    // FIXME!!
    ModelFactory* nnmf;
    if (embeddingsAsChannels)
    {
        std::cout << "Use embeddings as input channels" << std::endl;
#if defined(RUN_ON_GPU)
        nnmf = new NeuralNetModelFactory<NeuralNetModelCuda>( {
                                                                      // Emit 8 feature maps use a kernel width of 7 -- embeddings are 300 deep (L1)
                                                                      new TemporalConvolutionalLayerCuBlas(100, 300, 7),
                                                                      new MaxOverTimePoolingLayerCuda(100), new TanhLayerCuda(),
                                                                      new FullyConnectedLayerCuBlas(1, 100), new TanhLayerCuda() });
#else // Run on CPU

        nnmf = new NeuralNetModelFactory<NeuralNetModel>( {
                new TemporalConvolutionalLayerBlas(100, 300, 7),
                new MaxOverTimePoolingLayer(100), new TanhLayer(),
                new FullyConnectedLayerBlas(1, 100), new TanhLayer() });
#endif

    }
    else
    {
        nnmf = new NeuralNetModelFactory<>({
                // Emit 8 feature maps use a kernel width of 7 -- embeddings are 300 deep (L1)
                new TemporalConvolutionalLayer(4, 1, 7, 300), new TanhLayer(),
                // Cut the embedding dim down to 75 by averaging adjacent embedding rows
                new AverageFoldingLayer(4, 300, 4),
                // Do K dynamic pooling grabbing the 10 highest values from each signal
                new KMaxPoolingLayer(3, 4, 75), new TanhLayer(),
                new FullyConnectedLayer(100, 900), new TanhLayer(),
                new FullyConnectedLayer(1, 100), new TanhLayer() });


    }

    SGDLearner *learner = new SGDLearner(new LogLoss(), lambda, eta, nnmf);

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

        String trainFile = params("train");
        String evalFile = params("eval");
        String modelFile = params("embed");

        const double Lambda = 1e-4;
        const double Eta = 0.1;

        int epochs = valueOf<int>(params("epochs", "5"));

        int embeddingsAsChannels = valueOf<int>(params("embed2channels", "0"));
        std::cout << "Model file: " << modelFile << std::endl;

        Learner* learner = createTrainer(Lambda, Eta, embeddingsAsChannels);

        OrderedEmbeddedDatasetReader reader(modelFile, (7 - 1) / 2);

        long l0 = sgdtk::currentTimeSeconds();

        auto trainingSet = reader.load(trainFile);


        double elapsed = sgdtk::currentTimeSeconds() - l0;
        std::cout << "Training data ("  << trainingSet.size() << " examples) + loaded in " <<  elapsed <<  "s" << std::endl;

        trainingSet.resize(30000);

        std::vector<FeatureVector*> evalSet;

        if (!evalFile.empty())
        {

            evalSet = reader.load(evalFile);
            evalSet.resize(3590);
#ifdef __DEBUG
            // Shuffle deterministically
            WeightHacks::shuffle(evalSet);
#else
            std::random_shuffle(evalSet.begin(), evalSet.end());
#endif
        }



        Model* model = learner->create(nullptr);
#ifdef __DEBUG
        // Shuffle and fix the weights deterministically
        WeightHacks::hack((NeuralNetModel*) model);
        WeightHacks::shuffle(trainingSet);
#else
        std::random_shuffle(trainingSet.begin(), trainingSet.end());
#endif
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

            //learner->eval(model, trainingSet, metrics);
            //showMetrics(metrics, "Training Set Eval Metrics");
            //metrics.clear();

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