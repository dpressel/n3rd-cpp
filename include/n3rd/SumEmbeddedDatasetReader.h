#ifndef __N3RD_SVM_SUMEMBEDDEDDATASETREADER_H__
#define __N3RD_SVM_SUMEMBEDDEDDATASETREADER_H__

#include <sgdtk/Exception.h>
#include <sgdtk/Types.h>
#include <sgdtk/FeatureVector.h>
#include <sgdtk/FeatureProvider.h>
#include "n3rd/Word2VecModel.h"
#include <sgdtk/SVMLightFileFeatureProvider.h>
#include <fstream>
#include <memory>
#include <sgdtk/DenseVectorN.h>
#include <cassert>
#include <iostream>
namespace n3rd
{

    class SumEmbeddedDatasetReader : public sgdtk::FeatureProvider
    {
        enum { MAX_FEATURES = 4096 };
        std::ifstream *ifile;
        int largestVectorSeen;
        int paddingSzPerSide;
        int lineNumber = 0;
        Word2VecModel* word2vecModel;

    public:

        int getLargestVectorSeen()
        {
            return largestVectorSeen;
        }

        /**
         * Create a provider, and you need to cap it with a max number of features.
         * Anything beyond this feature vector length will be clipped out of the resultant FV.
         *
         * @param maxFeatures The feature vector width.
         */
        SumEmbeddedDatasetReader(std::string embeddings, int padding = 0) :
                paddingSzPerSide(padding), largestVectorSeen(0)
        {
            word2vecModel = Word2VecModel::loadWord2VecModel(embeddings);
            std::cout << "Done loading model" << std::endl;

        }

        virtual ~SumEmbeddedDatasetReader()
        { }

        /**
        * Slurp the entire file into memory.  This is not the recommended way to read large datasets, use
        * {@link #next()} to stream features from the file one by one.
        *
        * @param file An SVM light type file
        * @return One feature vector per line in the file
        * @throws IOException
        */
        std::vector<sgdtk::FeatureVector *> load(std::string file);

        /**
         * Get the next feature vector in the file
         *
         * @return The next feature vector, or null, if we are out of lines
         * @throws IOException
         */
        sgdtk::FeatureVector *next();

        /**
         * Open a file for reading.  All files are read only up to maxFeatures.
         * @param file An SVM light type file
         * @throws Exception
         */
        void open(std::string file);

        /**
         * Close the currently loaded file
         * @throws Exception
         */
        void close();


    };
}

#endif