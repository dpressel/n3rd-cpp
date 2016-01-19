#ifndef __N3RD_SVM_MNISTREADER_H__
#define __N3RD_SVM_MNISTREADER_H__

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

    class MNISTReader : public sgdtk::FeatureProvider
    {

        std::ifstream *imageFile;
        std::ifstream *labelFile;

        int numImages;
        int numRows;
        int numCols;
        int current;
        int zeroPadding;

        bool bigEndian;

        int readInt(std::istream& stream)
        {
            int x;
            stream.read((char*)&x, 4);
            return bigEndian ? x: sgdtk::byteSwap(x);
        }

        void open(std::string imageFileName, std::string labelFileName);

        double readLabel();

        void readImage(sgdtk::Tensor& zp);

    public:

        int getLargestVectorSeen()
        {
            return numRows * numCols;
        }

        MNISTReader(int padding = 0) :
                zeroPadding(padding)
        {
            bigEndian = sgdtk::isBigEndianSystem();
        }

        virtual ~MNISTReader()
        { }

        /**
        * Slurp the entire file into memory.  This is not the recommended way to read large datasets, use
        * {@link #next()} to stream features from the file one by one.
        *
        * @param file An SVM light type file
        * @return One feature vector per line in the file
        * @throws IOException
        */
        std::vector<sgdtk::FeatureVector *> load(std::string imageFile, std::string labelFile);

        /**
         * Get the next feature vector in the file
         *
         * @return The next feature vector, or null, if we are out of lines
         * @throws IOException
         */
        sgdtk::FeatureVector *next();

        /**
         * Close the currently loaded file
         * @throws Exception
         */
        void close();

    };
}

#endif