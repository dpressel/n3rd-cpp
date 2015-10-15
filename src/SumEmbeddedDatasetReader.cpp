#include "n3rd/SumEmbeddedDatasetReader.h"
#include <iostream>

using namespace n3rd;

void SumEmbeddedDatasetReader::open(std::string file)
{
    ifile = new std::ifstream;
    ifile->open(file.c_str());
    ifile->sync_with_stdio(false);
    if (!ifile->is_open())
    {
        throw sgdtk::Exception("File " + file + " was not opened!");
    }
    largestVectorSeen = word2vecModel->embedSz;
}

void SumEmbeddedDatasetReader::close()
{
    if (ifile)
    {
        delete ifile;
        ifile = NULL;
    }

}

std::vector<sgdtk::FeatureVector *> SumEmbeddedDatasetReader::load(std::string file)
{
    std::vector<sgdtk::FeatureVector *> fvs;
    open(file);
    sgdtk::FeatureVector *fv;
    //int readSoFar = 0;
    while ((fv = next()) != NULL)
    {

        fvs.push_back(fv);
        //if (++readSoFar > 100)
        //  break;

    }

    //close();
    return fvs;
}


sgdtk::FeatureVector *SumEmbeddedDatasetReader::next()
{
    if (ifile->eof())
    {
        return nullptr;
    }

    int lastIdxTotal = (int) MAX_FEATURES - 1;

    char buf[MAX_FEATURES];

    ifile->getline(buf, MAX_FEATURES);

    std::string line(buf);
    sgdtk::trim(line);
    
    std::transform(line.begin(), line.end(), line.begin(), ::tolower);
    sgdtk::StringArray ary = sgdtk::split(line, '\t');
    if (ary.size() != 2)
    {
        std::cout << "Bad line: " << line << std::endl;
        return next();
    }
    double y = sgdtk::valueOf<double>(ary[0]);


    sgdtk::DenseVectorN * x = new sgdtk::DenseVectorN(word2vecModel->embedSz);
    sgdtk::StringArray words = sgdtk::split(ary[1], ' ');

    sgdtk::FeatureVector *fv = new sgdtk::FeatureVector(y, x);

    sgdtk::Tensor& xArray = x->getX();

    // For each token in the sentence, we need to sum up the embeddings.  So remember that the feature vector
    // is going to be embedSz
    for (auto word: words)
    {

        auto vec = word2vecModel->getVec(word);
        assert(vec.size() <= word2vecModel->embedSz);
        for (int j = 0, vSz = vec.size(); j < vSz; ++j)
        {
            auto vecj = vec[j];
            xArray[j] += vecj;
        }
    }

    return fv;

}