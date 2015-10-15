#include "n3rd/OrderedEmbeddedDatasetReader.h"
#include <iostream>

using namespace n3rd;

void OrderedEmbeddedDatasetReader::open(std::string file)
{
    ifile = new std::ifstream;
    ifile->open(file.c_str());
    ifile->sync_with_stdio(false);
    if (!ifile->is_open())
    {
        throw sgdtk::Exception("File " + file + " was not opened!");
    }
}

void OrderedEmbeddedDatasetReader::close()
{
    if (ifile)
    {
        delete ifile;
        ifile = NULL;
    }

}

std::vector<sgdtk::FeatureVector *> OrderedEmbeddedDatasetReader::load(std::string file)
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


sgdtk::FeatureVector *OrderedEmbeddedDatasetReader::next()
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


    sgdtk::StringArray words = sgdtk::split(ary[1], ' ');

    sgdtk::StringArray culled;
    for (int i = 0, sz = words.size(); i < sz; ++i)
    {
        auto vec = word2vecModel->getVec(words[i]);
        if (!vec.empty())
        {
            culled.push_back(words[i]);
        }
    }
    words = culled;
    int sentenceSz = words.size();
    if (sentenceSz < 1)
    {
        return next();
    }
    int pitch = 2 * paddingSzPerSide + sentenceSz;
    sgdtk::DenseVectorN* x = new sgdtk::DenseVectorN((2*paddingSzPerSide + sentenceSz) * word2vecModel->embedSz);
    sgdtk::FeatureVector *fv = new sgdtk::FeatureVector(y, x);

    for (int i = 0, ibase = 0; i < sentenceSz; ++i, ibase += word2vecModel->embedSz)
    {

        auto vec = word2vecModel->getVec(words[i]);
        assert(vec.size() == word2vecModel->embedSz);
        for (int j = 0, vSz = vec.size(); j < vSz; ++j)
        {
            x->set(j * pitch + i + paddingSzPerSide, vec[j]);
        }
    }

    fv->getX()->organize();
    return fv;

}