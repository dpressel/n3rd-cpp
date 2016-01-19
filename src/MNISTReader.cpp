#include "n3rd/MNISTReader.h"
#include <iostream>

using namespace n3rd;

std::vector<sgdtk::FeatureVector *> MNISTReader::load(std::string imageFile, std::string labelFile)
{
    open(imageFile, labelFile);
    std::vector<sgdtk::FeatureVector*> fvs;

    sgdtk::FeatureVector* fv = NULL;

    while ((fv = next()) != NULL)
    {
        fvs.push_back(fv);
    }
    std::cout << "Read " << numImages << " images" << std::endl;
    close();
    return fvs;
}

sgdtk::FeatureVector *MNISTReader::next()
{
    if (current == numImages)
    {
        return NULL;
    }
    double label = readLabel();

    sgdtk::DenseVectorN* dv = new sgdtk::DenseVectorN(0);
    sgdtk::FeatureVector* fv = new sgdtk::FeatureVector(label, dv);
    readImage(dv->getX());
    ++current;
    return fv;
}

void MNISTReader::close()
{
    current = numCols = numRows = 0;
    imageFile->close();
    labelFile->close();
    delete imageFile;
    delete labelFile;
}


void MNISTReader::open(std::string imageFileName, std::string labelFileName)
{
    current = 0;
    labelFile = new std::ifstream(labelFileName, std::ios::binary);
    if (!labelFile->is_open())
    {
        throw sgdtk::Exception("File " + labelFileName + " was not opened!");
    }
    int x = readInt(*labelFile);
    if (x != 2049)
    {
        throw sgdtk::Exception("Bad magic");
    }
    int numLabels = readInt(*labelFile);


    imageFile = new std::ifstream(imageFileName, std::ios::binary);
    if (!imageFile->is_open())
    {
        throw sgdtk::Exception("File " + imageFileName + " was not opened!");
    }

    x = readInt(*imageFile);
    if (x != 2051)
    {
        throw sgdtk::Exception("Bad magic");
    }
    numImages = readInt(*imageFile);
    if (numLabels != numImages)
    {
        throw sgdtk::Exception("Label/image mismatch!");
    }
    numRows = readInt(*imageFile);
    numCols = readInt(*imageFile);


}

double MNISTReader::readLabel()
{
    unsigned char b;
    labelFile->read((char*)&b, 1);
    int ati = ((int)b & 0xFF);
    return ati + 1.0;
}
void MNISTReader::readImage(sgdtk::Tensor& zp)
{
    int numBytes = getLargestVectorSeen();
    std::vector<unsigned char> buffer(numBytes);
    imageFile->read((char*)&buffer[0], numBytes);
    sgdtk::Tensor tensor({1, numRows, numCols});
    for (int i = 0; i < numBytes; ++i)
    {
        int ati = ((int)buffer[i]) & 0xFF;
        tensor[i] = ati / 255.0;
    }
    sgdtk::embed(tensor, 0, zeroPadding, zeroPadding, zp);
}