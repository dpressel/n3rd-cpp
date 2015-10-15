#include "sgdtk/SVMLightFileFeatureProvider.h"
#include <iostream>


using namespace sgdtk;

Dims SVMLightFileFeatureProvider::findDims(String file)
{
    std::ifstream ifile;
    ifile.open(file.c_str());
    ifile.sync_with_stdio(false);
    if (!ifile.is_open())
    {
        throw Exception("File " + file + " was not opened!");
    }
    String line;

    int lastIdxTotal = 0;
    int n = 0;
    while (std::getline(ifile, line, '\n'))
    {
        StringArray strings = split(line);
        String end = strings[strings.size() - 1];
        int lastIdx = valueOf<int>(split(end, ':')[0]);
        lastIdxTotal = std::max<int>(lastIdx, lastIdxTotal);
        ++n;
    }

    ifile.close();
    Dims dims(lastIdxTotal + 1, n);
    return dims;
}

void SVMLightFileFeatureProvider::open(String file)
{
    ifile = new std::ifstream;
    ifile->open(file.c_str());
    ifile->sync_with_stdio(false);
    if (!ifile->is_open())
    {
        throw Exception("File " + file + " was not opened!");
    }
}

void SVMLightFileFeatureProvider::close()
{
    if (ifile)
    {
        delete ifile;
        ifile = NULL;
    }
    std::cout << "done deleting" << std::endl;
}

std::vector<FeatureVector *> SVMLightFileFeatureProvider::load(String file)
{
    std::vector<FeatureVector *> fvs;
    open(file);
    FeatureVector *fv;
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


FeatureVector *SVMLightFileFeatureProvider::next()
{
    if (ifile->eof())
    {
        return nullptr;
    }
    double y;
    double value;
    int lastIdxTotal = (int) maxFeatures - 1;
    *ifile >> std::skipws >> y >> std::ws;
    FeatureVector *fv = new FeatureVector(y);

    for (; ;)
    {
        int c = ifile->get();
        if (!ifile->good() || (c == '\n' || c == '\r'))
        {
            break;
        }
        if (::isspace(c))
            continue;
        int idx;
        ifile->unget();
        *ifile >> std::skipws >> idx >> std::ws;
        if (ifile->get() != ':')
        {
            ifile->unget();
            ifile->setstate(std::ios::badbit);
            throw Exception("Bad file");
        }


        *ifile >> std::skipws >> value;

        largestVectorSeen = std::max<int>(largestVectorSeen, idx + 1);
        if (lastIdxTotal > 0 && idx > lastIdxTotal)
        {
            std::cout << "Skipping index " << idx << std::endl;
            continue;
        }

        if (!ifile->good())
            throw Exception("Bad file");
        Offset offset(idx, value);
        fv->add(offset);

    }

    return fv;

}