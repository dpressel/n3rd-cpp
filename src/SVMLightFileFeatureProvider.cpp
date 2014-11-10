#include "sgdtk/SVMLightFileFeatureProvider.h"
#include <iostream>


using namespace sgdtk;

Dims SVMLightFileFeatureProvider::findDims(String file)
{
    std::ifstream ifile;
    ifile.open(file.c_str());
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
    
    if (! ifile->is_open())
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

std::vector<FeatureVector*> SVMLightFileFeatureProvider::load(String file)
{
    std::vector<FeatureVector*> fvs;
    open(file);
    FeatureVector* fv;
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
    
FeatureVector* SVMLightFileFeatureProvider::next()
{
    if (! ifile->is_open())
    {
        throw Exception("File was not opened!");
    }
    String line;
    if (!std::getline(*ifile, line, '\n'))
    {
        ifile->close();
        delete ifile;
        return NULL;
    }

    int lastIdxTotal = maxFeatures - 1;
    Offsets sv;

    StringArray strings = split(line);
    double label = valueOf<double>(strings[0]);

    FeatureVector* fv = new FeatureVector(lastIdxTotal + 1, label);
    for (int i = 1, sz = strings.size(); i < sz; ++i)
    {
        StringArray tok = split(strings[i], ':');
        int idx = valueOf<int>(tok[0]);
        
        if (idx > lastIdxTotal)
        {
            continue;
        }
        double value = valueOf<double>(tok[1]);
        Offset offset(idx, value);
        fv->add(offset);

    }
    return fv;

}