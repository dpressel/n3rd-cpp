#ifndef SGDTK_SVM_LIGHT_FILE_FEATURE_PROVIDER_H__
#define SGDTK_SVM_LIGHT_FILE_FEATURE_PROVIDER_H__

#include "sgdtk/Exception.h"
#include "sgdtk/Types.h"
#include "sgdtk/FeatureVector.h"
#include "sgdtk/FeatureProvider.h"
#include <fstream>
#include <memory>

namespace sgdtk
{

struct Dims
{
    Dims(size_t w = 0, size_t h = 0) :
        width(w), height(h) {}

    size_t width;
    size_t height;
};


/**
 * This reads in Sparse SVM light/Libsvm format data a stream (via a pull).
 *
 * We require a width to be provided for the feature vector.  This should be large enough to contain the vector.
 * This class is pretty quick and dirty, as its assumed that real-life problems will be more complex, and warrant
 * a different methodology and perhaps a {@link org.sgdtk.FeatureNameEncoder}, but for pre-processed sample data
 * which is already in vector form, encoding the features is not necessary, so this implementation can be trivial.
 *
 * @author dpressel
 */
class SVMLightFileFeatureProvider : public FeatureProvider
{
    size_t maxFeatures;
    std::ifstream* ifile;
    int largestVectorSeen;
public:

    int getLargestVectorSeen()
    {
        return largestVectorSeen;
    }
    /**
     * If you want to know the dimensions of an SVM light file, you can call this method, and it will give back
     * the number of vectors (as the height), and required feature vector size as the width to encompass all examples.
     *
     * @param file An SVM light type file
     * @return Number of feature vectors by number of features in feature vector
     * @throws IOException
     */ 
    static Dims findDims(String trainFile);

    /**
     * Create a provider, and you need to cap it with a max number of features.
     * Anything beyond this feature vector length will be clipped out of the resultant FV.
     *
     * @param maxFeatures The feature vector width.
     */
    SVMLightFileFeatureProvider(size_t mx = 0) :
        maxFeatures(mx), ifile(NULL), largestVectorSeen(0) {}

    virtual ~SVMLightFileFeatureProvider() {}

     /**
     * Slurp the entire file into memory.  This is not the recommended way to read large datasets, use
     * {@link #next()} to stream features from the file one by one.
     *
     * @param file An SVM light type file
     * @return One feature vector per line in the file
     * @throws IOException
     */
    std::vector<FeatureVector*> load(String file);

    /**
     * Get the next feature vector in the file
     *
     * @return The next feature vector, or null, if we are out of lines
     * @throws IOException
     */
    FeatureVector* next();

    /**
     * Open a file for reading.  All files are read only up to maxFeatures.
     * @param file An SVM light type file
     * @throws Exception
     */
    void open(String file);

    /**
     * Close the currently loaded file
     * @throws Exception
     */
    void close();



};
}

#endif