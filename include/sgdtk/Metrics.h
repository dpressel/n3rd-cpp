#ifndef __SGDTK_METRICS_H__
#define __SGDTK_METRICS_H__

namespace sgdtk
{
    
class Metrics
{
public:
    double cost;
    double totalLoss;
    double numExamplesSeen;
    double numEventsSeen;
    double totalError;

    Metrics()
    {
        clear();
    }
    ~Metrics()
    {

    }
    void clear()
    {
        cost = totalLoss = numEventsSeen = numExamplesSeen = totalError = 0.;
    }
    double getCost() const
    {
        return cost;
    }

    void setCost(double cost)
    {
        this->cost = cost;
    }

    double getTotalLoss() const
    {
        return totalLoss;
    }

    void setTotalLoss(double totalLoss)
    {
        this->totalLoss = totalLoss;
    }

    double getNumExamplesSeen() const
    {
        return numExamplesSeen;
    }

    void setNumExamplesSeen(double numExamplesSeen)
    {
        this->numExamplesSeen = numExamplesSeen;
    }

    void addToTotalExamples(double length)
    {
        numExamplesSeen += length;
    }

    double getTotalError() const
    {
        return totalError;
    }

    void setTotalError(double totalError)
    {
        this->totalError = totalError;
    }

    void addToTotalError(double error)
    {
        totalError += error;
    }

    void add(double loss, double error)
    {
        this->numExamplesSeen += 1.;
        this->numEventsSeen += 1.;

        this->totalLoss += loss;
        this->totalError += error;
    }

    double getLoss() const
    {
        double seen = numExamplesSeen > 0. ? numExamplesSeen : 1.;
        return totalLoss / seen;
    }

    double getError() const
    {
        double seen = numEventsSeen > 0. ? numEventsSeen : 1.;
        return totalError / seen;
    }

    double getNumEventsSeen() const
    {
        return numEventsSeen;
    }

    void setNumEventsSeen(double numEventsSeen)
    {
        this->numEventsSeen = numEventsSeen;
    }

    void addToTotalEvents(double length)
    {
        numEventsSeen += length;
    }
};
}
#endif