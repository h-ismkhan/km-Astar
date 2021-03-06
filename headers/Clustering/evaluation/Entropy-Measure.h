#ifndef _Entropy_Measure_H
#define _Entropy_Measure_H

#include <math.h>

class Entropy_Measure
{
protected:
	vector<numclustype> clusterids, classids;
	vector<vector<numclustype>> clusters, classes;
	//Ej = -Sigma( Pij * log(Pij) )
	double minusSigmaPijLogPij(int j)
	{
		int /*numofclusters = clusters.size(), */numofclasses = classes.size();
		double Ej = 0;
		double nj = clusters[j].size();
		double nij = 0;

		for (numclustype i = 0; i < numofclasses; i++)
		{
			sizetype jth_iter = 0;
			sizetype ith_iter = 0;
			sizetype clsize = classes[i].size();
			nij = 0;
			while (jth_iter < nj && ith_iter < clsize)
			{
				if (clusters[j][jth_iter] == classes[i][ith_iter])
				{
					nij = nij + 1;
					jth_iter++;
					ith_iter++;
				}
				else if (clusters[j][jth_iter] > classes[i][ith_iter])
					ith_iter++;
				else
					jth_iter++;
			}
			if (nij != 0)
				Ej = Ej + (nij / nj) * log(nij / nj);
		}
		return -Ej;
	}
public:
	virtual void setClusterIds(const vector<numclustype> &cluster_ids)
	{
		sizetype size = cluster_ids.size();
		this->clusterids = cluster_ids;

		//initialize clusters:
		numclustype numofclusters = 0;
		for (sizetype i = 0; i < size; i++)
		{
			if (cluster_ids[i] > numofclusters)
				numofclusters = cluster_ids[i];
		}
		numofclusters++;
		vector<numclustype> t;
		clusters.assign(numofclusters, t);

		for (sizetype i = 0; i < size; i++)
		{
			clusters[cluster_ids[i]].push_back(i);
		}
	}
	virtual void setClassIds(const vector<numclustype>& class_ids)
	{
		sizetype size = class_ids.size();
		this->classids = class_ids;

		//initialize clusters:
		numclustype numofclasses = 0;
		for (sizetype i = 0; i < size; i++)
		{
			if (class_ids[i] > numofclasses)
				numofclasses = class_ids[i];
		}
		numofclasses++;
		vector<numclustype> t;
		classes.assign(numofclasses, t);

		for (sizetype i = 0; i < size; i++)
		{
			classes[class_ids[i]].push_back(i);
		}
	}
	virtual double measure()
	{
		assert(clusterids.size() == classids.size());
		
		double Ecs = 0;
		double n = clusterids.size(), numofclusters = clusters.size();
		for (numclustype j = 0; j < numofclusters; j++)
		{
			double nj = clusters[j].size();
			Ecs = Ecs + nj * minusSigmaPijLogPij(j) / n;
		}
		return Ecs;
	}
};

#endif