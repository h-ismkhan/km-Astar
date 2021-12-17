#ifndef _k_means_H
#define _k_means_H

#include "k-means-interface.h"
#include <assert.h>

class k_means : public k_means_interface
{
	sizetype size;
	dimtype dim;
	numclustype the_k;

	vector<vector<coordtype>> starting_initial_centers;
	vector<vector<coordtype>> centers;

	const IInstance_Coordinated *inst;
	vector<numclustype> center_1st;
	vector<distype> center_1st_dis;
	vector<sizetype> sizeOf;

	ISeed_Initializer *seedInitializer = 0;
	size_t max_num_iter = SIZE_MAX;

	IDistance *dis_metric = 0;

protected:
	virtual void initiate_centers()
	{
		if (this->seedInitializer)
		{
			this->seedInitializer->set_instance(inst);
			this->seedInitializer->set_the_k(the_k);
		}
		else assert(this->seedInitializer != 0);
		centers.assign(the_k, vector<coordtype>(dim));
		const vector<const coordtype*>& seeds_coords = this->seedInitializer->next_seeds();
		for (numclustype ci = 0; ci < the_k; ci++)
		{
			for (dimtype di = 0; di < dim; di++)
				centers[ci][di] = seeds_coords[ci][di];
		}		
		starting_initial_centers = centers;
	}
	virtual void UpdateCenters()
	{
		for (numclustype ci = 0; ci < the_k; ci++)
		{
			sizeOf[ci] = 0;
			for (dimtype j = 0; j < dim; j++)
				centers[ci][j] = 0;
		}
		for (sizetype ni = 0; ni < size; ni++)
		{
			sizeOf[center_1st[ni]]++;
			numclustype c_id = center_1st[ni];
			const coordtype* row = (*inst)[ni];
			for (dimtype j = 0; j < dim; j++)
				centers[c_id][j] = centers[c_id][j] + row[j];
		}
		for (numclustype ci = 0; ci < the_k; ci++)
		{
			if (sizeOf[ci])
			{
				for (dimtype j = 0; j < dim; j++)
					centers[ci][j] = centers[ci][j] / sizeOf[ci];
			}
		}
	}

	virtual bool UpdateClustersID()
	{
		bool IsChanged = 0;
		for (sizetype ni = 0; ni < size; ni++)
		{
			distype min_dis = dismax;
			numclustype index = -1;
			for (numclustype cj = 0; cj < the_k; cj++)
			{
				distype cur_dis = dis_metric->measure((*inst)[ni], centers[cj].data());

				if (cur_dis < min_dis)
				{
					min_dis = cur_dis;
					index = cj;
				}
			}

			center_1st_dis[ni] = min_dis;

			if (index != center_1st[ni] && index >= 0)
			{
				IsChanged = 1;
				center_1st[ni] = index;
			}
		}
		return IsChanged;
	}
private:
	size_t iter_count = 0; 
public:
	virtual const vector<numclustype>& apply()
	{
		initiate_centers();
		Euclidean_Distance defult_dis(inst->dimension());
		if (!this->dis_metric)
			this->dis_metric = &defult_dis;

		iter_count = 0;
		while (iter_count < max_num_iter)
		{
			iter_count++;

			if (!UpdateClustersID())
				break;
			UpdateCenters();
		}

		return center_1st;
	}
	virtual inline void set_instance(const IInstance_Coordinated *inst)
	{
		this->inst = inst; this->size = inst->size(); this->dim = inst->dimension();
		center_1st.assign(inst->size(), -1);
		center_1st_dis.assign(inst->size(), dismax);
		sizeOf.assign(inst->size(), -0);
	}
	virtual inline void set_distance_metric(IDistance* dis_metric) { this->dis_metric = dis_metric; }
	virtual inline void set_max_num_iteration(size_t max_num_iter = SIZE_MAX){ this->max_num_iter = max_num_iter; }
	virtual inline void set_num_of_clusters(numclustype the_k){ this->the_k = the_k; }
	virtual inline void set_seeds_initializer(ISeed_Initializer *initializer){ this->seedInitializer = initializer; }

	virtual const vector<numclustype>& get_results()const { return center_1st; }
	virtual inline numclustype get_num_of_clusters()const{ return the_k; }
	virtual size_t get_consumed_iteration_count() const { return iter_count; }
	virtual const coordtype* get_starting_initial_centers(numclustype ci) const { return starting_initial_centers[ci].data(); }

	virtual string name()const { if (!seedInitializer)return "Lloyd"; return "Lloyd-" + seedInitializer->name(); }
};

#endif
