#ifndef _Elkan_H
#define _Elkan_H

#include "k-means-interface.h"
#include <assert.h>

class Elkan : public k_means_interface
{
	sizetype size;
	dimtype dim;
	numclustype the_k, numLowerBounds;

	vector<vector<coordtype>> starting_initial_centers;
	vector<vector<coordtype>> centers;

	const IInstance_Coordinated *inst;
	vector<numclustype> center_1st;
	vector<distype> center_1st_dis;
	vector<sizetype> sizeOf;

	vector<distype> centerMovement, centerCenterDistDiv2;
	vector<distype> s;// Half the distance between each center and its closest other center.
	vector<distype> upper, lower; // Lower/upper bound(s) for each point

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
	virtual numclustype UpdateCenters()
	{
		vector<vector<coordtype>> oldcenters = centers;

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

		for (numclustype ci = 0; ci < the_k; ci++)
			centerMovement[ci] = this->dis_metric->measure(centers[ci].data(), oldcenters[ci].data());

		numclustype furthestMovingCenter = 0;
		for (numclustype ci = 1; ci < the_k; ci++)
		if (centerMovement[ci] > centerMovement[furthestMovingCenter])
			furthestMovingCenter = ci;

		return furthestMovingCenter;
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

	void update_bounds() {
		for (sizetype i = 0; i < this->size; ++i) {
			upper[i] += centerMovement[center_1st[i]];
			for (numclustype j = 0; j < the_k; ++j) {
				lower[(sizetype)(i * numLowerBounds + j)] -= centerMovement[j];
			}
		}
	}

	void update_center_dists() {
		// find the inter-center distances
		for (numclustype c1 = 0; c1 < the_k; ++c1) {
			
				s[c1] = dismax;//Ismkhan: std::numeric_limits<double>::max() was error

				for (numclustype c2 = 0; c2 < the_k; ++c2) {
					// we do not need to consider the case when c1 == c2 as centerCenterDistDiv2[c1*k+c1]
					// is equal to zero from initialization, also this distance should not be used for s[c1]
					if (c1 != c2) {
						// divide by 2 here since we always use the inter-center
						// distances divided by 2
						centerCenterDistDiv2[c1 * the_k + c2] = dis_metric->measure(centers[c1].data(), centers[c2].data()) / 2.0;// sqrt(centerCenterDist2(c1, c2)) / 2.0;

						if (centerCenterDistDiv2[c1 * the_k + c2] < s[c1]) {
							s[c1] = centerCenterDistDiv2[c1 * the_k + c2];
						}
					}
				}
			
		}
	}
private:
	size_t iter_count = 0; 
public:
	virtual const vector<numclustype>& apply()
	{
		upper.assign(inst->size(), dismax);
		lower.assign(inst->size() * the_k, 0);

		initiate_centers();
		Euclidean_Distance defult_dis(inst->dimension());
		if (!this->dis_metric)
			this->dis_metric = &defult_dis;

		bool converged = false;
		iter_count = 1;
		UpdateClustersID();
		UpdateCenters();
		while ((iter_count < max_num_iter) && !converged) {
			++iter_count;

			update_center_dists();

			
			for (sizetype i = 0; i < this->size; ++i) {
				numclustype closest = center_1st[i];
				
				bool r = true;

				if (upper[i] <= s[closest]) {
					continue;
				}

				for (numclustype j = 0; j < the_k; ++j) {
					if (j == closest) { continue; }
					if (upper[i] <= lower[(sizetype)(i * the_k + j)]) { continue; }
					if (upper[i] <= centerCenterDistDiv2[closest * the_k + j]) { continue; }

					// ELKAN 3(a)
					if (r) {
						upper[i] = dis_metric->measure(centers[closest].data(), (*inst)[i]);//sqrt(pointCenterDist2(i, closest));
						/*cout << (i * k + closest) << endl;
						cout << (unsigned long long)(i * k + closest) << endl;*/
						lower[(sizetype)((sizetype)(i * the_k) + (sizetype)closest)] = upper[i];
						r = false;
						if ((upper[i] <= lower[(sizetype)(i * the_k + j)]) || (upper[i] <= centerCenterDistDiv2[closest * the_k + j])) {
							continue;
						}
					}

					// ELKAN 3(b)
					lower[(sizetype)(i * the_k + j)] = dis_metric->measure(centers[j].data(), (*inst)[i]); //sqrt(pointCenterDist2(i, j));
					if (lower[(sizetype)(i * the_k + j)] < upper[i]) {
						closest = j;
						upper[i] = lower[(sizetype)(i * the_k + j)];
					}
				}
				if (center_1st[i] != closest) {
					center_1st[i] = closest;
					//changeAssignment(i, closest, threadId);
				}
			}

			//verifyAssignment(iterations, startNdx, endNdx);

			// ELKAN 4, 5, AND 6
			//synchronizeAllThreads();
			//if (threadId == 0) {
			numclustype furthestMovingCenter = UpdateCenters();
			converged = (0.0 == centerMovement[furthestMovingCenter]);
			//}

			//synchronizeAllThreads();
			if (!converged) {
				update_bounds();
			}
			//synchronizeAllThreads();
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
	virtual inline void set_num_of_clusters(numclustype the_k)
	{ 
		numLowerBounds = the_k;
		s.assign(the_k, 0);
		centerCenterDistDiv2.assign(the_k * the_k, 0);
		centerMovement.assign(the_k, 0);
		this->the_k = the_k;
	}
	virtual inline void set_seeds_initializer(ISeed_Initializer *initializer){ this->seedInitializer = initializer; }

	virtual const vector<numclustype>& get_results()const { return center_1st; }
	virtual inline numclustype get_num_of_clusters()const{ return the_k; }
	virtual size_t get_consumed_iteration_count() const { return iter_count; }
	virtual const coordtype* get_starting_initial_centers(numclustype ci) const { return starting_initial_centers[ci].data(); }

	virtual string name()const { if (!seedInitializer)return "MyElkan"; return "MyElkan-" + seedInitializer->name(); }
};

#endif
