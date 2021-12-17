#ifndef _k_means_astar_H
#define _k_means_astar_H

#include "k-means-interface.h"
#include "../Distance/Euclidean-Distance.h"
#include <assert.h>
#include <random>
#include "../Utils/SparsVector.h"
#include "../../headers-external/annoy/annoylib.h"
#include "../../headers-external/annoy/kissrandom.h"

#define min_k_means_astar(a, b) ((a) < (b) ? (a) : (b))

class k_means_astar : public k_means_interface
{
	sizetype size;
	dimtype dim;
	numclustype the_k;

	sizetype maxsize = -1;
	numclustype maxnumclus = -1;

	vector<vector<coordtype>> starting_initial_centers;
	vector<vector<coordtype>> centers;

	const IInstance_Coordinated *inst;
	vector<numclustype> center_1st;
	vector<distype> center_1st_dis;
	vector<sizetype> sizeOf;

	ISeed_Initializer *seedInitializer = 0;
	size_t max_num_iter = SIZE_MAX;

	IDistance *dis_metric = 0;

	vector <bool> isCenterChanged;
	vector<numclustype> changedCentersList;
		
	vector<numclustype> center_2nd;
	vector<distype> center_2nd_dis;

	vector<numclustype> center_eff;
	vector<distype> center_eff_dis;

	vector<vector<numclustype>> adjListId;
	vector<vector<distype>> adjListId_dis;
	vector<vector<distype>> CCs_dis;
	
	vector<numclustype> center_old_1st;
	vector<vector<coordtype>> centers_mul_sizeOfC;

private:
	virtual void initiate_centers()
	{
		changedCentersList.clear();
		centers_mul_sizeOfC.clear();
		for (numclustype j = 0; j < the_k; j++)
		{
			if (isCenterChanged[j])
				changedCentersList.push_back(j);

			vector<coordtype> vec(dim, 0);
			centers_mul_sizeOfC.push_back(vec);
		}

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
		for (numclustype i = 0; i < the_k; i++)
			sizeOf[i] = 0;
		for (numclustype i = 0; i < size; i++)
			sizeOf[center_1st[i]]++;

		for (numclustype i = 0; i < the_k; i++)
		{
			if (sizeOf[i])
			{
				for (dimtype j = 0; j < dim; j++)
					centers[i][j] = centers_mul_sizeOfC[i][j] / (coordtype)sizeOf[i];
			}
		}


		if (center_old_1st.size() > 0)
		{
			this->isCenterChanged.assign(the_k, false);
			for (sizetype i = 0; i < size; i++)
			{
				if (this->center_1st[i] != this->center_old_1st[i])
				{
					this->isCenterChanged[center_1st[i]] = true;
					if (center_old_1st[i] != maxnumclus)
						this->isCenterChanged[center_old_1st[i]] = true;
				}
			}
		}
		else
			this->isCenterChanged.assign(the_k, true);
		center_old_1st = this->center_1st;
	}

	virtual bool isInActive(const sizetype point)
	{
		if (this->center_eff[point] != maxnumclus)
		{
			if (!this->isCenterChanged[this->center_1st[point]] &&
				!this->isCenterChanged[this->center_2nd[point]] &&
				!this->isCenterChanged[this->center_eff[point]])
				return true;
		}
		else if (this->center_2nd[point] != maxnumclus)
		{
			if (!this->isCenterChanged[this->center_1st[point]] &&
				!this->isCenterChanged[this->center_2nd[point]])
				return true;
		}

		return false;
	}
	virtual void updateSimilarpoints(vector<vector<sizetype>> &simlarpoints_List,
		vector<numclustype>& categoryId_of_points)
	{
		vector<numclustype> maxclus_vec(this->the_k + 1, maxnumclus);
		vector<vector<numclustype>> indexof_similarpointsList(this->the_k, maxclus_vec);
		simlarpoints_List.clear();

		for (sizetype point = 0; point < this->size; point++)
		{
			numclustype i = this->center_1st[point];
			numclustype j = this->center_2nd[point];

			if (j == maxnumclus)
			{
				if (indexof_similarpointsList[i][this->the_k] == maxnumclus)
				{
					indexof_similarpointsList[i][this->the_k] = simlarpoints_List.size();
					vector<sizetype> list;
					list.push_back(point);
					simlarpoints_List.push_back(list);
				}
				else
				{
					simlarpoints_List[indexof_similarpointsList[i][this->the_k]].push_back(point);
				}
				continue;
			}

			if (indexof_similarpointsList[i][j] == maxnumclus)
			{
				indexof_similarpointsList[i][j] = simlarpoints_List.size();
				vector<sizetype> list;
				list.push_back(point);
				simlarpoints_List.push_back(list);
			}
			else
			{
				simlarpoints_List[indexof_similarpointsList[i][j]].push_back(point);
			}
		}
		numclustype num_categories = simlarpoints_List.size();
		categoryId_of_points.assign(this->size, 0);
		for (numclustype i = 0; i < num_categories; i++)
		{
			sizetype catSize = simlarpoints_List[i].size();
			for (sizetype j = 0; j < catSize; j++)
				categoryId_of_points[simlarpoints_List[i][j]] = i;
		}
	}

	//NOTE: adjListId_dis is currently dismax
	virtual void updateAdjancy()
	{
		vector<numclustype> empty_clustype_vec;
		adjListId.assign(this->the_k, empty_clustype_vec);
		vector<distype> empty_distype_vec;
		adjListId_dis.assign(this->the_k, empty_distype_vec);

		vector<distype> max_distype_vec(this->the_k, dismax);
		CCs_dis.assign(this->the_k, max_distype_vec);

		vector<bool> vecBool(this->the_k, false);
		vector<vector<bool>> isIn(this->the_k, vecBool);

		for (sizetype point = 0; point < this->size; point++)
		{
			numclustype i = this->center_1st[point];
			numclustype j = this->center_2nd[point];
			numclustype k = this->center_eff[point];

			if (j == maxnumclus) continue;
			distype dis = dis_metric->measure(this->centers[i].data(), this->centers[j].data());
			CCs_dis[i][j] = CCs_dis[j][i] = dis;

			if (!isIn[i][j])
			{
				isIn[i][j] = true;
				adjListId[i].push_back(j);
				adjListId_dis[i].push_back(dis);
			}
			if (!isIn[j][i])
			{
				isIn[j][i] = true;
				adjListId[j].push_back(i);
				adjListId_dis[j].push_back(dis);
			}

			if (k == maxnumclus) continue;
			dis = dis_metric->measure(this->centers[i].data(), this->centers[k].data());
			CCs_dis[i][k] = CCs_dis[k][i] = dis;
			if (!isIn[i][k])
			{
				isIn[i][k] = true;
				adjListId[i].push_back(k);
				adjListId_dis[i].push_back(dis);
			}
			if (!isIn[k][i])
			{
				isIn[k][i] = true;
				adjListId[k].push_back(i);
				adjListId_dis[k].push_back(dis);
			}
		}
	}

	void determine_leaders(vector<vector<sizetype>> &Categories_ofpoints
		, vector<numclustype>& inverseIndex_Leaderpoint_to_Category
		, vector<bool>& isa_leaderpoint, vector<vector<sizetype>>& leaderpoints_category
		, vector<sizetype>& leaderpoints)
	{
		numclustype num_categories = Categories_ofpoints.size();

		vector<sizetype> blank_sizetype_vec;
		inverseIndex_Leaderpoint_to_Category.assign(this->size, maxsize);
		isa_leaderpoint.assign(this->size, false);
		leaderpoints_category.assign(num_categories, blank_sizetype_vec);

		SparsVector<sizetype> max_minus_third_first(this->the_k, maxsize);
		//SparsVector<sizetype> min_minus_third_first(this->the_k, maxsize);

		for (numclustype i = 0; i < num_categories; i++)
		{
			sizetype categorySize = Categories_ofpoints[i].size();

			//third centers process			
			for (sizetype j = 0; j < categorySize; j++)
			{
				if (isInActive(Categories_ofpoints[i][j]))
					continue;
				sizetype point = Categories_ofpoints[i][j];
				if (center_eff[point] != maxnumclus)
				{
					if (max_minus_third_first[center_eff[point]] != maxsize)
					{
						distype prevW = center_eff_dis[max_minus_third_first[center_eff[point]]];
						distype curW = center_eff_dis[point];

						if (curW < prevW)
							max_minus_third_first[center_eff[point]] = point;
					}
					else max_minus_third_first[center_eff[point]] = point;
					
				}
			}

			const vector<size_t>& ass = max_minus_third_first.getActiveIndexes();
			for (sizetype ji = 0; ji < ass.size(); ji++)
			{
				size_t j = ass[ji];
				sizetype point = max_minus_third_first[j];

				if (isInActive(point))
					continue;

				sizetype leader_point = maxsize;

				leader_point = max_minus_third_first[center_eff[point]];
				if (!isa_leaderpoint[leader_point])
				{
					isa_leaderpoint[leader_point] = true;
					leaderpoints_category[i].push_back(leader_point);
					leaderpoints.push_back(leader_point);
					inverseIndex_Leaderpoint_to_Category[leader_point] = i;
				}

			}
			max_minus_third_first.reset();
			
			sizetype min_minus_sec_first = maxsize;
			sizetype max_sum_first_sec = maxsize;
						
			distype min_minus_sec_first_dis = dismax;
			distype max_sum_first_sec_dis = 0;

			for (sizetype j = 0; j < categorySize; j++)
			{
				sizetype point = Categories_ofpoints[i][j];
				distype first_dis = this->center_1st_dis[point];
				distype second_dis = this->center_2nd_dis[point];

				distype the_sum = second_dis + first_dis, the_minus = second_dis - first_dis;
								
				if (the_minus < min_minus_sec_first_dis)
				{
					min_minus_sec_first_dis = second_dis - first_dis;
					min_minus_sec_first = point;
				}
				if (the_sum > max_sum_first_sec_dis)
				{
					max_sum_first_sec_dis = the_sum;
					max_sum_first_sec = point;
				}
			}
			sizetype sf_point;
			sf_point = min_minus_sec_first;
			if (sf_point != maxsize)
			{
				if (!isa_leaderpoint[sf_point])
				{
					isa_leaderpoint[sf_point] = true;
					leaderpoints_category[i].push_back(sf_point);
					leaderpoints.push_back(sf_point);
					inverseIndex_Leaderpoint_to_Category[sf_point] = i;
				}
			}
			sf_point = max_sum_first_sec;
			if (sf_point != maxsize)
			{
				if (!isa_leaderpoint[sf_point])
				{
					isa_leaderpoint[sf_point] = true;
					leaderpoints_category[i].push_back(sf_point);
					leaderpoints.push_back(sf_point);
					inverseIndex_Leaderpoint_to_Category[sf_point] = i;
				}
			}
		}//end of determining leader points
	}

	virtual void updated_1st_center(const sizetype point, SparsVector<distype> &visited_centers)
	{
		//each center is added to its adjacent, so we start with null nearest center.  		
		visited_centers[center_1st[point]] = this->center_1st_dis[point] =
			dis_metric->measure((*inst)[point], this->centers[center_1st[point]].data());

		while (true)
		{
			bool isUpdated = false;
			numclustype axisClusId = this->center_1st[point];
			numclustype lsize = adjListId[axisClusId].size();
			for (numclustype i = 0; i < lsize; i++)
			{
				numclustype cc = adjListId[axisClusId][i];
				distype cur_dis = visited_centers[cc];
				if (cur_dis == dismax)
					visited_centers[cc] = cur_dis = dis_metric->measure((*inst)[point], this->centers[cc].data());

				if (cur_dis < this->center_1st_dis[point])
				{
					this->center_1st_dis[point] = cur_dis;
					this->center_1st[point] = cc;
					isUpdated = true;
				}
			}
			if (!isUpdated)
				break;
		}
	}

	virtual void updated_2nd_3rd_center(const sizetype point, SparsVector<distype> &visited_centers)
	{
		this->center_2nd[point] = maxnumclus;
		this->center_2nd_dis[point] = dismax;
		this->center_eff[point] = maxnumclus;
		this->center_eff_dis[point] = dismax;

		numclustype axisClusId = this->center_1st[point];

		const vector<size_t>& visited_centers_list = visited_centers.getActiveIndexes();
		for (size_t li = 0; li < visited_centers_list.size(); li++)
		{
			numclustype cc = visited_centers_list[li];
			if (cc == center_1st[point])
				continue;
			distype cur_dis = visited_centers[cc];

			if (cur_dis < this->center_2nd_dis[point])
			{
				this->center_2nd_dis[point] = cur_dis;
				this->center_2nd[point] = cc;
			}
			if (cur_dis < this->center_eff_dis[point])
			{
				distype cc_dis = CCs_dis[center_1st[point]][cc];
				if (cc_dis == dismax)
					cc_dis = CCs_dis[center_1st[point]][cc] = CCs_dis[cc][center_1st[point]]
					= dis_metric->measure(centers[center_1st[point]].data(), centers[cc].data());

				if (cur_dis < cc_dis)
				{
					this->center_eff_dis[point] = cur_dis;
					this->center_eff[point] = cc;
				}
			}
		}

		/*if (this->center_2nd[point] == this->center_eff[point])
		{
			this->center_eff[point] = maxnumclus;
			this->center_eff_dis[point] = dismax;
		}*/

	}

	virtual bool UpdateClustersID_Approximately()
	{
		static distype pSSEDM = dismax;
		vector<numclustype> p_center_1st = center_1st;
		vector<distype> p_center_1st_dis = center_1st_dis;

		vector<vector<sizetype>> Categories_ofpoints;
		vector<numclustype> categoryId_of_points;
		updateSimilarpoints(Categories_ofpoints, categoryId_of_points);
		numclustype num_categories = Categories_ofpoints.size();

		//determining leader points
		vector<numclustype> inverseIndex_Leaderpoint_to_Category;
		vector<bool> isa_leaderpoint;
		vector<vector<sizetype>> leaderpoints_category;
		vector<sizetype> leaderpoints;
		SparsVector<bool> isThisCenterUsed(the_k, false);

		determine_leaders(Categories_ofpoints, inverseIndex_Leaderpoint_to_Category, isa_leaderpoint
			, leaderpoints_category, leaderpoints);

		updateAdjancy();

		bool IsChanged = false;

		//determining first and second centers of leader points
		sizetype leaderpoints_size = leaderpoints.size();

		//___L this sort is needed for large scale reading points from file:
		sort(leaderpoints.begin(), leaderpoints.end());

		SparsVector<distype> visited_centers(this->the_k, dismax);

		vector<distype> dis_C_vec(this->the_k, dismax);
		for (sizetype lp = 0; lp < leaderpoints_size; lp++)
		{
			sizetype point = leaderpoints[lp];
			if (isInActive(point))
				continue;

			//this is used in the end of the loop, but it is here to use cach option of Large_inst.
			const coordtype* rowpoint = (*inst)[point];

			numclustype oldIndex = center_1st[point];
			visited_centers.reset();
			updated_1st_center(point, visited_centers);
			updated_2nd_3rd_center(point, visited_centers);
			if (oldIndex != center_1st[point])
				IsChanged = 1;

			numclustype old_cid = center_old_1st[point];
			numclustype nw_cid = center_1st[point];
			if (nw_cid != old_cid)
			{
				for (dimtype j = 0; j < dim; j++)
				{
					centers_mul_sizeOfC[nw_cid][j] = centers_mul_sizeOfC[nw_cid][j] + rowpoint[j];
					centers_mul_sizeOfC[old_cid][j] = centers_mul_sizeOfC[old_cid][j] - rowpoint[j];
				}
			}
		}// end of determining first and second centers of leader points

		//determining which centers should be considered for other points, rather than leader point, of same category.
		vector<numclustype> blank_numclustype_vec;
		vector<vector<numclustype>> selectedCentersof_category(num_categories, blank_numclustype_vec);

		for (sizetype i = 0; i < num_categories; i++)
		{

			sizetype categoryLeadersCount = leaderpoints_category[i].size();
			for (sizetype j = 0; j < categoryLeadersCount; j++)
			{
				numclustype center = this->center_1st[leaderpoints_category[i][j]];
				if (!isThisCenterUsed[center])
				{
					selectedCentersof_category[i].push_back(center);
					isThisCenterUsed[center] = true;
				}

				center = this->center_2nd[leaderpoints_category[i][j]];
				if (center == maxnumclus) continue;
				if (!isThisCenterUsed[center])
				{
					selectedCentersof_category[i].push_back(center);
					isThisCenterUsed[center] = true;
				}
				center = this->center_eff[leaderpoints_category[i][j]];
				if (center == maxnumclus) continue;
				if (!isThisCenterUsed[center])
				{
					selectedCentersof_category[i].push_back(center);
					isThisCenterUsed[center] = true;
				}
			}
			isThisCenterUsed.reset();
		}// end of determining which centers should be considered for other points, rather than leader point, of same category.

		//determining first and second centers of NON-leader points		
		for (sizetype i = 0; i < size; i++)
		{
			if (isa_leaderpoint[i]) continue;
			if (isInActive(i))
				continue;

			distype min_dis = dismax;
			numclustype index = maxnumclus;

			vector<distype> dis_c;
			numclustype selectedCenterSize = selectedCentersof_category[categoryId_of_points[i]].size();
			if (selectedCenterSize == 0)
			{
				continue;
			}

			//this is used in the end of the loop, but it is here to use cach option of Large_inst.
			const coordtype* point = (*inst)[i];

			for (numclustype cj = 0; cj < selectedCenterSize; cj++)
			{
				numclustype j = selectedCentersof_category[categoryId_of_points[i]][cj];
				distype cur_dis = dis_metric->measure((*inst)[i], centers[j].data());

				if (cur_dis < min_dis)
				{
					min_dis = cur_dis;
					index = j;
				}
				dis_c.push_back(cur_dis);
			}

			center_1st_dis[i] = min_dis;

			if (index != center_1st[i] && index != maxnumclus)
			{
				IsChanged = 1;
				center_1st[i] = index;

				numclustype previous_cid = center_old_1st[i];
				numclustype c_id = center_1st[i];
				if (c_id != previous_cid)
				{
					for (dimtype j = 0; j < dim; j++)
					{
						centers_mul_sizeOfC[c_id][j] = centers_mul_sizeOfC[c_id][j] + point[j];
						centers_mul_sizeOfC[previous_cid][j] = centers_mul_sizeOfC[previous_cid][j] - point[j];
					}
				}
			}

			this->center_2nd[i] = maxnumclus;
			this->center_2nd_dis[i] = dismax;
			this->center_eff[i] = maxnumclus;
			this->center_eff_dis[i] = dismax;

			for (numclustype cj = 0; cj < selectedCenterSize; cj++)
			{
				numclustype j = selectedCentersof_category[categoryId_of_points[i]][cj];

				if (j == center_1st[i]) continue;

				distype cur_dis = dis_c[cj];

				if (cur_dis < this->center_2nd_dis[i])
				{
					this->center_2nd_dis[i] = cur_dis;
					this->center_2nd[i] = j;
				}
				if (cur_dis < this->center_eff_dis[i])
				{
					distype cc_dis = dis_metric->measure(centers[center_1st[i]].data(), centers[j].data());
					if (cur_dis < cc_dis)
					{
						this->center_eff_dis[i] = cur_dis;
						this->center_eff[i] = j;
					}
				}
			}
			/*if (this->center_2nd[i] == this->center_eff[i])
			{
				this->center_eff[i] = maxnumclus;
				this->center_eff_dis[i] = dismax;
			}*/
		}

		distype cSSEDM = 0;
		for (sizetype node = 0; node < inst->size(); node++)
			cSSEDM = cSSEDM + center_1st_dis[node] * center_1st_dis[node];
		if (cSSEDM < pSSEDM)
		{
#ifdef printmode
			std::cout << "itr: " << iter_count << ", SSEDM til now: " << cSSEDM << std::endl;
#endif
			if (IsChanged)
			{
				pSSEDM = cSSEDM;
				return true;
			}
		}
		center_1st_dis = p_center_1st_dis;
		center_1st = p_center_1st;
		pSSEDM = dismax;

		return false;
	}
	virtual bool UpdateClustersID()
	{
		if (iter_count > 1)
			return UpdateClustersID_Approximately();

		AnnoyIndex<numclustype, coordtype, Euclidean, Kiss32Random> annoyTree((int)dim);

		for (numclustype i = 0; i < this->the_k; ++i){
			const coordtype *vec = centers[i].data();
			annoyTree.add_item(i, vec);
		}
		annoyTree.build(min_k_means_astar(2 * (int)dim, 20));
		
		std::default_random_engine generator;
		std::normal_distribution<double> distribution(0.5, 0.5);
		vector<double> randomVec(this->dim, 0);
		for (dimtype i = 0; i < this->dim; i++)
		{
			randomVec[i] = distribution(generator); //Random01();
			while (randomVec[i] == 0)
				randomVec[i] = distribution(generator);
		}

		vector<pair<double, numclustype>> innerProductCenters(this->the_k);
		for (numclustype i = 0; i < this->the_k; i++)
		{
			innerProductCenters[i].first = 0;
			innerProductCenters[i].second = i;
			for (dimtype j = 0; j < this->dim; j++)
				innerProductCenters[i].first = innerProductCenters[i].first + randomVec[j] * this->centers[i][j];
		}
		sort(innerProductCenters.begin(), innerProductCenters.end());

		vector<distype> dis_C_vec(this->the_k, dismax);
		vector<vector<distype>> ccdis(this->the_k, dis_C_vec);

		SparsVector<bool> is_dis_p_c_available(the_k, false);
		for (sizetype point = 0; point < inst->size(); point++)
		{
			is_dis_p_c_available.reset();

			vector<pair<coordtype, numclustype>> dis_point_from_C_incOrder;

			//get info from annoy
			vector<numclustype> ids;
			vector<coordtype> cdis_anny;
			vector<distype> cdis_compatible_ours;
			annoyTree.get_nns_by_vector((*inst)[point], this->the_k, 3, &ids, &cdis_anny);
			for (size_t cdi = 0; cdi < cdis_anny.size(); cdi++)
			{
				if (is_dis_p_c_available[ids[cdi]]) continue;
				is_dis_p_c_available[ids[cdi]] = true;
				dis_point_from_C_incOrder.push_back(make_pair((distype)cdis_anny[cdi], ids[cdi]));
			}

			//get info from hash vector
			distype innerp_point = 0;
			const coordtype* point_vec = (*inst)[point];
			for (dimtype j = 0; j < this->dim; j++)
				innerp_point = innerp_point + randomVec[j] * point_vec[j];			
			
			vector<pair<double, numclustype>> diff_innerpP_from_innerpC = innerProductCenters;
			for (pair<double, numclustype>& element : diff_innerpP_from_innerpC)
				element.first = abs(element.first - innerp_point);
			sort(diff_innerpP_from_innerpC.begin(), diff_innerpP_from_innerpC.end());	

			numclustype firstForSize = min_k_means_astar(4, this->the_k);
			vector<numclustype> centersToBeConsidered(firstForSize);
			for (numclustype cci = 0; cci < firstForSize; cci++)
				centersToBeConsidered[cci] = diff_innerpP_from_innerpC[cci].second;

			//combine what gained from annoy and what from hash-vec						
			for (numclustype cj = 0; cj < firstForSize; cj++)
			{
				if (is_dis_p_c_available[centersToBeConsidered[cj]]) continue;
				is_dis_p_c_available[centersToBeConsidered[cj]] = true;

				dis_point_from_C_incOrder.push_back(
					make_pair(dis_metric->measure((*inst)[point], centers[centersToBeConsidered[cj]].data()),
					centersToBeConsidered[cj]));
			}
			sort(dis_point_from_C_incOrder.begin(), dis_point_from_C_incOrder.end());

			center_1st[point] = dis_point_from_C_incOrder[0].second;
			center_1st_dis[point] = dis_point_from_C_incOrder[0].first;

			center_2nd[point] = dis_point_from_C_incOrder[1].second;
			center_2nd_dis[point] = dis_point_from_C_incOrder[1].first;

			numclustype first = center_1st[point];
			for (size_t i = 1; i < dis_point_from_C_incOrder.size(); i++)
			{
				numclustype otherC = dis_point_from_C_incOrder[i].second;
				distype cc = ccdis[first][otherC];
				if (cc == dismax)
					cc = ccdis[first][otherC] = ccdis[otherC][first]
					= dis_metric->measure(centers[first].data(), centers[otherC].data());

				if (cc > dis_point_from_C_incOrder[i].first)
				{
					center_eff[point] = dis_point_from_C_incOrder[i].second;
					center_eff_dis[point] = dis_point_from_C_incOrder[i].first;
					break;
				}
			}
			numclustype c_id = center_1st[point];
			for (dimtype j = 0; j < dim; j++)
				centers_mul_sizeOfC[c_id][j] = centers_mul_sizeOfC[c_id][j] + point_vec[j];
		}
		return true;
	}
private:
	size_t iter_count = 0;
public:
	virtual const vector<numclustype>& apply()
	{
		set_instance(inst);
		set_num_of_clusters(the_k);
		
		Euclidean_Distance defult_dis(inst->dimension());
		if (!this->dis_metric)
			this->dis_metric = &defult_dis;

		dimtype origdim = this->dim;
		this->dim = this->dis_metric->get_dimension();

		initiate_centers();
		
		iter_count = 0;
		while (iter_count < max_num_iter)
		{
			iter_count++;

			if (!UpdateClustersID())
				break;
			UpdateCenters();
		}

		this->dim = origdim;

		return center_1st;
	}
	virtual inline void set_instance(const IInstance_Coordinated *inst)
	{
		this->inst = inst; this->size = inst->size(); this->dim = inst->dimension();
		center_1st.assign(inst->size(), maxnumclus);
		center_1st_dis.assign(inst->size(), dismax);
		sizeOf.assign(inst->size(), 0);

		center_2nd_dis.assign(size, dismax);
		center_2nd.assign(size, maxnumclus);
		center_eff_dis.assign(size, dismax);
		center_eff.assign(size, maxnumclus);

		center_old_1st.clear();
	}

	virtual inline void set_distance_metric(IDistance* dis_metric) { this->dis_metric = dis_metric; }
	virtual inline void set_max_num_iteration(size_t max_num_iter = SIZE_MAX){ this->max_num_iter = max_num_iter; }
	virtual inline void set_num_of_clusters(numclustype the_k){ this->the_k = the_k; isCenterChanged.assign(the_k, true); }
	virtual inline void set_seeds_initializer(ISeed_Initializer *initializer){ this->seedInitializer = initializer; }

	virtual const vector<numclustype>& get_results()const { return center_1st; }
	virtual inline numclustype get_num_of_clusters()const{ return the_k; }
	virtual size_t get_consumed_iteration_count() const { return iter_count; }
	virtual const coordtype* get_starting_initial_centers(numclustype ci) const { return starting_initial_centers[ci].data(); }

	virtual string name()const { if (!seedInitializer)return "km-astar"; return "km-astar-" + seedInitializer->name(); }
};
#endif
