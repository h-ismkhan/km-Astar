#ifndef _ball_noRing_orig_k_means_H
#define _ball_noRingorig_k_means_H

//Original code from: https://github.com/syxiaa/ball-k-means

#include "k-means-interface.h"
#include "ball-km-def.h"

class ball_noRing_k_means_orig : public k_means_interface
{
	/*sizetype size;
	dimtype dim;*/
	numclustype the_k;
	MatrixOur data;


	size_t max_num_iter = SIZE_MAX;
	IDistance *dis_metric = 0;
	ISeed_Initializer *seedInitializer = 0;
	largesttype cal_dist_num;
	
	vector<numclustype> center_1st;
		
	VectorXi ball_k_means_noRing(MatrixOur& dataset, MatrixOur& centroids) {
		
		bool judge = true;

		const size_t dataset_rows = dataset.rows();
		const size_t dataset_cols = dataset.cols();
		const size_t k = centroids.rows();

		OurType stable_field_dist = 0;

		ClusterIndexVector temp_clusters_point_index;
		ClusterIndexVector clusters_point_index;
		ClusterIndexVector clusters_neighbors_index;

		ClusterDistVector point_center_dist;

		MatrixOur new_centroids(k, dataset_cols);
		MatrixOur old_centroids = centroids;
		MatrixOur centers_dist(k, k);

		VectorXb flag(k);
		VectorXb old_flag(k);

		VectorXi labels(dataset_rows);
		VectorOur delta(k);

		vector<size_t> now_data_index;
		vector<size_t> old_now_index;

		VectorOur the_rs(k);

		size_t now_centers_rows;
		size_t num_of_neighbour;
		size_t neighbour_num;
		size_t data_num;

		//bool key = true;

		MatrixOur::Index minCol;
		new_centroids.setZero();
		this->iter_count = 0;
		num_of_neighbour = 0;
		cal_dist_num = 0;
		flag.setZero();

		//initialize clusters_point_index and point_center_dist
		initialize(dataset, centroids, labels, clusters_point_index, clusters_neighbors_index, point_center_dist);

		temp_clusters_point_index.assign(clusters_point_index.begin(), clusters_point_index.end());
		
		while (true) {
			old_flag = flag;
			//record clusters_point_index from the previous round
			clusters_point_index.assign(temp_clusters_point_index.begin(), temp_clusters_point_index.end());
			
			this->iter_count += 1;
			

			//update the matrix of centroids
			new_centroids = update_centroids(dataset, clusters_point_index, k, dataset_cols, flag, this->iter_count,
				old_centroids);


			if (iter_count == max_num_iter)
			{
				break;
			}

			if (new_centroids != old_centroids) 
			{

				//delta: distance between each center and the previous center
				delta = (((new_centroids - old_centroids).rowwise().squaredNorm())).array().sqrt();

				old_centroids = new_centroids;

				//get the radius of each centroids
				update_radius(dataset, clusters_point_index, new_centroids, point_center_dist, the_rs, flag,
					this->iter_count, cal_dist_num, k);
				//Calculate distance between centers

				cal_centers_dist(new_centroids, this->iter_count, k, the_rs, delta, centers_dist);

				flag.setZero();

				//returns the set of neighbors

				//nowball;
				size_t now_num = 0;
				for (size_t now_ball = 0; now_ball < k; now_ball++) {

					sortedNeighbors neighbors = get_sorted_neighbors_noRing(the_rs, centers_dist, now_ball, k,
						clusters_neighbors_index[now_ball]);

					now_num = point_center_dist[now_ball].size();
					if (the_rs(now_ball) == 0) continue;

					//Get the coordinates of the neighbors and neighbors of the current ball
					old_now_index.clear();
					old_now_index.assign(clusters_neighbors_index[now_ball].begin(),
						clusters_neighbors_index[now_ball].end());
					clusters_neighbors_index[now_ball].clear();
					neighbour_num = neighbors.size();
					MatrixOur now_centers(neighbour_num, dataset_cols);

					for (size_t i = 0; i < neighbour_num; i++) {
						clusters_neighbors_index[now_ball].push_back(neighbors[i].index);
						now_centers.row(i) = new_centroids.row(neighbors[i].index);

					}

					num_of_neighbour += neighbour_num;

					now_centers_rows = now_centers.rows();

					judge = true;

					if (clusters_neighbors_index[now_ball] != old_now_index)
						judge = false;
					else {
						for (size_t i = 0; i < clusters_neighbors_index[now_ball].size(); i++) {
							if (old_flag(clusters_neighbors_index[now_ball][i]) != false) {
								judge = false;
								break;
							}
						}
					}

					if (judge) {
						continue;
					}

					now_data_index.clear();

					stable_field_dist = the_rs(now_ball);

					for (size_t j = 1; j < neighbour_num; j++) {
						stable_field_dist = min(stable_field_dist,
							(OurType)centers_dist(clusters_neighbors_index[now_ball][j], now_ball) / 2);
					}

					for (size_t i = 0; i < now_num; i++) {

						if (point_center_dist[now_ball][i] > stable_field_dist) {
							now_data_index.push_back(clusters_point_index[now_ball][i]);
						}

					}


					data_num = now_data_index.size();

					if (data_num == 0) {
						continue;
					}

					MatrixOur temp_distance = cal_ring_dist_noRing(data_num, dataset_cols, dataset, now_centers,
						now_data_index);

					cal_dist_num += data_num * now_centers.rows();


					size_t new_label;
					for (size_t i = 0; i < data_num; i++) {
						temp_distance.row(i).minCoeff(&minCol);  //clusters_neighbors_index(minCol)
						new_label = clusters_neighbors_index[now_ball][minCol];
						if (labels[now_data_index[i]] != new_label) {


							flag(now_ball) = true;
							flag(new_label) = true;

							//Update localand global labels
							vector<size_t>::iterator it = (temp_clusters_point_index[labels[now_data_index[i]]]).begin();
							while ((it) != (temp_clusters_point_index[labels[now_data_index[i]]]).end()) {
								if (*it == now_data_index[i]) {
									it = (temp_clusters_point_index[labels[now_data_index[i]]]).erase(it);
									break;
								}
								else {
									++it;
								}
							}
							temp_clusters_point_index[new_label].push_back(now_data_index[i]);
							labels[now_data_index[i]] = new_label;
						}
					}

				}

			}
			else {
				break;
			}
		}
		
		return labels;
	}

	inline MatrixOur
		update_centroids(MatrixOur& dataset, ClusterIndexVector& cluster_point_index, size_t k, size_t n,
		VectorXb& flag, size_t iteration_counter, MatrixOur& old_centroids) {
			/*

			*Summary: Update the center point of each cluster

			*Parameters:

			*     dataset: dataset in eigen matrix format.*   

			*     clusters_point_index: global position of each point in the cluster.* 

			*     k: number of center points.*  

			*     dataset_cols: data set dimensions*  

			*     flag: judgment label for whether each cluster has changed.*  

			*     iteration_counter: number of iterations.*  

			*     old_centroids: center matrix of previous round.*  

			*Return : updated center matrix.

			*/

			size_t cluster_point_index_size = 0;
			size_t temp_num = 0;
			MatrixOur new_c(k, n);
			VectorOur temp_array(n);
			for (size_t i = 0; i < k; i++) {
				temp_num = 0;
				temp_array.setZero();
				cluster_point_index_size = cluster_point_index[i].size();
				if (flag(i) != 0 || iteration_counter == 1) {
					for (size_t j = 0; j < cluster_point_index_size; j++) {
						temp_array += dataset.row(cluster_point_index[i][j]);
						temp_num++;
					}
					new_c.row(i) = (temp_num != 0) ? (temp_array / temp_num) : temp_array;
				}
				else new_c.row(i) = old_centroids.row(i);
			}
			return new_c;
		}

	inline void update_radius(MatrixOur& dataset, ClusterIndexVector& cluster_point_index,
		MatrixOur& new_centroids, ClusterDistVector& temp_dis, VectorOur& the_rs,
		VectorXb& flag, size_t iteration_counter, largesttype& cal_dist_num, size_t the_rs_size) {

		/*

		*Summary: Update the radius of each cluster

		*Parameters:

		*     dataset: dataset in eigen matrix format.*   

		*     clusters_point_index: global position of each point in the cluster.* 

		*     new_centroids: updated center matrix.*  

		*     point_center_dist: distance from point in cluster to center*  

		*     the_rs: The radius of each cluster.*  

		*     flag: judgment label for whether each cluster has changed.*  

		*     iteration_counter: number of iterations.*  

		*     cal_dist_num: distance calculation times.* 

		*     the_rs_size: number of clusters.* 

		*/

		OurType temp = 0;
		size_t cluster_point_index_size = 0;
		for (size_t i = 0; i < the_rs_size; i++) {
			cluster_point_index_size = cluster_point_index[i].size();
			if (flag(i) != 0 || iteration_counter == 1) {
				the_rs(i) = 0;
				temp_dis[i].clear();
				for (size_t j = 0; j < cluster_point_index_size; j++) {
					cal_dist_num++;
					temp = sqrt((new_centroids.row(i) - dataset.row(cluster_point_index[i][j])).squaredNorm());
					temp_dis[i].push_back(temp);
					if (the_rs(i) < temp) the_rs(i) = temp;
				}
			}
		}
	};
		

	inline sortedNeighbors
		get_sorted_neighbors_noRing(VectorOur& the_rs, MatrixOur& centers_dist, size_t now_ball, size_t k,
		vector<size_t>& now_center_index) {
			/*

			*Summary: Get the sorted neighbors

			*Parameters:

			*     the_rs: the radius of each cluster.*   

			*     centers_dist: distance matrix between centers.* 

			*     now_ball: current ball label.*  

			*     k: number of center points*  

			*     now_center_index: nearest neighbor label of the current ball.*  

			*/

			VectorXi flag = VectorXi::Zero(k);
			sortedNeighbors neighbors;

			Neighbor temp;
			temp.distance = 0;
			temp.index = now_ball;
			neighbors.push_back(temp);
			flag(now_ball) = 1;


			for (size_t j = 1; j < now_center_index.size(); j++) {
				if (centers_dist(now_ball, now_center_index[j]) == 0 ||
					2 * the_rs(now_ball) - centers_dist(now_ball, now_center_index[j]) < 0) {
					flag(now_center_index[j]) = 1;
				}
				else {
					flag(now_center_index[j]) = 1;
					temp.distance = centers_dist(now_ball, now_center_index[j]);
					temp.index = now_center_index[j];
					neighbors.push_back(temp);
				}
			}


			for (size_t j = 0; j < k; j++) {
				if (flag(j) == 1) {
					continue;
				}
				if (centers_dist(now_ball, j) != 0 && 2 * the_rs(now_ball) - centers_dist(now_ball, j) >= 0) {
					temp.distance = centers_dist(now_ball, j);
					temp.index = j;
					neighbors.push_back(temp);
				}

			}


			return neighbors;
		}

	inline void
		cal_centers_dist(MatrixOur& new_centroids, size_t iteration_counter, size_t k, VectorOur& the_rs,
		VectorOur& delta, MatrixOur& centers_dis) {

			/*

			*Summary: Calculate the distance matrix between center points

			*Parameters:

			*     new_centroids: current center matrix.*   

			*     iteration_counter: number of iterations.* 

			*     k: number of center points.*  

			*     the_rs: the radius of each cluster*  

			*     delta: distance between each center and the previous center.*  

			*     centers_dist: distance matrix between centers.*  

			*/

			if (iteration_counter == 1) centers_dis = cal_dist(new_centroids, new_centroids).array().sqrt();
			else {
				for (size_t i = 0; i < k; i++) {
					for (size_t j = 0; j < k; j++) {
						if (centers_dis(i, j) >= 2 * the_rs(i) + delta(i) + delta(j))
							centers_dis(i, j) = centers_dis(i, j) - delta(i) - delta(j);
						else {
							centers_dis(i, j) = sqrt((new_centroids.row(i) - new_centroids.row(j)).squaredNorm());
						}
					}
				}
			}
		}

	inline MatrixOur cal_dist(MatrixOur& dataset, MatrixOur& centroids) {

		/*

		*Summary: Calculate distance matrix between dataset and center point

		*Parameters:

		*     dataset: dataset matrix.*   

		*     centroids: centroids matrix.* 

		*Return : distance matrix between dataset and center point.

		*/

		return (((-2 * dataset * (centroids.transpose())).colwise() + dataset.rowwise().squaredNorm()).rowwise() +
			(centroids.rowwise().squaredNorm()).transpose());
	}

	inline MatrixOur
		cal_ring_dist_noRing(size_t data_num, size_t dataset_cols, MatrixOur& dataset, MatrixOur& now_centers,
		vector<size_t>& now_data_index) {
			/*

			*Summary: Calculate the distance matrix from the point in the ring area to the corresponding nearest neighbor

			*Parameters:

			*     data_num: number of points in the ring area.*   

			*     dataset_cols: data set dimensions.* 

			*     dataset: dataset in eigen matrix format.* 

			*     now_centers: nearest ball center matrix corresponding to the current ball.* 

			*     now_data_index: labels for points in the ring.* 

			*Return : distance matrix from the point in the ring area to the corresponding nearest neighbor.

			*/

			MatrixOur data_in_area(data_num, dataset_cols);

			for (size_t i = 0; i < data_num; i++) {
				data_in_area.row(i) = dataset.row(now_data_index[i]);
			}

			return (((-2 * data_in_area * (now_centers.transpose())).colwise() +
				data_in_area.rowwise().squaredNorm()).rowwise() + (now_centers.rowwise().squaredNorm()).transpose());
		}
	
	void initialize(MatrixOur& dataset, MatrixOur& centroids, VectorXi& labels, ClusterIndexVector& cluster_point_index,
		ClusterIndexVector& clusters_neighbors_index, ClusterDistVector& temp_dis) {

		/*

		*Summary: Initialize related variables

		*Parameters:

		*     dataset: dataset in eigen matrix format.*   

		*     centroids: dcentroids matrix.* 

		*     labels: the label of the cluster where each data is located.* 

		*     clusters_point_index: two-dimensional vector of data point labels within each cluster.* 

		*     clusters_neighbors_index: two-dimensional vector of neighbor cluster labels for each cluster.* 

		*     point_center_dist: distance from point in cluster to center.* 

		*/

		MatrixOur::Index minCol;
		for (size_t i = 0; i < centroids.rows(); i++) {
			cluster_point_index.push_back(vector<size_t>());
			clusters_neighbors_index.push_back(vector<size_t>());
			temp_dis.push_back(vector<OurType>());
		}
		MatrixOur M = cal_dist(dataset, centroids);
		for (size_t i = 0; i < dataset.rows(); i++) {
			M.row(i).minCoeff(&minCol);
			labels(i) = minCol;
			cluster_point_index[minCol].push_back(i);
		}
	}
	
	inline MatrixOur initial_centroids(MatrixOur dataset, size_t k, size_t random_seed = -1) {
		int dataset_cols = dataset.cols();
		int dataset_rows = dataset.rows();

		vector<int> flag(dataset_rows, 0);
		MatrixOur centroids(k, dataset_cols);
#ifdef inst_del
		const size_t* seeds_indexes = this->seedInitializer->seeds_indexes_in_instance();
		for (numclustype ci = 0; ci < the_k; ci++)
		{
			for (dimtype di = 0; di < dataset_cols; di++)
				centroids(ci, di) = dataset(seeds_indexes[ci], di);
		}
#else
		const vector<const coordtype*>& seeds_coords = this->seedInitializer->next_seeds();
		for (numclustype ci = 0; ci < the_k; ci++)
		{
			for (dimtype di = 0; di < dataset_cols; di++)
				centroids(ci, di) = seeds_coords[ci][di];
		}
#endif
		return centroids;
	}
private:
	size_t iter_count = 0; 
public:
	virtual const vector<numclustype>& apply()
	{
		this->iter_count = 0;
		cal_dist_num = 0;

#ifdef mywarnon
		cout << "Before assigning centers..." << endl;
#endif
		MatrixOur centroids = initial_centroids(data, the_k);	
#ifdef mywarnon
		cout << "Centers are assigned..." << endl;
#endif

		VectorXi labels = ball_k_means_noRing(data, centroids);
		center_1st = vector<numclustype>(data.rows());
		for (sizetype i = 0; i < data.rows(); i++)
			center_1st[i] = labels(i);
		this->dis_metric->set_counter(cal_dist_num);
#ifdef inst_del
		data = MatrixOur(2, 2);//this is small enough
#endif
		return this->center_1st;		
	}
	virtual inline void set_instance(const IInstance_Coordinated *inst)
	{
#ifdef mywarnon
		cout << "Before setting intance..." << endl;
#endif
				
#ifdef inst_del
		cout << "data reading . . . ";
		inst->clearAll();
		data = MatrixOur(inst->size(), inst->dimension());
		string path = ((Inst_CoordpLine*)inst)->path();
		ifstream in(path);
		OurType t;
		in >> t;
		in >> t;
		for (size_t i = 0; i < inst->size(); i++)
		for (size_t j = 0; j < inst->dimension(); j++)
		{
			in >> t;
			data(i, j) = t;
		}
		cout << "ok" << endl;
#else
		data = MatrixOur(inst->size(), inst->dimension());
		for (size_t i = 0; i < inst->size(); i++)
		for (size_t j = 0; j < inst->dimension(); j++)
			data(i, j) = (*inst)[i][j];
#endif		
	}
	virtual inline void set_distance_metric(IDistance* dis_metric) { this->dis_metric = dis_metric; }
	virtual inline void set_max_num_iteration(size_t max_num_iter = SIZE_MAX){ this->max_num_iter = max_num_iter; }
	virtual inline void set_num_of_clusters(numclustype the_k){ this->the_k = the_k; }
	virtual inline void set_seeds_initializer(ISeed_Initializer *initializer){ this->seedInitializer = initializer; }

	virtual const vector<numclustype>& get_results()const { return center_1st; }
	virtual inline numclustype get_num_of_clusters()const{ return the_k; }
	virtual size_t get_consumed_iteration_count() const { return iter_count; }
	virtual const coordtype* get_starting_initial_centers(numclustype ci) const { return 0; /*starting_initial_centers[ci].data();*/ }

	virtual string name()const { if (!seedInitializer)return "ball-noRing-km"; return "ball-noRing-km-" + seedInitializer->name(); }
};

#endif
