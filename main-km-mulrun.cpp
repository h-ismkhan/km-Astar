#include<iostream>
using namespace std;
#include<numeric>

#include "headers/Instance/Inst-CoordpLine.h"
#include "headers/Distance/Euclidean-Distance.h"
#include "headers/Clustering/k-means.h"
#include "headers/Clustering/Elkan.h"
#include "headers/Clustering/k-means-astar.h"
#include "headers/Clustering/ball-km-ring.h"
#include "headers/Clustering/ball-km-noRing.h"
#include "headers/Clustering/exp-ball.h"

#include "headers/Clustering/k-means-initializers/Seeds-Random-Initializer.h"
#include "headers/Clustering/k-means-initializers/Seeds-ReadFile-Initializer.h"
#include "headers/Clustering/evaluation/SSEDM.h"
#include "headers/Utils/myTimer.h"
#include "headers/Utils/myString.h"


void create_init_C(string path, numclustype the_k, size_t numrun);

int main(int argc, char** argv)
{
	//graphicfunc();
	cout << "Hello,... This main should be changed in minor, to applied to Inst with correct cid...!" << endl;
	cout << "You can end your path file with '-'!" << endl;
	string path_txt;
	size_t maxrun = 0;
	if (argc > 1)
	{
		cout << "your max run: " << argv[1] << endl;
		maxrun = stoi(argv[1]);
	}

	path_txt = "path.txt";

	int sysresult = system("mkdir _initial-centers");
	if (sysresult < 0) cout << "";
	cout << "Warning! Only one instance is red from path.txt!" << endl;

	ifstream in(path_txt);
	vector<string> pathes, abs_names;
	vector<numclustype> the_ks;
	vector<bool> has_actual_cid;
	while (!in.eof())
	{
		string pathinst = "";
		bool hasActualClusters = false;
		largesttype theK_ll;
		numclustype theK;

		in >> pathinst;
		if (pathinst.size() < 3)
			break;
		in >> theK_ll;

		if (theK_ll <= 0)
		{
			theK = (numclustype)-1;
			hasActualClusters = true;
		}
		else theK = (numclustype)theK_ll;
		if (theK_ll > 0)
			hasActualClusters = false;
		pathes.push_back(pathinst);
		abs_names.push_back(remove_extension(extract_name(pathinst)));
		the_ks.push_back(theK);
		has_actual_cid.push_back(hasActualClusters);
	}
	in.close();

	for (size_t i = 0; i < pathes.size(); i++)
	{
		string pathinst = pathes[i];
		pair<sizetype, dimtype> size_dim = IInstance_Coordinated::read_size_dim(pathinst);
		cout << i << ": " << abs_names[i] << ", #point: " << size_dim.first
			<< ", dim: " << size_dim.second << ", k: " << the_ks[i] << endl;
	}

	if (maxrun == 0)
	{
		cout << "Enter maxrun:";
		cin >> maxrun;
	}

	vector<string> alg_names_should_beIn;
	string algFilePath = "algorithms.txt";
	ifstream in2(algFilePath);
	while (!in2.eof())
	{
		string algname = "";
		in2 >> algname;
		if (algname.size() < 1)
			break;
		if (algname.size() == 1 && algname[0] == '-')
			break;
		alg_names_should_beIn.push_back(algname);
	}
	in2.close();

	vector<k_means_interface*> all_alg_list, alg_list;
	Elkan elkan;
	all_alg_list.push_back(&elkan);
	
	ball_ring_k_means_orig ball_ring_km;
	all_alg_list.push_back(&ball_ring_km);
	ball_noRing_k_means_orig ball_noRing_km;
	all_alg_list.push_back(&ball_noRing_km);
	exp_ball_k_means exp_of_ball;
	all_alg_list.push_back(&exp_of_ball);	
	k_means km;
	all_alg_list.push_back(&km);
	k_means_astar km_astar1;
	all_alg_list.push_back(&km_astar1);

	for (size_t i = 0; i < alg_names_should_beIn.size(); i++)
	{
		for (size_t ai = 0; ai < all_alg_list.size(); ai++)
		{
			if (all_alg_list[ai]->name().find(alg_names_should_beIn[i]) != std::string::npos)
			{
				alg_list.push_back(all_alg_list[ai]);
				break;
			}
		}
	}
	if (alg_list.size() == 0)
		alg_list = all_alg_list;

	cout << "list of algorithms: " << endl;
	for (size_t ai = 0; ai < alg_list.size(); ai++)
	{
		cout << ai << ": " << alg_list[ai]->name() << endl;
	}

	//check wether initial-Centers do exsist, and of no then create init-C files.
	for (size_t pi = 0; pi < pathes.size(); pi++)
	{
		string centers_path = Seeds_ReadFile_Init::get_standard_path_initC(pathes[pi], the_ks[pi], maxrun);

		Seeds_ReadFile_Init seedfile;
		ISeed_Initializer *myseed_1, *myseed_2;

		bool initC_exsist = seedfile.load_ifcan_path_run_thek(pathes[pi], maxrun, the_ks[pi]);
		if (initC_exsist)
		{
			myseed_1 = myseed_2 = &seedfile;
		}
		else
		{
			cout << "no seed file for " << pathes[pi] << "...! creating..." << endl;
			create_init_C(pathes[pi], the_ks[pi], maxrun);
		}
	}
	/*size_t numitermax = 0;
	cout << "max iter: ";
	cin >> numitermax;*/
	for (size_t ai = 0; ai < alg_list.size(); ai++)
	{
		k_means_interface *km_interface = alg_list[ai];

		string resultpath = "_results-" + km_interface->name() + folder_sperator;
		cout << "resultpath: " << resultpath << endl;
		string command_resultpath_create = command_dir_creator + resultpath;
		sysresult = system(command_resultpath_create.c_str());

		//clear file:
		{
			ofstream ssedmfile(resultpath + "_SSEDM.txt");
			ofstream timefile(resultpath + "_time.txt");
			ofstream disnumfile(resultpath + "_#dis.txt");
			ofstream iterscountfile(resultpath + "_#iters.txt");
			ofstream mycommentsfile(resultpath + "_comments.txt");

			ssedmfile.close();
			timefile.close();
			disnumfile.close();
			iterscountfile.close();
			mycommentsfile.close();
		}

		Inst_CoordpLine inst(pathes[0]);
		for (size_t pi = 0; pi < pathes.size(); pi++)
		{

			if (pi > 0 && pathes[pi] != pathes[pi - 1])
				inst = Inst_CoordpLine(pathes[pi]);

			string centers_path = Seeds_ReadFile_Init::get_standard_path_initC(pathes[pi], the_ks[pi], maxrun);
			cout << "centers path: " << centers_path << endl;

			Seeds_ReadFile_Init seedfile;
			Seeds_Random_Init seedrand;
			ISeed_Initializer* myseed;

			bool initC_exsist = seedfile.load_ifcan_path_run_thek(pathes[pi], maxrun, the_ks[pi]);
			if (initC_exsist)
				myseed = &seedfile;
			else myseed = &seedrand;
			myseed->set_instance(&inst);
			myseed->set_the_k(the_ks[pi]);

			vector<sizetype> initC_vec;

			vector<vector<string>> mycomments_vec;
			vector<distype> SSEDMvec;
			vector<double> time_vec;
			vector<largesttype_un> numdis_vec;
			vector<size_t> iterscountvec;
			for (unsigned int run = 0; run < maxrun; run++)
			{
				Euclidean_Distance dis_metric(inst.dimension());
				//km_interface->set_max_num_iteration(numitermax);
				km_interface->set_instance(&inst);
				km_interface->set_distance_metric(&dis_metric);
				km_interface->set_num_of_clusters(the_ks[pi]);
				km_interface->set_seeds_initializer(myseed);

				MyChronometer timer;
				timer.start();
				const vector<numclustype>& result = km_interface->apply();
				timer.end();

				/*const sizetype* cur_initC = myseed->seeds_indexes_in_instance();
				for (numclustype ci = 0; ci < the_ks[pi]; ci++)
				initC_vec.push_back(cur_initC[ci]);*/

				iterscountvec.push_back(km_interface->get_consumed_iteration_count());

				Euclidean_Distance dis_metric_forSSEDM(inst.dimension());
				SSEDM_Measure ssedm;
				ssedm.set_instance(&inst);
				ssedm.set_clusterIds(result.data());
				ssedm.set_distance_metric(&dis_metric_forSSEDM);
				distype ssedmval = ssedm.measure();

				SSEDMvec.push_back(ssedmval);
				time_vec.push_back(timer.duration_recent());
				numdis_vec.push_back(dis_metric.counter());

				mycomments_vec.push_back(km_interface->my_comments());

				double sumTTime = accumulate(time_vec.begin(), time_vec.end(), 0.0);
				distype sumSSEDM = accumulate(SSEDMvec.begin(), SSEDMvec.end(), 0.0);
				largesttype_un numdis = accumulate(numdis_vec.begin(), numdis_vec.end(), (largesttype_un)0);
				size_t numitrs = accumulate(iterscountvec.begin(), iterscountvec.end(), (size_t)0);

				if (run == 0)
					cout << km_interface->name() << ", " << abs_names[pi] << ", " << endl;
				cout << run << ", k:" << the_ks[pi] << ", #dis: " << numdis / (run + 1);
				cout << ", #itr: " << numitrs / (run + 1) << ", ";
				cout << round(sumTTime / (run + 1)) << ", " << sumSSEDM / (run + 1) << endl;

				cout << "SSEDM      :" << SSEDMvec[SSEDMvec.size() - 1] << endl;
				cout << "#dist      :" << numdis_vec[numdis_vec.size() - 1] << endl;
				cout << "#itrs      :" << iterscountvec[iterscountvec.size() - 1] << endl;
				cout << "Total Time :" << time_vec[time_vec.size() - 1] << endl;
			}

			ofstream ssedmfile(resultpath + "_SSEDM.txt", std::ios_base::app);
			ofstream timefile(resultpath + "_time.txt", std::ios_base::app);
			ofstream disnumfile(resultpath + "_#dis.txt", std::ios_base::app);
			ofstream iterscountfile(resultpath + "_#iters.txt", std::ios_base::app);
			ofstream mycommentsfile(resultpath + "_comments.txt", std::ios_base::app);

			/*if (!initC_exsist)
			{
			ofstream out(Seeds_ReadFile_Init::get_standard_path_initC(pathes[pi], the_ks[pi], maxrun));
			for (size_t ci = 0; ci < the_ks[pi] * maxrun - 1; ci++)
			out << initC_vec[ci] << endl;
			out << initC_vec[the_ks[pi] * maxrun - 1];
			out.close();
			initC_exsist = true;
			}*/

			for (size_t runIndex = 0; runIndex < maxrun; runIndex++)
			{
				ssedmfile << fixed << SSEDMvec[runIndex] << endl;
				timefile << fixed << time_vec[runIndex] << endl;
				disnumfile << fixed << numdis_vec[runIndex] << endl;
				iterscountfile << fixed << iterscountvec[runIndex] << endl;
				for (size_t mci = 0; mci < mycomments_vec[runIndex].size(); mci++)
					mycommentsfile << mycomments_vec[runIndex][mci] << endl;
			}


			SSEDMvec.clear();
			time_vec.clear();
			iterscountvec.clear();
			mycomments_vec.clear();

			ssedmfile.close();
			timefile.close();
			iterscountfile.close();
			mycommentsfile.close();
		}
	}
	//getch();

	return 0;
}

void create_init_C(string path, numclustype the_k, size_t numrun)
{
	pair<sizetype, dimtype> size_dim = Inst_CoordpLine::read_size_dim(path);
	sizetype size = size_dim.first;
	dimtype dim = size_dim.second;

	cout << path << endl;
	cout << "Size: " << size << ",	Dim: " << dim << ",	k: " << the_k << endl << endl;

	Seeds_Random_Init randInit(size);
	randInit.set_the_k(the_k);
	vector<sizetype> initial_points;
	for (size_t ri = 0; ri < numrun; ri++)
	{
		randInit.next_seeds();
		const sizetype* cur_init_points = randInit.seeds_indexes_in_instance();
		for (numclustype ci = 0; ci < the_k; ci++)
			initial_points.push_back(cur_init_points[ci]);
	}

	string name = extract_name_minusExtension(path);
	string init_path1 = "_initial-centers" /*+ folder_sperator*/;
	string init_path2 = init_path1 + folder_sperator;
	string init_path = init_path2 + "initC." + to_string(numrun) + "." + to_string(the_k) + "." + name + ".txt";

	ofstream out(init_path);
	for (size_t pi = 0; pi < initial_points.size(); pi++)
		out << initial_points[pi] << endl;
	out.close();
}