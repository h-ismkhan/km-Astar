#ifndef _ball_km_defs_H
#define _ball_km_defs_H


#include <Eigen/Dense>
using namespace Eigen;


typedef distype OurType;
#define MaxOfOurType dismax
typedef vector<vector<OurType>> ClusterDistVector;

#ifdef coords_are_double
typedef VectorXd VectorOur;
typedef MatrixXd MatrixOur;
#else
typedef VectorXf VectorOur;
typedef MatrixXf MatrixOur;
#endif

typedef vector<vector<size_t>> ClusterIndexVector;

typedef Array<bool, 1, Dynamic> VectorXb;

typedef struct Neighbor
{
	OurType distance;
	size_t index;
};

typedef vector<Neighbor> sortedNeighbors;

bool LessSort(Neighbor a, Neighbor b) {
	return (a.distance < b.distance);
}

#endif
