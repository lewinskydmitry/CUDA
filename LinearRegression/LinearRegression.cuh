#pragma once

#include "../Matrix/Matrix.h"
#include "../MatrixOperations/MatrixOperations.cuh"



class LinearRegression {
public:
	Matrix result;
	Matrix losses;
	Matrix grads;
	Matrix b;
	Matrix gradW;
	Matrix predict;
	double h_THETA = 0.1;
	Matrix fit(Matrix X, Matrix y, int epoch);
};
