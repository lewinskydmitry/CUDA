#pragma once

#include "../Matrix/Matrix.h"
#include "../MatrixOperations/MatrixOperations.cuh"



class LinearRegression {
public:
	Matrix difference;
	Matrix losses;
	Matrix grads;
	Matrix b;
	Matrix predict;
	double REG_TERM = 2;
	double h_THETA = 0.1;
	Matrix fit(Matrix X, Matrix y, int epoch);
};
