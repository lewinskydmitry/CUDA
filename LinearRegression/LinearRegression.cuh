#pragma once

#include "../Matrix/Matrix.h"
#include "../MatrixOperations/MatrixOperations.cuh"





class LinearRegression {
public:
	Matrix losses;
	float THETA = 0.00001;
	Matrix fit(Matrix X, Matrix y, int epoch);
};

