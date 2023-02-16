#pragma once

#include "../Matrix/Matrix.h"
#include "../MatrixOperations/MatrixOperations.cuh"



class LogisticRegression {
public:
	// Matrix of the difference between prediction and true values
	Matrix difference;
	// Losses of loss function in each epoch
	Matrix losses;
	// Gradients
	Matrix grads;
	// Bias
	Matrix b;
	// Matrices with predicts
	Matrix predict;
	// Variables for Ridge regression
	double REG_TERM = 2;
	// Start learning rate. It will be decrease if losses will increase. It's not necessary to change it
	double h_THETA = 0.1;
	// The main function of fitting egression
	Matrix fit(Matrix X, Matrix y, int epoch);
};
