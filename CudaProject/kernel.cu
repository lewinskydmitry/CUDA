#include "../Matrix/Matrix.h"
#include "../MatrixOperations/MatrixOperations.cuh"
#include "../LinearRegression/LinearRegression.cuh"


int main() {
	// Test linear regression
	Matrix X_lin = Matrix::read_csv("C:/Users/Dmitry/source/repos/lewinskydmitry/CUDA/CudaProject/data_linear/X.csv",',');
	Matrix y_lin = Matrix::read_csv("C:/Users/Dmitry/source/repos/lewinskydmitry/CUDA/CudaProject/data_linear/y.csv", ',');
	Matrix C_lin = fit(X_lin, y_lin, 30);
	C_lin.print();

    return 0;
}