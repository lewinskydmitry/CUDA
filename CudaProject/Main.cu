#include "../Matrix/Matrix.h"
#include "../MatrixOperations/MatrixOperations.cuh"
#include "../LinearRegression/LinearRegression.cuh"


// Function for testing Linear regression
// if value = 1 we can check all logs (weights, bias and other computations on each step)
// if value = 0 (or any other value) we will check losses on each step
void test(int value) {
	// Read CSV data
	Matrix X_lin = Matrix::read_csv("C:/Users/Dmitry/source/repos/lewinskydmitry/CUDA/CudaProject/data/real_estate_X.csv", ',');
	Matrix y_lin = Matrix::read_csv("C:/Users/Dmitry/source/repos/lewinskydmitry/CUDA/CudaProject/data/real_estate_y.csv", ',');

	// Here we check logs or losses. This if depends on our "value"
	if (value == 1) {
		for (int i = 0; i < 5; i += 1) {
			LinearRegression reg;
			Matrix C_lin = reg.fit(X_lin, y_lin, i);

			std::cout << "\n stage:" << i << "\n weigths : \n";
			reg.grads.print();

			std::cout << "\n bias:" << reg.b.data[0] << "\n";

			std::cout << "\n predict: \n";
			reg.predict.print();

			std::cout << "\n difference: \n";
			C_lin.print();
			std::cout << "\n";

		}
	}
	else
	{
		LinearRegression reg;
		Matrix C_lin = reg.fit(X_lin, y_lin, 500);
		reg.losses.print();
	}
}

// The main function for testing everything
int main() {
	// Example of using testing function for checking logs
	test(1);
    return 0;
}
