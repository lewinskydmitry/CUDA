#include "../Matrix/Matrix.h"
#include "../MatrixOperations/MatrixOperations.cuh"
#include "../LinearRegression/LinearRegression.cuh"

void test(int value) {
	Matrix X_lin = Matrix::read_csv("C:/Users/Dmitry/source/repos/lewinskydmitry/CUDA/CudaProject/data/real_estate_X.csv", ',');
	Matrix y_lin = Matrix::read_csv("C:/Users/Dmitry/source/repos/lewinskydmitry/CUDA/CudaProject/data/real_estate_y.csv", ',');


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


int main() {

	test(1);
	

    return 0;
}
