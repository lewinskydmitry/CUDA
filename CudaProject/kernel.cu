#include "../Matrix/Matrix.h"
#include "../MatrixOperations/MatrixOperations.cuh"


int main() {
	Matrix rand_matrix_1 = Matrix::create_matrix(2, 4, 0, 5);
	Matrix rand_matrix_2 = Matrix::create_matrix(3, 3, 0, 5);

	rand_matrix_1.print();
	std::cout << "\n";

	Matrix A = MatTranspose(rand_matrix_1);
	A.print();


	Matrix::equal(A, Matrix::transpose(rand_matrix_1));
    return 0;
}