#include "../Matrix/Matrix.h"
#include "../MatrixOperations/MatrixOperations.cuh"


int main() {
	Matrix rand_matrix_1 = Matrix::create_matrix(3, 4, 0, 5);
	Matrix rand_matrix_2 = Matrix::create_matrix(3, 3, 0, 5);

	Matrix::equal(rand_matrix_1, rand_matrix_2);
    return 0;
}