

#include "../Matrix/Matrix.h"
#include "../VectorSum/VectorSum.cuh"
#include "../CpuMatMul/CpuMatMul.cuh"


int main() {
	// DataLoader testing
	std::cout << "Matrix A" << "\n";
	Matrix A = Matrix::read_csv("./A.csv", ';');
	A.print();
	std::cout << "\n";

	std::cout << "Matrix B" << "\n";
	Matrix B = Matrix::read_csv("./B.csv", ';');
	B.print();
	std::cout << "\n";

	//Matrix summation testing
	std::cout << "Matrix sum" << std::endl;
	Matrix sum_matrix = AddVector(A, B);
	sum_matrix.print();
	std::cout << std::endl;

	//Matrix multiplication testing
	std::cout << "Matrix mut" << std::endl;
    Matrix mut_matrix = cpu_matrix_mult(A, B);
	mut_matrix.print();
	std::cout << std::endl;

	// Generate random matrixes
	std::cout << "Random matrix 1" << std::endl;
	Matrix rand_matrix_1 = Matrix::create_matrix(2, 5, 0, 5);
	rand_matrix_1.print();
	std::cout << std::endl;

	std::cout << "Random matrix 2" << std::endl;
	Matrix rand_matrix_2 = Matrix::create_matrix(5, 2, 0, 5);
	rand_matrix_2.print();
	std::cout << std::endl;

	// Multipliply random matrixes
	Matrix C2 = cpu_matrix_mult(rand_matrix_1, rand_matrix_2);
	C2.print();

    return 0;
}
