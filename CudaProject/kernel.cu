#include "../Matrix/Matrix.h"
#include "../MatrixOperations/MatrixOperations.cuh"


int main() {
	//// DataLoader testing
	//std::cout << "Matrix A" << "\n";
	//Matrix A = Matrix::read_csv("./A.csv", ';');
	//A.print();

	////std::cout << "Matrix B" << "\n";
	//Matrix B = Matrix::read_csv("./B.csv", ';');
	//B.print();
	//std::cout << "\n";

	////Matrix summation testing
	//std::cout << "Matrix sum" << std::endl;
	//Matrix sum_matrix = AddMatrix(A, B);
	//sum_matrix.print();
	//std::cout << std::endl;

	////Matrix multiplication testing
	//std::cout << "Matrix mut" << std::endl;
    //Matrix mut_matrix = cpu_matrix_mult(A, B);
	//mut_matrix.print();
	//std::cout << std::endl;

	//// Generate random matrixes
	//std::cout << "Random matrix 1" << std::endl;
	Matrix rand_matrix_1 = Matrix::create_matrix(23, 6, 0, 5);
	//rand_matrix_1.print();
	//std::cout << std::endl;

	//std::cout << "Random matrix 2" << std::endl;
	Matrix rand_matrix_2 = Matrix::create_matrix(6, 23, 0, 5);
	//rand_matrix_2.print();
	//std::cout << std::endl;


	Matrix C3 = cpu_matrix_mult(rand_matrix_1, rand_matrix_2);
	C3.print();
	std::cout << std::endl;
	std::cout << std::endl;


	Matrix C1 = MatMul(rand_matrix_1, rand_matrix_2);
	C1.print();
	std::cout << std::endl;
	std::cout << std::endl;


	Matrix C2 = MatMulSH(rand_matrix_1, rand_matrix_2);
	C2.print();
	std::cout << std::endl;
	std::cout << std::endl;

    return 0;
}