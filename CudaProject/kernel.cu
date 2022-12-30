

#include "../Matrix/Matrix.h"
#include "../MatrixOperations/MatrixOperations.cuh"
void set(Matrix A)
{
	A.data[0] = 5;
};

int main() {
	//// DataLoader testing
	//std::cout << "Matrix A" << "\n";
	//Matrix A = Matrix::read_csv("./A.csv", ';');
	//A.print();
	//std::cout << "\n";

	//std::cout << "Matrix B" << "\n";
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
 //   Matrix mut_matrix = cpu_matrix_mult(A, B);
	//mut_matrix.print();
	//std::cout << std::endl;

	//// Generate random matrixes
	//std::cout << "Random matrix 1" << std::endl;
	Matrix rand_matrix_1 = Matrix::create_matrix(16, 16, 0, 5);
	//rand_matrix_1.print();
	//std::cout << std::endl;

	//std::cout << "Random matrix 2" << std::endl;
	Matrix rand_matrix_2 = Matrix::create_matrix(16, 16, 0, 5);
	//rand_matrix_2.print();
	//std::cout << std::endl;

	//// Multipliply random matrixes
	Matrix C2 = cpu_matrix_mult(rand_matrix_1, rand_matrix_2);
	C2.print();
	std::cout << std::endl;
	std::cout << std::endl;

	SubMatrix A;
	A.length = rand_matrix_1.length;
	A.width = rand_matrix_1.width;
	A.data = rand_matrix_1.data;

	SubMatrix B;
	B.length = rand_matrix_2.length;
	B.width = rand_matrix_2.width;
	B.data = rand_matrix_2.data;


	Matrix C = MatMulSH(A, B);
	C.print();
    return 0;
}