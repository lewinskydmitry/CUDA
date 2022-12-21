
#include "../DataLoader/DataLoader.h"
#include "../VectorSum/VectorSum.cuh"

int main() {
	// DataLoader testing
	std::cout << "Matrix A" << "\n";
	DataLoader data_A("./A.csv", ';');
	data_A.print();
	std::cout << "\n";

	std::cout << "Matrix B" << "\n";
	DataLoader data_B("./B.csv", ';');
	data_B.print();
	std::cout << "\n";

	std::cout << "Matrix D" << "\n";
	DataLoader data_D("./D.csv", ';');
	data_D.print();
	std::cout << "\n";

	//Matrix summation
	std::cout << "Matrix C as sum of matrixes" << "\n";
	int length = data_A.length;
	int width = data_A.width;
	int array_size = length * width;
	double* c = new double[array_size];

	cudaCallAddVectorKernel(data_A.data, data_B.data, c, array_size);
	DataLoader::print_matrix(c, length, width);

	return 0;
}
