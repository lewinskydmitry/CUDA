
#include "../DataLoader/DataLoader.h"
#include "../VectorSum/VectorSum.cuh"
#include "../CpuMatMul/CpuMatMul.cuh"

__host__ int fill(float** Lmatrix, float** Rmatrix, int LdimX, int LdimY, int RdimX, int RdimY) {

    int sqr_dim_X, sqr_dim_Y, size;

    sqr_dim_X = RdimX;
    if (LdimX > RdimX) {
        sqr_dim_X = LdimX;
    }

    sqr_dim_Y = RdimY;
    if (LdimY > RdimY) {
        sqr_dim_Y = LdimY;
    }

    size = sqr_dim_Y;
    if (sqr_dim_X > sqr_dim_Y) {
        size = sqr_dim_X;
    }

    int temp = size / 16 + (size % 16 == 0 ? 0 : 1);
    size = temp * 16;

    size_t pt_size = size * size * sizeof(float);

    *Lmatrix = (float*)malloc(pt_size);
    *Rmatrix = (float*)malloc(pt_size);

    memset(*Lmatrix, 0, pt_size);
    memset(*Rmatrix, 0, pt_size);

    for (int i = 0; i < LdimX; i++) {
        for (int j = 0; j < LdimY; j++) {
            int dummy = size * i + j;
            (*Lmatrix)[dummy] = sinf(dummy);
        }
    }
    for (int i = 0; i < RdimX; i++) {
        for (int j = 0; j < RdimY; j++) {
            int dummy = size * i + j;
            (*Rmatrix)[dummy] = cosf(dummy);
        }
    }
    return size;
}

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

	//Matrix summation testing
	std::cout << "Matrix C as sum of matrixes" << "\n";
	int length = data_A.length;
	int width = data_A.width;
	int array_size = length * width;
	double* c = new double[array_size];

	cudaCallAddVectorKernel(data_A.data, data_B.data, c, array_size);
	DataLoader::print_matrix(c, length, width);
    std::cout << "\n";

    double* cc = new double[array_size];
    cpu_matrix_mult(data_A.data, data_B.data, cc, length);
    DataLoader::print_matrix(cc, length, width);

    return 0;
}
