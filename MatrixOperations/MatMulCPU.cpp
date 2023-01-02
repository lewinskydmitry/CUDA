#include "MatrixOperations.cuh"

Matrix cpu_matrix_mult(Matrix A, Matrix B) {
    if (A.width != B.length) {
        try {
            throw std::invalid_argument("Dimensions do not match");
        }
        catch (const std::invalid_argument& e) {
            std::cout << "Matrix multiplication error:" << "\n";
            std::cout << e.what() << std::endl;
            exit(1);
        }
    }

    Matrix C;
    C.length = A.length;
    C.width = B.width;
    C.data = new double[C.length * C.width];

    for (int row_a = 0; row_a < A.length; row_a++) {
        for (int col_b = 0; col_b < B.width; col_b++) {

            double temp = 0;
            for (int i = 0; i < A.width; i++) {
                temp += A.data[row_a * A.width + i] * B.data[i * B.width + col_b];
            }
            C.data[row_a * B.width + col_b] = temp;
        }
    }
    return C;
}