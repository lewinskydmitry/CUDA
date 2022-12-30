#include "MatrixOperations.cuh"

Matrix cpu_matrix_mult(Matrix a, Matrix b) {
    if (a.width != b.length) {
        std::cout << "Shapes are not matching";
    }

    Matrix c;
    c.length = a.length;
    c.width = b.width;
    c.data = new double[c.length * c.width];

    for (int row_a = 0; row_a < a.length; row_a++) {
        for (int col_b = 0; col_b < b.width; col_b++) {

            double temp = 0;
            for (int i = 0; i < a.width; i++) {
                temp += a.data[row_a * a.width + i] * b.data[i * b.width + col_b];
            }
            c.data[row_a * b.width + col_b] = temp;
        }
    }
    return c;
}