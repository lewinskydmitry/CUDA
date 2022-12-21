#include "CpuMatMul.cuh"

void cpu_matrix_mult(double* h_a, double* h_b, double* h_result, int m) {
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            double tmp = 0.0;
            for (int h = 0; h < m; ++h)
            {
                tmp += h_a[i * m + h] * h_b[h * m + j];
            }
            h_result[i * m + j] = tmp;
        }
    }
}