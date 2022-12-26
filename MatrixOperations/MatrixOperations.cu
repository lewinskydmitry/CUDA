#include "MatrixOperations.cuh"

__global__
void cudaAddMatrixKernel(const double* a,
    const double* b,
    double* c,
    const int size) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (thread_idx < size) {
        c[thread_idx] = a[thread_idx] + b[thread_idx];
        thread_idx += blockDim.x * gridDim.x;
    }
}

void cudaCallAddMatrixKernel(const double* a,
    const double* b,
    double* c,
    const int size) {

    double* dev_a = 0;
    double* dev_b = 0;
    double* dev_c = 0;
    int per_block_thread_count = 1024;

    int block_count = (int)ceil(size / (float)per_block_thread_count);


    cudaMalloc((void**)&dev_c, size * sizeof(double));
    cudaMalloc((void**)&dev_a, size * sizeof(double));
    cudaMalloc((void**)&dev_b, size * sizeof(double));


    cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);

    cudaAddMatrixKernel << < block_count, per_block_thread_count >> > (dev_a, dev_b, dev_c, size);

    cudaDeviceSynchronize();

    cudaMemcpy(c, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);


    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
};

Matrix AddMatrix(Matrix a, Matrix b) {
    int size;
    Matrix c;
    double* dev_a = 0;
    double* dev_b = 0;
    double* dev_c = 0;
    int per_block_thread_count = 1024;

    if (a.length != b.length && a.width != b.width) {
        throw std::invalid_argument("Shapes aren't matching");
    }
    else {
        size = a.length * a.width;
        c.length = a.length;
        c.width = a.width;
        c.data = new double[size];
    };

    int block_count = (int)ceil(size / (float)per_block_thread_count);


    cudaMalloc((void**)&dev_c, size * sizeof(double));
    cudaMalloc((void**)&dev_a, size * sizeof(double));
    cudaMalloc((void**)&dev_b, size * sizeof(double));


    cudaMemcpy(dev_a, a.data, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data, size * sizeof(double), cudaMemcpyHostToDevice);

    cudaAddMatrixKernel << < block_count, per_block_thread_count >> > (dev_a, dev_b, dev_c, size);

    cudaDeviceSynchronize();

    cudaMemcpy(c.data, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);


    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    return c;
};

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