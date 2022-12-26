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




// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, double* C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    double Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.data[row * A.width + e] * B.data[e * B.width + col];
    C[row * B.width + col] = Cvalue;
}

Matrix MatMul(Matrix A, Matrix B)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.length = A.length;
    size_t size = A.width * A.length;
    cudaMalloc(&d_A.data, size * sizeof(double));
    cudaMemcpy(d_A.data, A.data, size * sizeof(double), cudaMemcpyHostToDevice);
    
    Matrix d_B;
    d_B.width = B.width; d_B.length = B.length;
    size = B.width * B.length;
    cudaMalloc(&d_B.data, size * sizeof(double));
    cudaMemcpy(d_B.data, B.data, size * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate C in device memory
    double* dev_c = 0;
    cudaMalloc((void**)&dev_c, A.length * B.width * sizeof(double));

    // Invoke kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(B.width / dimBlock.x + 1, A.length / dimBlock.y + 1);
    MatMulKernel << <dimGrid, dimBlock >> > (d_A, d_B, dev_c);

    Matrix C;
    C.length = A.length;
    C.width = B.width;
    size = A.length * B.width;
    C.data = new double[size];
    // Read C from device memory
    cudaMemcpy(C.data, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A.data);
    cudaFree(d_B.data);
    cudaFree(dev_c);
    return C;
}

