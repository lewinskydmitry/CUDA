#include "MatrixOperations.cuh"

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