#include "MatrixOperations.cuh"

// THIS FILE CONTAINS NAIVE MATRIX MULTIPLICATION

// Kernel for performing matrices multiplication
__global__ void MatMulNaiveKernel(Matrix A, Matrix B, Matrix C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < A.length && col < B.width) {
        double Cvalue = 0;
        for (int e = 0; e < A.width; ++e)
            Cvalue += A.data[row * A.width + e] * B.data[e * B.width + col];
        C.data[row * B.width + col] = Cvalue;
    }
}


// Hose code for performing matrices multiplication
Matrix MatMulNaive(Matrix A, Matrix B)
{
    // This code for catching errors if dimensions of matrixes don't match
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

    Matrix d_C;
    d_C.length = A.length;
    d_C.width = B.width;
    size = A.length * B.width;
    cudaMalloc(&d_C.data, size * sizeof(double));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, (A.length + dimBlock.y - 1) / dimBlock.y);
    MatMulNaiveKernel << <dimGrid, dimBlock >> > (d_A, d_B, d_C);

    Matrix C;
    C.length = A.length;
    C.width = B.width;
    size = A.length * B.width;
    C.data = new double[size];


    cudaMemcpy(C.data, d_C.data, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A.data);
    cudaFree(d_B.data);
    cudaFree(d_C.data);

    return C;
}