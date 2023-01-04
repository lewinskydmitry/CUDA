#include "MatrixOperations.cuh"

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    double Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.data[row * A.width + e] * B.data[e * B.width + col];
    C.data[row * B.width + col] = Cvalue;
}

Matrix MatMul(Matrix A, Matrix B)
{

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
    Matrix d_C;
    d_C.length = A.length;
    d_C.width = B.width;
    size = A.length * B.width;
    cudaMalloc(&d_C.data, size * sizeof(double));

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, (A.length + dimBlock.y - 1) / dimBlock.y);
    MatMulKernel << <dimGrid, dimBlock >> > (d_A, d_B, d_C);

    Matrix C;
    C.length = A.length;
    C.width = B.width;
    size = A.length * B.width;
    C.data = new double[size];

    // Read C from device memory
    cudaMemcpy(C.data, d_C.data, size * sizeof(double), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A.data);
    cudaFree(d_B.data);
    cudaFree(d_C.data);
    return C;
}