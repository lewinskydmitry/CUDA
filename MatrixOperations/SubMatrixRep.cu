#include "MatrixOperations.cuh"


// Kernel for performing matrices substraction with replacement
__global__ void SubMatrixRepKernel(Matrix& A, Matrix B)
{
    int size = A.width * A.length;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx < size) {
        A.data[thread_idx] -= B.data[thread_idx];
    }
}


// Host code for performing matrices substraction with replacement
void SubMatrixRep(Matrix& A, Matrix B) {

    // This code for catching errors if dimensions of matrices don't match
    if (A.length != B.length && A.width != B.width) {
        try {
            throw std::invalid_argument("Dimensions do not match");
        }
        catch (const std::invalid_argument& e) {
            std::cout << "Matrix addition error:" << "\n";
            std::cout << e.what() << std::endl;
            exit(1);
        }
    }

    Matrix d_A;
    d_A.width = A.width; d_A.length = A.length;
    int size = A.width * A.length;
    cudaMalloc(&d_A.data, size * sizeof(double));
    cudaMemcpy(d_A.data, A.data, size * sizeof(double), cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = B.width; d_B.length = B.length;
    cudaMalloc(&d_B.data, size * sizeof(double));
    cudaMemcpy(d_B.data, B.data, size * sizeof(double), cudaMemcpyHostToDevice);

    int blocksPerGrid = (d_A.width * d_A.length + threadsPerBlock - 1) / threadsPerBlock;
    SubMatrixRepKernel << < blocksPerGrid, threadsPerBlock >> > (d_A, d_B);

    cudaMemcpy(A.data, d_A.data, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A.data);
    cudaFree(d_B.data);
};