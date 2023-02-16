#include "MatrixOperations.cuh"


// Kernel code for performing matrices addition
__global__ void AddMatrixKernel(const Matrix A, const Matrix B, Matrix C)
{
    int size = A.width * A.length;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx < size) {
        C.data[thread_idx] = A.data[thread_idx] + B.data[thread_idx];
    }
}


// Host code for performing matrices addition
Matrix AddMatrix(Matrix A, Matrix B) {

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


    Matrix d_C;
    d_C.width = B.width; d_C.length = B.length;
    cudaMalloc(&d_C.data, size * sizeof(double));


    int blocksPerGrid = (d_A.width * d_A.length + threadsPerBlock - 1) / threadsPerBlock;
    AddMatrixKernel << < blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C);

    Matrix C;
    C.width = B.width; C.length = B.length;
    C.data = new double[C.width * C.length];

    cudaMemcpy(C.data, d_C.data, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A.data);
    cudaFree(d_B.data);
    cudaFree(d_C.data);

    return C;
};