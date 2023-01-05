#include "MatrixOperations.cuh"


__global__
void cudaAddMatrixKernel(const Matrix A, const Matrix B, Matrix C) 
{
    int size = A.width * A.length;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (thread_idx < size) {
        C.data[thread_idx] = A.data[thread_idx] + B.data[thread_idx];
        thread_idx += blockDim.x * gridDim.x;
    }
}


Matrix AddMatrix(Matrix A, Matrix B) {

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


    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((A.width + dimBlock.x - 1) / dimBlock.x, (A.length + dimBlock.y - 1) / dimBlock.y);
    
    cudaAddMatrixKernel << < dimGrid, dimBlock >> > (d_A, d_B, d_C);

    Matrix C;
    C.width = B.width; C.length = B.length;
    C.data = new double[C.width * C.length];

    cudaMemcpy(C.data, d_C.data, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A.data);
    cudaFree(d_B.data);
    cudaFree(d_C.data);

    return C;
};