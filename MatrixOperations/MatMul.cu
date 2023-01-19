#include "MatrixOperations.cuh"

__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.data[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col, double value)
{
    A.data[row * A.stride + col] = value;
}

__device__ Matrix GetMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.length = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.data = &A.data[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}



__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix Csub = GetMatrix(C, blockRow, blockCol);

    double Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < ((A.width + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m) {

        Matrix Asub = GetMatrix(A, blockRow, m);
        Matrix Bsub = GetMatrix(B, m, blockCol);

        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        if (m * BLOCK_SIZE + threadIdx.x < A.width && blockRow * BLOCK_SIZE + threadIdx.y < A.length) {
            As[row][col] = GetElement(Asub, row, col);
        }
        else
        {
            As[row][col] = 0;
        }

        if (m * BLOCK_SIZE + threadIdx.y < B.length && blockCol * BLOCK_SIZE + threadIdx.x < B.width) {
            Bs[row][col] = GetElement(Bsub, row, col);
        }
        else
        {
            Bs[row][col] = 0;
        }

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e) {
            Cvalue += As[row][e] * Bs[e][col];
        }
        __syncthreads();
    }

    if (blockIdx.y * BLOCK_SIZE + threadIdx.y < C.length && blockIdx.x * BLOCK_SIZE + threadIdx.x < C.width) {
        SetElement(Csub, row, col, Cvalue);
    }
}



Matrix MatMul(const Matrix A, const Matrix B)
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

    Matrix d_A;
    d_A.width = A.width; d_A.length = A.length, d_A.stride = A.width;
    size_t size = A.width * A.length;
    cudaMalloc(&d_A.data, size * sizeof(double));
    cudaMemcpy(d_A.data, A.data, size * sizeof(double), cudaMemcpyHostToDevice);
    
    Matrix d_B;
    d_B.width = B.width; d_B.length = B.length, d_B.stride = B.width;
    size = B.width * B.length;
    cudaMalloc(&d_B.data, size * sizeof(double));
    cudaMemcpy(d_B.data, B.data, size * sizeof(double), cudaMemcpyHostToDevice);

    Matrix d_C;
    d_C.width = B.width; d_C.length = A.length, d_C.stride = B.width;
    size = B.width * A.length;
    cudaMalloc(&d_C.data, size * sizeof(double));

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, (A.length + dimBlock.y - 1) / dimBlock.y);
    MatMulKernel <<< dimGrid, dimBlock >>> (d_A, d_B, d_C);

    Matrix C;
    C.length = A.length; C.width = B.width;
    size = A.length * B.width;
    C.data = new double[size];

    cudaMemcpy(C.data, d_C.data, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A.data);
    cudaFree(d_B.data);
    cudaFree(d_C.data);

    return C;
}
