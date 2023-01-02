#include "MatrixOperations.cuh"


__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.data[row * A.width + col];
}


__device__ void SetElement(Matrix A, int row, int col, double value)
{
    A.data[row * A.width + col] = value;
}


// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.length = BLOCK_SIZE;
    Asub.data = &A.data[A.width * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}


__global__ void MatMulKernelSH(Matrix A, Matrix B, Matrix C)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix Csub = GetMatrix(C, blockRow, blockCol);

    double Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        Matrix Asub = GetMatrix(A, blockRow, m);
        Matrix Bsub = GetMatrix(B, m, blockCol);

        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e) {
            Cvalue += As[row][e] * Bs[e][col];
        }
        __syncthreads();
    }

    SetElement(Csub, row, col, Cvalue);
}


Matrix MatMulSH(const Matrix A, const Matrix B)
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
    d_C.width = B.width; d_C.length = A.length;
    size = B.width * A.length;
    cudaMalloc(&d_C.data, size * sizeof(double));

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.length / dimBlock.y);
    MatMulKernelSH <<< dimGrid, dimBlock >>> (d_A, d_B, d_C);

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
