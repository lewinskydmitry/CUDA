#include "MatrixOperations.cuh"



__device__ float GetElement(const SubMatrix A, int row, int col)
{
    return A.data[row * A.width + col];
}


__device__ void SetElement(SubMatrix A, int row, int col, double value)
{
    A.data[row * A.width + col] = value;
}


__device__ SubMatrix GetSubMatrix(SubMatrix A, int row, int col)
{
    SubMatrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.length = BLOCK_SIZE;
    Asub.data = &A.data[A.width * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}


Matrix MatMulSH(const Matrix A, const Matrix B)
{
    // Load A and B to device memory
    SubMatrix d_A;
    d_A.width = A.width; d_A.length = A.length;
    size_t size = A.width * A.length;
    cudaMalloc(&d_A.data, size * sizeof(double));
    cudaMemcpy(d_A.data, A.data, size * sizeof(double), cudaMemcpyHostToDevice);
    
    SubMatrix d_B;
    d_B.width = B.width; d_B.length = B.length;
    size = B.width * B.length;
    cudaMalloc(&d_B.data, size * sizeof(double));
    cudaMemcpy(d_B.data, B.data, size * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate C in device memory
    SubMatrix d_C;
    d_C.width = B.width; d_C.length = A.length;
    size = B.width * A.length;
    cudaMalloc(&d_C.data, size * sizeof(double));

    // Invoke kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(B.width / dimBlock.x, A.length / dimBlock.y);
    MatMulKernelSH << <dimGrid, dimBlock >> > (d_A, d_B, d_C);

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

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    return C;
}


__global__ void MatMulKernelSH(SubMatrix A, SubMatrix B, SubMatrix C)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    SubMatrix Csub = GetSubMatrix(C, blockRow, blockCol);

    double Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        SubMatrix Asub = GetSubMatrix(A, blockRow, m);
        SubMatrix Bsub = GetSubMatrix(B, m, blockCol);

        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        __syncthreads();
    }

    SetElement(Csub, row, col, Cvalue);
}