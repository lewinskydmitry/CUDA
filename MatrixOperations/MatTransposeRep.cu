#include "MatrixOperations.cuh"

__global__ void TransposeKernelRep(Matrix Matrix)
{
    __shared__ double tile[BLOCK_SIZE * BLOCK_SIZE];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    tile[threadIdx.y * BLOCK_SIZE + threadIdx.x] = Matrix.data[y * Matrix.width + x];
    __syncthreads();
    Matrix.data[y * Matrix.width + x] = tile[threadIdx.x * BLOCK_SIZE + threadIdx.y];
}


void TransposeRep(Matrix A)
{
    Matrix d_A;
    d_A.width = A.width; d_A.length = A.length;
    size_t size = A.width * A.length;
    cudaMalloc(&d_A.data, size * sizeof(double));
    cudaMemcpy(d_A.data, A.data, size * sizeof(double), cudaMemcpyHostToDevice);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((A.width + dimBlock.x - 1) / dimBlock.x, (A.length + dimBlock.y - 1) / dimBlock.y);
    TransposeKernelRep << <dimGrid, dimBlock >> > (d_A);

    // Read C from device memory
    cudaMemcpy(A.data, d_A.data, size * sizeof(double), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A.data);
}