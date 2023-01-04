#include "MatrixOperations.cuh"

__global__ void copySharedMem(const Matrix init, Matrix result)
{
    __shared__ float tile[BLOCK_SIZE * BLOCK_SIZE];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int width = gridDim.x * BLOCK_SIZE;

    for (int j = 0; j < BLOCK_SIZE; j += BLOCK_SIZE)
        tile[(threadIdx.y + j) * BLOCK_SIZE + threadIdx.x] = init.data[(y + j) * width + x];

    __syncthreads();

    for (int j = 0; j < BLOCK_SIZE; j += BLOCK_SIZE)
        result.data[(y + j) * width + x] = tile[(threadIdx.y + j) * BLOCK_SIZE + threadIdx.x];
}


Matrix MatTranspose(Matrix A)
{
    Matrix d_A;
    d_A.width = A.width; d_A.length = A.length;
    size_t size = A.width * A.length;
    cudaMalloc(&d_A.data, size * sizeof(double));
    cudaMemcpy(d_A.data, A.data, size * sizeof(double), cudaMemcpyHostToDevice);

    Matrix d_C;
    d_C.length = A.width;
    d_C.width = A.length;
    size = A.length * A.width;
    cudaMalloc(&d_C.data, size * sizeof(double));

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((A.width + dimBlock.x - 1) / dimBlock.x, (A.length + dimBlock.y - 1) / dimBlock.y);
    copySharedMem << <dimGrid, dimBlock >> > (d_A, d_C);

    Matrix C;
    C.length = A.width;
    C.width = A.length;
    size = A.length * A.width;
    C.data = new double[size];

    // Read C from device memory
    cudaMemcpy(C.data, d_C.data, size * sizeof(double), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A.data);
    cudaFree(d_C.data);
    return C;
}