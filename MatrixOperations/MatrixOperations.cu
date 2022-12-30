#include "MatrixOperations.cuh"

__global__
void cudaAddMatrixKernel(const double* a,
    const double* b,
    double* c,
    const int size) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (thread_idx < size) {
        c[thread_idx] = a[thread_idx] + b[thread_idx];
        thread_idx += blockDim.x * gridDim.x;
    }
}

void cudaCallAddMatrixKernel(const double* a,
    const double* b,
    double* c,
    const int size) {

    double* dev_a = 0;
    double* dev_b = 0;
    double* dev_c = 0;
    int per_block_thread_count = 1024;

    int block_count = (int)ceil(size / (float)per_block_thread_count);


    cudaMalloc((void**)&dev_c, size * sizeof(double));
    cudaMalloc((void**)&dev_a, size * sizeof(double));
    cudaMalloc((void**)&dev_b, size * sizeof(double));


    cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);

    cudaAddMatrixKernel << < block_count, per_block_thread_count >> > (dev_a, dev_b, dev_c, size);

    cudaDeviceSynchronize();

    cudaMemcpy(c, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);


    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
};

Matrix AddMatrix(Matrix a, Matrix b) {
    int size;
    Matrix c;
    double* dev_a = 0;
    double* dev_b = 0;
    double* dev_c = 0;
    int per_block_thread_count = 1024;

    if (a.length != b.length && a.width != b.width) {
        throw std::invalid_argument("Shapes aren't matching");
    }
    else {
        size = a.length * a.width;
        c.length = a.length;
        c.width = a.width;
        c.data = new double[size];
    };

    int block_count = (int)ceil(size / (float)per_block_thread_count);


    cudaMalloc((void**)&dev_c, size * sizeof(double));
    cudaMalloc((void**)&dev_a, size * sizeof(double));
    cudaMalloc((void**)&dev_b, size * sizeof(double));


    cudaMemcpy(dev_a, a.data, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data, size * sizeof(double), cudaMemcpyHostToDevice);

    cudaAddMatrixKernel << < block_count, per_block_thread_count >> > (dev_a, dev_b, dev_c, size);

    cudaDeviceSynchronize();

    cudaMemcpy(c.data, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);


    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    return c;
};


//--------------------------------------------------------------
// CPU MULTIPLICATION
//--------------------------------------------------------------
Matrix cpu_matrix_mult(Matrix a, Matrix b) {
    if (a.width != b.length) {
        std::cout << "Shapes are not matching";
    }

    Matrix c;
    c.length = a.length;
    c.width = b.width;
    c.data = new double[c.length * c.width];

    for (int row_a = 0; row_a < a.length; row_a++) {
        for (int col_b = 0; col_b < b.width; col_b++) {

            double temp = 0;
            for (int i = 0; i < a.width; i++) {
                temp += a.data[row_a * a.width + i] * b.data[i * b.width + col_b];
            }
            c.data[row_a * b.width + col_b] = temp;
        }
    }
    return c;
}

//--------------------------------------------------------------
// MULTIPLICATION WITHOUT SHARED MEMORY
//--------------------------------------------------------------


// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, double* C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    double Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.data[row * A.width + e] * B.data[e * B.width + col];
    C[row * B.width + col] = Cvalue;
}

Matrix MatMul(Matrix A, Matrix B)
{
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
    double* dev_c = 0;
    cudaMalloc((void**)&dev_c, A.length * B.width * sizeof(double));

    // Invoke kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(B.width / dimBlock.x + 1, A.length / dimBlock.y + 1);
    MatMulKernel << <dimGrid, dimBlock >> > (d_A, d_B, dev_c);

    Matrix C;
    C.length = A.length;
    C.width = B.width;
    size = A.length * B.width;
    C.data = new double[size];
    // Read C from device memory
    cudaMemcpy(C.data, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A.data);
    cudaFree(d_B.data);
    cudaFree(dev_c);
    return C;
}

//--------------------------------------------------------------
// MULTIPLICATION WITH SHARED MEMORY
//--------------------------------------------------------------

// Get a matrix element
__device__ float GetElement(const SubMatrix A, int row, int col)
{
    return A.data[row * A.width + col];
}
// Set a matrix element
__device__ void SetElement(SubMatrix A, int row, int col,
    float value)
{
    A.data[row * A.width + col] = value;
}
// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ SubMatrix GetSubMatrix(SubMatrix A, int row, int col)
{
    SubMatrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.length = BLOCK_SIZE;
    Asub.data = &A.data[A.width * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
Matrix MatMulSH(const SubMatrix A, const SubMatrix B)
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
    return C;
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernelSH(SubMatrix A, SubMatrix B, SubMatrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Each thread block computes one sub-matrix Csub of C
    SubMatrix Csub = GetSubMatrix(C, blockRow, blockCol);
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        SubMatrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        SubMatrix Bsub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}