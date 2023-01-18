#include "LinearRegression.cuh"


__global__ void summation(Matrix A, Matrix B)
{
    int size = A.width * A.length;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (thread_idx < size) {
        A.data[thread_idx] += B.data[thread_idx];
        thread_idx += blockDim.x * gridDim.x;
    }
}



__global__ void substraction(Matrix A, Matrix B, Matrix C)
{
    int size = A.width * A.length;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (thread_idx < size) {
        C.data[thread_idx] = A.data[thread_idx] - B.data[thread_idx];
        thread_idx += blockDim.x * gridDim.x;
    }
}


__global__ void bias_update(Matrix difference, Matrix bias)
{
    int size = difference.width * difference.length;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (thread_idx < size) {
        double result = 0;
        for (int i = 0; i < size; i++) {
            result += difference.data[thread_idx] / difference.length;
        }
        bias.data[thread_idx] = THETA * 2 * result;
        thread_idx += blockDim.x * gridDim.x;
    }
}


__global__ void weights_update(Matrix weights, Matrix weightsGrad)
{
    int size = weightsGrad.width * weightsGrad.length;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (thread_idx < size) {
        weights.data[thread_idx] += THETA * 2 * weightsGrad.data[thread_idx] / weightsGrad.length;
        thread_idx += blockDim.x * gridDim.x;
    }
}


__global__ void Transpose(Matrix A)
{
    __shared__ double tile[BLOCK_SIZE * BLOCK_SIZE];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    for (int j = 0; j < BLOCK_SIZE; j += BLOCK_SIZE)
        tile[(threadIdx.y + j) * BLOCK_SIZE + threadIdx.x] = A.data[(y + j) * A.width + x];

    __syncthreads();

    for (int j = 0; j < BLOCK_SIZE; j += BLOCK_SIZE)
        A.data[(y + j) * A.width + x] = tile[(threadIdx.x + j) * BLOCK_SIZE + threadIdx.y];
}


__global__ void LossFunc(Matrix erMatrix, Matrix difference, int iteration) {
    int size = difference.width * difference.length;
    float error = 0;
    for (int e = 0; e < size; e++) {
        error += pow(difference.data[e],2) / difference.length;
    }
    erMatrix.data[iteration] = error;
}




Matrix fit(Matrix X, Matrix y, int epochs) {

    // INIT MATRICES
    Matrix d_X;
    d_X.width = X.width; d_X.length = X.length; d_X.stride = X.width;
    cudaMalloc(&d_X.data, X.width * X.length * sizeof(double));
    cudaMemcpy(d_X.data, X.data, X.width * X.length * sizeof(double), cudaMemcpyHostToDevice);

    Matrix d_y;
    d_y.width = y.width; d_y.length = y.length;  d_y.stride = y.width;
    cudaMalloc(&d_y.data, y.width * y.length * sizeof(double));
    cudaMemcpy(d_y.data, y.data, y.width * y.length * sizeof(double), cudaMemcpyHostToDevice);

    // VARIABLES FOR OPTIMIZATION
    Matrix d_w;
    d_w.width = y.width; d_w.length = X.width; d_w.stride = y.width;
    cudaMalloc(&d_w.data, X.length * sizeof(double));

    Matrix d_wGrad;
    d_wGrad.width = y.width; d_wGrad.length = X.width; d_wGrad.stride = y.width;
    cudaMalloc(&d_wGrad.data, X.length * sizeof(double));

    Matrix d_b;
    d_w.width = y.width; d_w.length = y.length; d_w.stride = y.width;
    cudaMalloc(&d_b.data, y.width * y.length * sizeof(double));

    // VARIABLES FOR COMPUTING
    Matrix d_pred;
    d_pred.width = y.width; d_pred.length = y.length; d_pred.stride = y.width;
    cudaMalloc(&d_pred.data, y.width * y.length * sizeof(double));

    Matrix d_difference;
    d_difference.width = y.width; 
    d_difference.length = y.length;
    d_difference.stride = y.width;
    cudaMalloc(&d_difference.data, y.width * y.length * sizeof(double));

    // ERROR COUNTING
    Matrix d_error;
    d_error.width = 1; d_error.length = epochs;
    cudaMalloc(&d_error.data, epochs * sizeof(double));


    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((y.width + dimBlock.x - 1) / dimBlock.x, (X.length + dimBlock.y - 1) / dimBlock.y);
    int blocksPerGrid = (y.width * y.length + threadsPerBlock - 1) / threadsPerBlock;

    for (int epoch = 0; epoch < epochs; epoch++) {

        //---------FORWARD PASS--------------------------
        //------------------------------------------------------------
        // Multyply X*w
        //------------------------------------------------------------
        MatMulKernelSH << < dimGrid, dimBlock >> > (d_X, d_w, d_pred);

        //------------------------------------------------------------
        // Add bias
        //------------------------------------------------------------
        summation << < blocksPerGrid, threadsPerBlock >> > (d_pred, d_b);

        //------------------------------------------------------------
        // Calculate loss
        //------------------------------------------------------------
        substraction << < blocksPerGrid, threadsPerBlock >> > (d_y, d_pred, d_difference);


        //---------BACKWARD PASS--------------------------
        LossFunc << < 1, 1 >> > (d_error, d_difference, epoch);
        bias_update << < blocksPerGrid, threadsPerBlock >> > (d_difference, d_b);
        Transpose << < dimGrid, dimBlock >> > (d_X);
        MatMulKernelSH << < dimGrid, dimBlock >> > (d_X, d_difference, d_wGrad);
        Transpose << < dimGrid, dimBlock >> > (d_X);
        weights_update << < blocksPerGrid, threadsPerBlock >> > (d_w, d_wGrad);
        
    }


    Matrix result;
    result.width = y.width; result.length = y.length;
    result.data = new double[result.width * result.length];
    cudaMemcpy(result.data, d_pred.data, result.width * result.length * sizeof(double), cudaMemcpyDeviceToHost);

    Matrix h_error;
    h_error.width = 1; h_error.length = epochs;
    h_error.data = new double[epochs];
    cudaMemcpy(h_error.data, d_error.data, epochs * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_X.data);
    cudaFree(d_y.data);
    cudaFree(d_pred.data);

    return h_error;
};