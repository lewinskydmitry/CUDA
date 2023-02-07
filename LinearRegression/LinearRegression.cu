#include "LinearRegression.cuh"



__global__ void bias_update(Matrix difference, Matrix bias, double* THETA)
{
    int size = difference.width * difference.length;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    double result = 0;

    for (int i = 0; i < size; i++) {
        result += 2 * *THETA * difference.data[i] / difference.length;
    }

    for (int i = 0; i < size; i++) {
        bias.data[i] -= result;
    }
}


__global__ void clear_matrix(Matrix matrix)
{
    int size = matrix.width * matrix.length;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx < size) {
        matrix.data[thread_idx] = 0;
    }
}


__global__ void step_control(Matrix d_error, Matrix weights, Matrix b,  int epoch, double* THETA)
{
    int blocksPerGrid_w = (weights.width * weights.length + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid_b = (b.width * b.length + threadsPerBlock - 1) / threadsPerBlock;

    if (d_error.data[epoch - 1] * 1.2 < d_error.data[epoch] && epoch != 0) {
        *THETA = *THETA / 10;
        clear_matrix << < blocksPerGrid_w, threadsPerBlock >> > (weights);
        clear_matrix << < blocksPerGrid_b, threadsPerBlock >> > (b);
    }
}


__global__ void weights_update(Matrix weights, Matrix weightsGrad, double* THETA, int length_data)
{
    int size = weightsGrad.width * weightsGrad.length;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx < size) {
        weights.data[thread_idx] += 2 * *THETA * (weightsGrad.data[thread_idx] / length_data + 2 * weights.data[thread_idx]);
    }
}


__global__ void LossFuncRed(Matrix erMatrix, Matrix difference, int iteration) {

    int size = difference.width * difference.length;
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = 0;

    if (i < size) {
        sdata[tid] = pow(difference.data[i], 2) / difference.length;
    }
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        erMatrix.data[iteration] = sdata[0];
    }
}



Matrix LinearRegression::fit(Matrix X, Matrix y, int epochs) {

    // INIT MATRICES
    Matrix XT = Transpose(X);

    Matrix d_X;
    d_X.width = X.width; d_X.length = X.length; d_X.stride = X.width;
    cudaMalloc(&d_X.data, X.width * X.length * sizeof(double));
    cudaMemcpy(d_X.data, X.data, X.width * X.length * sizeof(double), cudaMemcpyHostToDevice);

    double* THETA;
    cudaMalloc((void**)&THETA, sizeof(double));
    cudaMemcpy(THETA, &h_THETA, sizeof(double), cudaMemcpyHostToDevice);

    Matrix d_XT;
    d_XT.width = XT.width; d_XT.length = XT.length; d_XT.stride = XT.width;
    cudaMalloc(&d_XT.data, XT.width * XT.length * sizeof(double));
    cudaMemcpy(d_XT.data, XT.data, XT.width * XT.length * sizeof(double), cudaMemcpyHostToDevice);

    Matrix d_y;
    d_y.width = y.width; d_y.length = y.length;  d_y.stride = y.width;
    cudaMalloc(&d_y.data, y.width * y.length * sizeof(double));
    cudaMemcpy(d_y.data, y.data, y.width * y.length * sizeof(double), cudaMemcpyHostToDevice);

    // VARIABLES FOR OPTIMIZATION
    Matrix d_w;
    d_w.width = y.width; d_w.length = X.width; d_w.stride = y.width;
    cudaMalloc(&d_w.data, X.width * sizeof(double));
    cudaMemset(d_w.data, 0, X.width * sizeof(double));

    Matrix d_wGrad;
    d_wGrad.width = y.width; d_wGrad.length = X.width; d_wGrad.stride = y.width;
    cudaMalloc(&d_wGrad.data, X.width * sizeof(double));

    Matrix d_b;
    d_b.width = y.width; d_b.length = y.length; d_b.stride = y.width;
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
    cudaMemset(&d_error, 0, epochs * sizeof(double));


    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((y.width + dimBlock.x - 1) / dimBlock.x, (X.length + dimBlock.y - 1) / dimBlock.y);
    int blocksPerGrid = (y.width * y.length + threadsPerBlock - 1) / threadsPerBlock;

    
    for (int epoch = 0; epoch < epochs; epoch++) {

        //---------FORWARD PASS--------------------------
        //------------------------------------------------------------
        // Multyply X*w
        //------------------------------------------------------------
        
        MatMulKernel << < dimGrid, dimBlock >> > (d_X, d_w, d_pred);

        //------------------------------------------------------------
        // Add bias
        //------------------------------------------------------------
        AddMatrixRepKernel << < blocksPerGrid, threadsPerBlock >> > (d_pred, d_b);

        //------------------------------------------------------------
        // Calculate loss
        //------------------------------------------------------------
        SubMatrixKernel << < blocksPerGrid, threadsPerBlock >> > (d_y, d_pred, d_difference);


        //---------BACKWARD PASS--------------------------
        LossFuncRed << < blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double) >> > (d_error, d_difference, epoch);
        bias_update << < 1,1 >> > (d_difference, d_b, THETA);
        MatMulKernel << < dimGrid, dimBlock >> > (d_XT, d_difference, d_wGrad);
        weights_update << < blocksPerGrid, threadsPerBlock >> > (d_w, d_wGrad, THETA, y.length);
        step_control << < 1, 1 >> > (d_error, d_w, d_b, epoch, THETA);
    }
    

    Matrix result;
    result.width = y.width; result.length = y.length;
    result.data = new double[result.width * result.length];
    cudaMemcpy(result.data, d_difference.data, result.width * result.length * sizeof(double), cudaMemcpyDeviceToHost);

    losses.width = 1; losses.length = epochs;
    losses.data = new double[epochs];
    cudaMemcpy(losses.data, d_error.data, epochs * sizeof(double), cudaMemcpyDeviceToHost);

    grads.width = y.width; grads.length = X.width;
    grads.data = new double[X.width];
    cudaMemcpy(grads.data, d_w.data, X.width * sizeof(double), cudaMemcpyDeviceToHost);

    b.width = y.width; b.length = y.length;
    b.data = new double[y.length];
    cudaMemcpy(b.data, d_b.data, y.length * sizeof(double), cudaMemcpyDeviceToHost);

    gradW.width = y.width; gradW.length = X.width;
    gradW.data = new double[X.width];
    cudaMemcpy(gradW.data, d_wGrad.data, X.width * sizeof(double), cudaMemcpyDeviceToHost);

    predict.width = y.width; predict.length = y.length;
    predict.data = new double[y.length];
    cudaMemcpy(predict.data, d_pred.data, y.length * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_X.data);
    cudaFree(d_XT.data);
    cudaFree(d_y.data);

    cudaFree(d_w.data);
    cudaFree(d_wGrad.data);
    cudaFree(d_b.data);
    cudaFree(d_pred.data);
    cudaFree(d_difference.data);
    cudaFree(d_error.data);
    cudaFree(THETA);

    return result;
};