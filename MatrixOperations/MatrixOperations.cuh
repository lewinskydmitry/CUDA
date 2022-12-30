#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include "../Matrix/Matrix.h"
#include <stdexcept>

void cudaCallAddMatrixKernel(const double* a,
    const double* b,
    double* c,
    const int size);

Matrix AddMatrix(Matrix a, Matrix b);

Matrix cpu_matrix_mult(Matrix a, Matrix b);

// Forward declaration of the matrix multiplication kernel
//__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

Matrix MatMul(Matrix A, Matrix B);

//-------------------------------------------------------------
// SHARED MEMORY
//-------------------------------------------------------------
// 
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int length;
    double* data;
} SubMatrix;


// Get a matrix element
__device__ float GetElement(const SubMatrix A, int row, int col);

// Set a matrix element
__device__ void SetElement(SubMatrix A, int row, int col, double value);

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ SubMatrix GetSubMatrix(SubMatrix A, int row, int col);

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernelSH(const SubMatrix, const SubMatrix, SubMatrix);
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
Matrix MatMulSH(const Matrix A, const Matrix B);
#define BLOCK_SIZE 16


