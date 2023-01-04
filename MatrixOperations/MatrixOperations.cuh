#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include "../Matrix/Matrix.h"
#include <stdexcept>
#define BLOCK_SIZE 16


//-------------------------------------------------------------
// GPU MATRIX ADDITION
//-------------------------------------------------------------

Matrix AddMatrix(Matrix A, Matrix B);

//-------------------------------------------------------------
// CPU MATRIX MULTIPLICATION
//-------------------------------------------------------------

Matrix cpu_matrix_mult(Matrix a, Matrix b);

//-------------------------------------------------------------
// GPU MATRIX MULTIPLICATION
//-------------------------------------------------------------

Matrix MatMul(Matrix A, Matrix B);

//-------------------------------------------------------------
// SHARED MEMORY GPU MATRIX MULTIPLICATION
//-------------------------------------------------------------

Matrix MatMulSH(const Matrix A, const Matrix B);
