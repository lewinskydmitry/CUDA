#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include "../Matrix/Matrix.h"
#include <stdexcept>


void cudaCallAddVectorKernel(const double* a,
    const double* b,
    double* c,
    const int size);

Matrix AddVector(Matrix a, Matrix b);