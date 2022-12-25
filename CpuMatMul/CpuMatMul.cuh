#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include "../Matrix/Matrix.h"

Matrix cpu_matrix_mult(Matrix a, Matrix b);