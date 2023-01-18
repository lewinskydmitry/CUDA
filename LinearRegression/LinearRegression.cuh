#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <cmath>

#include "../Matrix/Matrix.h"
#include "../MatrixOperations/MatrixOperations.cuh"

Matrix fit(Matrix X, Matrix y, int epoch);
