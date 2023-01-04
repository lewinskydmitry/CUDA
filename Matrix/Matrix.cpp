#include "Matrix.h"

Matrix Matrix::read_csv(std::string path, char delimiter =';') {
    Matrix result;
    std::vector< std::vector<double> > vector2d;

    std::ifstream fin(path);
    for (std::string line; std::getline(fin, line); )
    {
        std::replace(line.begin(), line.end(), delimiter, ' ');
        std::istringstream in(line);
        vector2d.push_back(
            std::vector<double>(std::istream_iterator<double>(in),
                std::istream_iterator<double>()));
    }
    result.length = vector2d.size();
    result.width = vector2d[0].size();
    result.data = new double[result.length * result.width];
    result.create_data(vector2d, result);
    return result;
}


void Matrix::create_data(std::vector< std::vector<double> > vector2d, Matrix matrix) {
    for (int i = 0; i < matrix.length; i++) {
        for (int j = 0; j < matrix.width; j++) {
            matrix.data[i * matrix.width + j] = vector2d[i][j];
        }
    }
}


void Matrix::print() {
    for (int i = 0; i < length * width; i++) {
        if ((i + 1) % width == 0 && i != 0) {
            std::cout << data[i] << "\n";
        }
        else {
            std::cout << data[i] << " ";
        }
    }
}


void Matrix::print_matrix(Matrix matrix) {
    size_t length = matrix.length;
    size_t width = matrix.width;

    for (int i = 0; i < length * width; i++) {
        if ((i + 1) % width == 0 && i != 0) {
            std::cout << matrix.data[i] << "\n";
        }
        else {
            std::cout << matrix.data[i] << " ";
        }
    }
}


Matrix Matrix::create_matrix(int length, int width, int min, int max) {
    Matrix rand;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(min, max);

    rand.length = length;
    rand.width = width;
    size_t size = rand.length * rand.width;
    rand.data = new double[size];

    for (int i = 0; i < size; i++) {
        rand.data[i] = uni(rng);
    }
    return rand;
}

int Matrix::equal(Matrix comp_matrix) {
    if (comp_matrix.length != length || comp_matrix.width != width) {
        std::cout << "Matrixes are not equal SHAPES" << std::endl;
        return 1;
    }
    for (int row = 0; row < length; row++) {
        for (int col = 0; col < width; col++) {
            if (data[row * width + col] != comp_matrix.data[row * width + col]) {
                std::cout << "Matrixes are not equal#"<< row * width + col << std::endl;
                return 1;
            }
        }
    }
    std::cout << "Matrixes are equal" << std::endl;
    return 0;
}
