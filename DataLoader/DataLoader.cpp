#include "DataLoader.h"

DataLoader::DataLoader(std::string path, char delimiter = ';') {
    std::ifstream fin(path);
    for (std::string line; std::getline(fin, line); )
    {
        std::replace(line.begin(), line.end(), delimiter, ' ');
        std::istringstream in(line);
        vector2d.push_back(
            std::vector<double>(std::istream_iterator<double>(in),
                std::istream_iterator<double>()));
    }
    length = vector2d.size();
    width = vector2d[0].size();
    data = new double[length * width];
    create_data();
};


void DataLoader::create_data() {
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < width; j++) {
            data[i * width + j] = vector2d[i][j];
        }
    }
}

void DataLoader::print() {
    for (int i = 0; i < length * width; i++) {
        if ((i + 1) % width == 0 && i != 0) {
            std::cout << data[i] << "\n";
        }
        else {
            std::cout << data[i] << " ";
        }
    }
}

void DataLoader::print_matrix(double* matrix, int length, int width) {
    for (int i = 0; i < length * width; i++) {
        if ((i + 1) % width == 0 && i != 0) {
            std::cout << matrix[i] << "\n";
        }
        else {
            std::cout << matrix[i] << " ";
        }
    }
}
