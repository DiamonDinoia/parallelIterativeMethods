//
// Created by mbarb on 24/01/2018.
//

#ifndef PARALLELITERATIVE_UTILS_H
#define PARALLELITERATIVE_UTILS_H

#include "Eigen"

namespace Eigen {
//
//    template<typename Scalar, int SIZE>
//    using SquareMatrix = Matrix<Scalar, SIZE, SIZE>;

    template <typename Scalar, int SIZE>
    using ColumnVector = Matrix <Scalar, SIZE, 1>;


	template <typename Scalar, int SIZE>
	void swap(Eigen::ColumnVector<Scalar, SIZE> &a, Eigen::ColumnVector<Scalar, SIZE> &b) {
		Eigen::ColumnVector<Scalar, SIZE> tmp = a;
		a = b;
		b = tmp;
	}

};

namespace Iterative {

    using ulong = unsigned long;
    using ulonglong = unsigned long long;

    typedef struct Index {
		Index(ulonglong startCol, ulonglong endCol, ulonglong startRow, ulonglong endRow) : 
    		startCol(startCol), endCol(endCol), startRow(startRow), endRow(endRow){}
        const ulonglong startRow;
        const ulonglong endRow;
		const ulonglong startCol;
        const ulonglong endCol;
    } Index;


    template<typename T>
    inline T rand_t(T min, T max) {
        T range = max - min;
        T r = (T) rand() / (T) RAND_MAX;
        return (r * range) + min;
    }


    template<typename T>
    inline void generate_vector(const ulong size, std::vector<T> &v, const T min, const T max) {
        for (int i = 0; i < size; ++i) {
            v.emplace_back(rand_t(min, max));
        }
    }


    template<typename Scalar, ulonglong Size>
    void generate_diagonal_dominant_matrix(const ulong size, Eigen::Matrix<Scalar, Size, Size> &matrix,
                                           const Scalar min, const Scalar max) {
        if (Size == Eigen::Dynamic)
            matrix = Eigen::Matrix<Scalar,Size,Size>::Random(size);
        else matrix = Eigen::Matrix<Scalar,Size,Size>::Random();
//        for (ulong i = 0; i < size; ++i) {
//            T sum = T(0);
//            for (auto &val: matrix[i]) sum += abs(val);
//            sum -= abs(matrix[i][i]);
//            matrix[i][i] = abs(matrix[i][i]) + sum;
//        }
    }

    template<typename T>
    T check_error(const std::vector<std::vector<T>> &matrix, const std::vector<T> &terms,
                  const std::vector<T> &solution) {
        T tmp = solution[0] - solution[0];
        T error = tmp;

        for (int i = 0; i < matrix.size(); ++i) {
            tmp = terms[i];
            for (int j = 0; j < matrix[i].size(); ++j) {
                tmp -= (matrix[i][j] * solution[j]);
            }
            error += tmp;
        }
        return error;
    }

}
#endif //PARALLELITERATIVE_UTILS_H
