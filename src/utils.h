//
// Created by mbarb on 24/01/2018.
//

#ifndef PARALLELITERATIVE_UTILS_H
#define PARALLELITERATIVE_UTILS_H

#include <chrono>
#include "Eigen"

namespace Eigen {
	//
	//    template<typename Scalar, int SIZE>
	//    using SquareMatrix = Matrix<Scalar, SIZE, SIZE>;

	template <typename Scalar, int SIZE>
	using ColumnVector = Matrix<Scalar, SIZE, 1>;


};

namespace Iterative {

	using ulong = unsigned long;
	using ulonglong = unsigned long long;

	class Index {
        public:
            explicit Index(const ulonglong startCol, const ulonglong cols, const ulonglong startRow,
						   const ulonglong rows) :
					startCol(startCol), cols(cols), startRow(startRow), rows(rows){}

			ulonglong startCol;
			ulonglong cols;
            ulonglong startRow;
            ulonglong rows;

		friend std::ostream& operator<< (std::ostream &out, const Index &index){

			out << index.startRow << ' ';
			out << index.startCol << ' ';
			out << index.cols << ' ';
			out << index.rows << ' ';

			return out << std::endl;
		}

//		Index &operator = (const Index &index){
//
//		}
	};

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<double> dsec;


	template <typename T>
	T rand_t(T min, T max) {
		T range = max - min;
		T r = (T)rand() / (T)RAND_MAX;
		return (r * range) + min;
	}


//	template <typename T>
//	void generate_vector(const ulong size, std::vector<T>& v, const T min, const T max) {
//		for (int i = 0; i < size; ++i) { v.emplace_back(rand_t(min, max)); }
//	}


	template <typename T>
	void generate_diagonal_dominant_matrix(T& matrix) {
        for (int i = 0; i < matrix.rows(); ++i) {
            matrix.row(i)=matrix.row(i)/matrix.row(i).max();
            auto value = matrix.row(i).template lpNorm<1>();
            value-=matrix(i,i);
            matrix(i,i)=value;

        }

	}

	template <typename T>
	T check_error(const std::vector<std::vector<T>>& matrix, const std::vector<T>& terms,
	              const std::vector<T>& solution) {
		T tmp = solution[0] - solution[0];
		T error = tmp;

		for (int i = 0; i < matrix.size(); ++i) {
			tmp = terms[i];
			for (int j = 0; j < matrix[i].size(); ++j) { tmp -= (matrix[i][j] * solution[j]); }
			error += tmp;
		}
		return error;
	}

}
#endif //PARALLELITERATIVE_UTILS_H
