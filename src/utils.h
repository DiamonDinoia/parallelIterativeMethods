//
// Created by mbarb on 24/01/2018.
//

#ifndef PARALLELITERATIVE_UTILS_H
#define PARALLELITERATIVE_UTILS_H

#include <chrono>
#include "Eigen"

namespace Eigen {


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

	};

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<double> dsec;


}
#endif //PARALLELITERATIVE_UTILS_H
