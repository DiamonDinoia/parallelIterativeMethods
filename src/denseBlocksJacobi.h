//
// Created by mbarb on 23/01/2018.
//

#ifndef PARALLELITERATIVE_BLOCKSJACOBI_H
#define PARALLELITERATIVE_BLOCKSJACOBI_H


#include "Eigen"
#include "utils.h"
#include "jacobi.h"

namespace Iterative {

	template <typename Scalar, long long SIZE>
	class denseBlocksJacobi : public jacobi<Scalar, SIZE> {

	public:

		explicit denseBlocksJacobi(
			const Eigen::Matrix<Scalar, SIZE, SIZE>& matrix,
			const Eigen::ColumnVector<Scalar, SIZE>& vector,
			const ulonglong iterations,
			const Scalar tolerance,
			const ulong workers,
            const ulonglong blockSize = 0L) :
			jacobi(matrix, vector, iterations, tolerance, workers){

            if (blockSize==0)
                this->blockSize = std::max(this->matrix.cols()/workers,1);
			splitter();
        }

		Eigen::ColumnVector<Scalar, SIZE> solve() {
			Eigen::ColumnVector<Scalar, SIZE> buffer(this->solution);
			Scalar error = this->tolerance - this->tolerance;


			for (int i = 0; i < this->iterations; ++i) {


				if (error <= this->tolerance) break;
			}

			return Eigen::ColumnVector<Scalar, SIZE>(this->solution);
		}

	protected:
        ulonglong blockSize;
		std::vector<Index> blocks;

		void splitter() {
			for (auto i = 0; i < this->matrix.cols(); i += blockSize)
				for (auto j = 0; j < this->matrix.rows(); i += blockSize)
					blocks.emplace_back(new Index(i, std::min(i + blockSize, this->matrix.cols()),
						j, std::min(j + blockSize,this->matrix.rows())));			
		}


	private:



	};

}

#endif //PARALLELITERATIVE_BLOCKSJACOBI_H
