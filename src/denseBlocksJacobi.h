//
// Created by mbarb on 23/01/2018.
//

#ifndef PARALLELITERATIVE_BLOCKSJACOBI_H
#define PARALLELITERATIVE_BLOCKSJACOBI_H


#include "Eigen"
#include "utils.h"
#include "jacobi.h"
#include <typeinfo>


namespace Iterative {

	template <typename Scalar, long long SIZE>
	class denseBlocksJacobi : public jacobi<Scalar, SIZE> {

	public:
        /**
         *
         * @param matrix linear system matrix
         * @param vector known term vector
         * @param iterations max number of iterations
         * @param tolerance min error tolerated
         * @param workers number of threads
         * @param blockSize size of the block
         */
		explicit denseBlocksJacobi(
			const Eigen::Matrix<Scalar, SIZE, SIZE>& matrix,
			const Eigen::ColumnVector<Scalar, SIZE>& vector,
			const ulonglong iterations,
			const Scalar tolerance,
			const ulong workers=0L,
			const ulonglong blockSize = 0L):
				jacobi<Scalar,SIZE>::jacobi(matrix, vector, iterations, tolerance, workers) {

			this->blockSize = blockSize;

			if (blockSize == 0)
				this->blockSize = std::max(ulong(this->matrix.cols() / workers), (ulong) 1L);
			splitter();
		}


		Eigen::ColumnVector<Scalar, SIZE> solve() {

            Eigen::ColumnVector<Scalar, SIZE> old_solution(this->solution);
			Scalar error = this->tolerance - this->tolerance;
            std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> inverses (blocks.size());

            // compute the inverses of the blocks and memorize it
            #pragma omp parallel for
            for (int i = 0; i < blocks.size(); ++i) {
                inverses[i] = this->matrix.block(blocks[i]->startCol, blocks[i]->startRow, blocks[i]->cols,
                                                 blocks[i]->rows).inverse();
            }

            Eigen::ColumnVector<Scalar, SIZE> buffer(this->solution);

            // start iterations
			for (auto iteration = 0; iteration < this->iterations; ++iteration) {


                #pragma omp parallel for firstprivate(buffer) schedule(static)
                for (int i = 0; i < inverses.size(); ++i) {
                    // set zero the components of the solution vector that corresponds to the inverse
                    buffer.segment(blocks[i]->startCol, blocks[i]->cols).setZero();
                    // the segment of the solution vector that this inverse approximates
                    auto block = this->solution.segment(blocks[i]->startCol,blocks[i]->cols);
                    // approximate the solution using the inverse and the solution at the previous iteration
					block = inverses[i]*(this->vector-(this->matrix*buffer)).segment(blocks[i]->startCol,
                                                                                             blocks[i]->cols);
                }

                //compute the error
                error += (this->solution - old_solution).template lpNorm<1>();
                // check the error
                error /= this->solution.size();
                if (error <= this->tolerance) break;

                std::swap(this->solution, old_solution);
                buffer = this->solution;

            }


            return Eigen::ColumnVector<Scalar, SIZE>(this->solution);
		}

	protected:
		ulonglong blockSize;
		std::vector<Index*> blocks;

		void splitter() {
			for (ulonglong i = 0; i < this->matrix.cols(); i += blockSize) {
                blocks.emplace_back(new Index(i, std::min(blockSize, (ulonglong) this->matrix.cols() - i),
                                              i, std::min(blockSize, (ulonglong) this->matrix.rows() - i)));
            }
		}


	private:


	};

}

#endif //PARALLELITERATIVE_BLOCKSJACOBI_H
