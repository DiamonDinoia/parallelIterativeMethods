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
		/**
		 *
		 * @param A linear system matrix
		 * @param b known term vector
		 * @param iterations max number of iterations
		 * @param tolerance min error tolerated
		 * @param workers number of threads
		 * @param blockSize size of the block
		 */
		explicit denseBlocksJacobi(
			const Eigen::Matrix<Scalar, SIZE, SIZE>& A,
			const Eigen::ColumnVector<Scalar, SIZE>& b,
			const ulonglong iterations,
			const Scalar tolerance,
			const ulong workers = 0L,
			const ulonglong blockSize = 0L) :
			jacobi<Scalar, SIZE>::jacobi(A, b, iterations, tolerance, workers) {

			this->blockSize = blockSize;

			if (blockSize == 0)
				this->blockSize = std::max(ulong(this->A.cols() / workers), (ulong)1L);
			splitter();
		}

		/**
		 *
		 * @return
		 */
		Eigen::ColumnVector<Scalar, SIZE> solve() {

			Eigen::ColumnVector<Scalar, SIZE> oldSolution(this->solution);
			std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> inverses(blocks.size());

			// compute the inverses of the blocks and memorize it
            #pragma omp parallel for
			for (int i = 0; i < blocks.size(); ++i) {
				inverses[i] = this->A.block(blocks[i].startCol, blocks[i].startRow, blocks[i].cols,
					blocks[i].rows).inverse();
			}

			// start iterations
			auto iteration = 0L;

			std::vector<int> index;

			for (iteration; iteration < this->iterations; ++iteration) {


                #pragma omp parallel for firstprivate(oldSolution) schedule(dynamic)
				for (int i = 0; i < inverses.size(); ++i) {
					// set zero the components of the solution b that corresponds to the inverse
					Eigen::ColumnVector<Scalar, Eigen::Dynamic> oldBlock = oldSolution.segment(blocks[i].startCol,
						blocks[i].cols);

                    auto zeroBlock = oldSolution.segment(blocks[i].startCol, blocks[i].cols);

                    zeroBlock.setZero();
					// the segment of the solution b that this inverse approximates
					auto block = this->solution.segment(blocks[i].startCol, blocks[i].cols);
					// approximate the solution using the inverse and the solution at the previous iteration
					block = inverses[i] *
						(this->b - (this->A * oldSolution)).segment(blocks[i].startCol, blocks[i].cols);

                    zeroBlock = oldBlock;

					if ((oldBlock - block).template lpNorm<1>() / block.size() <= this->tolerance) {
                        #pragma omp critical
						index.emplace_back(i);
					}
				}

				if (!index.empty()) {
					std::sort(index.rbegin(), index.rend());
					for (auto i : index) {
						blocks.erase(blocks.begin() + i);
						inverses.erase(inverses.begin() + i);
					}
					index.clear();
					if (inverses.empty()) break;
				}
				std::swap(this->solution, oldSolution);

			}
            std::cout << iteration << std::endl;
			return Eigen::ColumnVector<Scalar, SIZE>(this->solution);
		}

	protected:
		ulonglong blockSize;
		std::vector<Index> blocks;

		void splitter() {
			for (ulonglong i = 0; i < this->A.cols(); i += blockSize) {
				blocks.emplace_back(Index(i, std::min(blockSize, (ulonglong)this->A.cols() - i),
					i, std::min(blockSize, (ulonglong)this->A.rows() - i)));
			}
		}


	private:

	};

}

#endif //PARALLELITERATIVE_BLOCKSJACOBI_H
