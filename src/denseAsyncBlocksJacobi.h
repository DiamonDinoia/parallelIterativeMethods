//
// Created by mbarb on 14/02/2018.
//

#ifndef PARALLELITERATIVE_DENSEASYNCBLOCKJACOBI_H
#define PARALLELITERATIVE_DENSEASYNCBLOCKJACOBI_H


#include <Eigen>
#include <iostream>
#include <omp.h>
#include "utils.h"
#include "denseParallelJacobi.h"

namespace Iterative {

    template <typename Scalar, long long SIZE>
    class denseAsyncBlocksJacobi : public denseParallelJacobi<Scalar, SIZE> {

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
	    explicit denseAsyncBlocksJacobi(
		    const Eigen::Matrix<Scalar, SIZE, SIZE>& matrix,
		    const Eigen::ColumnVector<Scalar, SIZE>& vector,
		    const ulonglong iterations,
		    const Scalar tolerance,
		    const ulong workers=0L,
		    const ulonglong blockSize = 0L):
                denseParallelJacobi<Scalar,SIZE>::denseParallelJacobi(matrix, vector, iterations, tolerance, workers) {

            this->blockSize = blockSize;

            if (blockSize == 0)
                this->blockSize = std::max(ulong(this->A.cols() / workers), (ulong) 1L);
            splitter();
        }


        const Eigen::ColumnVector<Scalar, SIZE> solve() {

            Eigen::ColumnVector<Scalar, SIZE> oldSolution(this->solution);
            std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> inverses(blocks.size());

            // compute the inverses of the blocks and memorize it
            #pragma omp parallel for
            for (int i = 0; i < blocks.size(); ++i) {
                inverses[i] = this->A.block(blocks[i].startCol, blocks[i].startRow, blocks[i].cols,
                                            blocks[i].rows).inverse();
            }


            std::vector<int> index;

			auto stop = false;

            for (this->iteration=0L; this->iteration < this->iterations && !stop; ++this->iteration) {
                #pragma omp parallel
                #pragma omp for private(oldSolution) schedule(dynamic) nowait
                for (int i = 0; i < inverses.size(); ++i) {

                    oldSolution = this->solution;
                    // set zero the components of the solution b that corresponds to the inverse
                    Eigen::ColumnVector<Scalar, Eigen::Dynamic> oldBlock = oldSolution.segment(
                            blocks[i].startCol,
                            blocks[i].cols);

                    auto zeroBlock = oldSolution.segment(blocks[i].startCol, blocks[i].cols);

                    zeroBlock.setZero();
                    // the segment of the solution b that this inverse approximates
                    auto block = this->solution.segment(blocks[i].startCol, blocks[i].cols);
                    // approximate the solution using the inverse and the solution at the previous iteration
                    block = inverses[i] *
                            (this->b - (this->A * oldSolution)).segment(blocks[i].startCol, blocks[i].cols);


                    zeroBlock = block;


                    if ((oldBlock - block).template lpNorm<1>() / block.size() <= this->tolerance) {
                        #pragma omp critical
                        index.emplace_back(i);
                    }
                }
                if (!index.empty()) {
                    #pragma omp barrier

                    #pragma omp single
                    {
                        std::sort(index.rbegin(), index.rend());
                        for (auto i : index) {
                            blocks.erase(blocks.begin() + i);
                            inverses.erase(inverses.begin() + i);
                        }
                        index.clear();
                        stop = inverses.empty();
                    };
                }

            }
            #pragma omp barrier
            std::cout << this->iteration << std::endl;
            return this->solution;
        }

    protected:
        ulonglong blockSize;
        std::vector<Index> blocks;

        void splitter() {
            for (ulonglong i = 0; i < this->A.cols(); i += blockSize) {
                blocks.emplace_back(Index(i, std::min(blockSize, (ulonglong) this->A.cols() - i),
                                          i, std::min(blockSize, (ulonglong) this->A.rows() - i)));
            }
        }


    private:

    };

}



#endif //PARALLELITERATIVE_ASYNCJACOBI_H
