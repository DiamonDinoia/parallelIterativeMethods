//
// Created by mbarb on 21/02/2018.
//

#ifndef PARALLELITERATIVE_DENSEFIXEDBLOCKSJACOBI_H
#define PARALLELITERATIVE_DENSEFIXEDBLOCKSJACOBI_H


#include "Eigen"
#include "utils.h"
#include "denseParallelJacobi.h"

namespace Iterative {

    template <typename Scalar, long long SIZE, int BLOCKSIZE>
    class denseFixedBlocksJacobi : public denseParallelJacobi<Scalar, SIZE> {

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
        explicit denseFixedBlocksJacobi(
                const Eigen::Matrix<Scalar, SIZE, SIZE>& A,
                const Eigen::ColumnVector<Scalar, SIZE>& b,
                const ulonglong iterations,
                const Scalar tolerance,
                const ulong workers = 0L,
                const ulonglong blockSize = 0L) :
                denseParallelJacobi<Scalar, SIZE>::denseParallelJacobi(A, b, iterations, tolerance, workers) {

            this->blockSize = blockSize;

            if (blockSize == 0)
                this->blockSize = std::max(ulong(this->A.cols() / workers), (ulong)1L);
            splitter();
        }

        /**
         *
         * @return
         */
        const Eigen::ColumnVector<Scalar, SIZE> solve() {

            Eigen::ColumnVector<Scalar, SIZE> oldSolution(this->solution);
            std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, BLOCKSIZE, BLOCKSIZE>>
                    inverses(blocks.size());

            // compute the inverses of the blocks and memorize it
            #pragma omp parallel for
            for (int i = 0; i < blocks.size(); ++i) {
                inverses[i] = this->A.block(blocks[i].startCol, blocks[i].startRow, blocks[i].cols,
                                            blocks[i].rows).inverse();
            }

            // start iterations
            auto iteration = 0L;

            std::vector<int> index;

            auto lastElem = inverses.size();

            for (iteration; iteration < this->iterations; ++iteration) {


                #pragma omp parallel for firstprivate(oldSolution) schedule(dynamic)
                for (int i = 0; i < lastElem; ++i) {
                    // set zero the components of the solution b that corresponds to the inverse
                    Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::AutoAlign, BLOCKSIZE, 1> oldBlock =
                            oldSolution.segment(blocks[i].startCol, blocks[i].cols);

                    auto zeroBlock = oldSolution.segment(blocks[i].startCol, blocks[i].cols);

                    zeroBlock.setZero();
                    // the segment of the solution b that this inverse approximates
                    auto block = this->solution.segment(blocks[i].startCol, blocks[i].cols);
                    // approximate the solution using the inverse and the solution at the previous iteration
                    block = inverses[i] *
                            (this->b - (this->A * oldSolution)).segment(blocks[i].startCol, blocks[i].cols);

                    zeroBlock = block;

                    if ((oldBlock - block).template lpNorm<1>() <= this->tolerance*block.size()) {
                        #pragma omp critical
                        index.emplace_back(i);
                    }
                }
                if(BLOCKSIZE != Eigen::Dynamic){
                    if (!index.empty()) {
                        std::sort(index.rbegin(), index.rend());
                        for (auto i : index) {
                            if(i!=lastElem-1) {
                                std::iter_swap(blocks.begin() + i, blocks.begin()+lastElem-1);
                                std::iter_swap(inverses.begin() + i, inverses.begin()+lastElem-1);
                            }
                            lastElem--;
                        }
                        if (lastElem<=0) break;
                        index.clear();
                    }

                }else{
                    if (!index.empty()) {
                        std::sort(index.rbegin(), index.rend());
                        for (auto i : index) {
                            blocks.erase(blocks.begin() + i);
                            inverses.erase(inverses.begin() + i);
                        }
                        index.clear();
                        lastElem = inverses.size();
                        if (inverses.empty()) break;
                    }
                }

                std::swap(this->solution, oldSolution);

            }
            std::cout << iteration << std::endl;
            return this->solution;
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


#endif //PARALLELITERATIVE_DENSEFIXEDBLOCKSJACOBI_H
