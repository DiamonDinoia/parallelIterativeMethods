//
// Created by mbarb on 23/02/2018.
//

#ifndef PARALLELITERATIVE_SPARSEOPTIMIZEDBLOCKSJACOBI_H
#define PARALLELITERATIVE_SPARSEOPTIMIZEDBLOCKSJACOBI_H


#include "Eigen"
#include "utils.h"
#include "sparseParallelJacobi.h"

namespace Iterative {

    template <typename Scalar>
    class sparseOptimizedBlocksJacobi : public sparseParallelJacobi<Scalar> {

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
        explicit sparseOptimizedBlocksJacobi(
                const Eigen::SparseMatrix<Scalar>& A,
                const Eigen::ColumnVector<Scalar, Eigen::Dynamic>& b,
                const ulonglong iterations,
                const Scalar tolerance,
                const ulong workers = 0L,
                const ulonglong blockSize = 0L) :
                sparseParallelJacobi<Scalar>::sparseParallelJacobi(A, b, iterations, tolerance, workers) {

            this->blockSize = blockSize;

            if (blockSize == 0)
                this->blockSize = std::max(ulong(this->A.cols() / workers), (ulong)1L);
            splitter();
        }

        /**
         *
         * @return
         */
        const Eigen::ColumnVector<Scalar, Eigen::Dynamic> &solve() {

            Eigen::ColumnVector<Scalar, Eigen::Dynamic> oldSolution(this->solution);
            std::vector<Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>> inverses(blocks.size());

            Eigen::Matrix<Scalar,Eigen::Dynamic, Eigen::Dynamic> I(this->blockSize,this->blockSize);
            Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;

            I.setIdentity();

            // compute the inverses of the blocks and memorize it
            #pragma omp parallel for firstprivate(I) private(solver)
            for (int i = 0; i < blocks.size()-1; ++i) {
                Eigen::SparseMatrix<Scalar> block = this->A.block(blocks[i].startCol, blocks[i].startRow, blocks[i].cols,
                                                                  blocks[i].rows);
                solver.compute(block);
                inverses[i] = solver.solve(I);
            }
            {

                Eigen::SparseMatrix<Scalar> block = this->A.block(blocks.back().startCol, blocks.back().startRow,
                                                                  blocks.back().cols,blocks.back().rows);
                if(block.cols()!=this->blockSize || block.rows()!=this->blockSize){
                    I.resize(block.rows(), block.cols());
                    I.setIdentity();
                }
                solver.compute(block);
                inverses.back() = solver.solve(I);

            }


            // start iterations

            std::vector<int> index;


            Eigen::ColumnVector<Scalar, Eigen::Dynamic> Ax =
                    Eigen::ColumnVector<Scalar, Eigen::Dynamic>::Zero(this->solution.rows(),this->solution.cols());


            for (this->iteration=0L; this->iteration < this->iterations; ++this->iteration) {

                Ax = this->A*oldSolution;

                #pragma omp parallel for schedule(dynamic)
                for (auto i = 0; i < inverses.size(); ++i) {


                    auto oldBlock = oldSolution.segment(blocks[i].startCol, blocks[i].cols);//
                    // the segment of the solution b that this inverse approximates
                    auto block = this->solution.segment(blocks[i].startCol, blocks[i].cols);
                    // approximate the solution using the inverse and the solution at the previous iteration

                    Eigen::ColumnVector<Scalar,Eigen::Dynamic> correction =
                            Eigen::ColumnVector<Scalar,Eigen::Dynamic>::Zero(oldSolution.rows(), oldSolution.cols());


                    for (auto col = blocks[i].startCol; col < blocks[i].startCol+blocks[i].cols; ++col) {
                        correction+=this->A.col(col)*oldSolution[col];
                    }

                    block = inverses[i] * (this->b - Ax + correction).segment(blocks[i].startCol, blocks[i].cols);


                    if ((oldBlock - block).template lpNorm<1>() <= this->tolerance*block.size()) {
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
                    if (inverses.empty()) break;
                    index.clear();
                }
                std::swap(this->solution, oldSolution);

            }
            std::cout << this->iteration << std::endl;
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


#endif //PARALLELITERATIVE_SPARSEOPTIMIZEDBLOCKSJACOBI_H
