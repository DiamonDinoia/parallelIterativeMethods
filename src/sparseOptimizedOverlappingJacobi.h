//
// Created by mbarb on 23/02/2018.
//

#ifndef PARALLELITERATIVE_SPARSEOPRIMIZEDOVERLAPPINGJACOBI_H
#define PARALLELITERATIVE_SPARSEOPRIMIZEDOVERLAPPINGJACOBI_H


#include "Eigen"
#include "utils.h"
#include "sparseParallelJacobi.h"


namespace Iterative {

    template <typename Scalar>
    class sparseOptimizedOverlappingJacobi : public sparseParallelJacobi<Scalar> {
    public:

        explicit sparseOptimizedOverlappingJacobi(
                const Eigen::SparseMatrix<Scalar>& A,
                const Eigen::ColumnVector<Scalar, Eigen::Dynamic>& b,
                const ulonglong iterations,
                const Scalar tolerance,
                const ulong workers=0L,
                const ulonglong blockSize = 0L,
                const ulonglong overlap = 0L) :
                sparseParallelJacobi<Scalar>::sparseParallelJacobi(A, b, iterations, tolerance, workers) {

            this->blockSize = blockSize;
            if (blockSize == 0)
                this->blockSize = std::max(ulong(this->A.cols() / workers), (ulong) 1L);
            if (overlap == 0)
                this->overlap = blockSize/2;
            splitter();
        }


        const Eigen::ColumnVector<Scalar, Eigen::Dynamic> solve() {

            Eigen::ColumnVector<Scalar, Eigen::Dynamic> oldSolution(this->solution);
            Scalar error = this->tolerance - this->tolerance;
            std::vector<std::pair<ulonglong, Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>>> inverses(blocks.size());

            Eigen::ColumnVector<Scalar, Eigen::Dynamic> even_solution(this->solution);
            Eigen::ColumnVector<Scalar, Eigen::Dynamic> odd_solution(this->solution);

            Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;
            Eigen::Matrix<Scalar,Eigen::Dynamic, Eigen::Dynamic> I(blocks[0].rows, blocks[0].cols);

            // Compute the inverses in parallel
            #pragma omp parallel for schedule(dynamic) private(solver)
            for (long i = 0; i < blocks.size()-1; ++i) {
                Eigen::SparseMatrix<Scalar> block = this->A.block(blocks[i].startCol, blocks[i].startRow, blocks[i].cols,
                                                                  blocks[i].rows);
//                Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver(block);
                solver.compute(block);
                if(I.size() != block.size()){
                    I.resize(block.rows(), block.cols());
                    I.setIdentity();
                }
                inverses[i].first = i;
                inverses[i].second = solver.solve(I);
//                inverses[i] = std::pair<ulonglong, Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>>(i,solver.solve(I));
            }
            {
                Eigen::SparseMatrix<Scalar> block = this->A.block(blocks.back().startCol, blocks.back().startRow,
                                                                  blocks.back().cols, blocks.back().rows);
                solver.compute(block);
                I.resize(block.rows(), block.cols());
                I.setIdentity();

                inverses.back().first = blocks.size()-1;
                inverses.back().second = solver.solve(I);
            }
            auto nInverses = blocks.size();

            auto iteration = 0L;
            std::vector<int> index;
            Eigen::ColumnVector<Scalar, Eigen::Dynamic> Ax =
                    Eigen::ColumnVector<Scalar, Eigen::Dynamic>::Zero(this->solution.rows(),this->solution.cols());


            for (iteration; iteration < this->iterations; ++iteration) {

                Ax = this->A*oldSolution;

                // Calculate the solution in parallel
                #pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < inverses.size(); ++i) {

                    Eigen::ColumnVector<Scalar, Eigen::Dynamic> oldBlock = inverses[i].first%2 ?
                                                                           odd_solution.segment(blocks[i].startCol, blocks[i].cols) :
                                                                           even_solution.segment(blocks[i].startCol, blocks[i].cols);

                    Eigen::ColumnVector<Scalar,Eigen::Dynamic> correction =
                            Eigen::ColumnVector<Scalar,Eigen::Dynamic>::Zero(oldSolution.rows(), oldSolution.cols());


                    for (auto col = blocks[i].startCol; col < blocks[i].startCol+blocks[i].cols; ++col) {
                        correction+=this->A.col(col)*oldSolution[col];
                    }


                    auto block = inverses[i].first%2 ? odd_solution.segment(blocks[i].startCol, blocks[i].cols) :
                                 even_solution.segment(blocks[i].startCol, blocks[i].cols);

                    block = inverses[i].second * (this->b - Ax + correction).segment(blocks[i].startCol,
                                                                                             blocks[i].cols);

                    if ((oldBlock - block).template lpNorm<1>() <= this->tolerance*block.size()) {
                        #pragma omp critical
                        index.emplace_back(i);
                    }

                }

                // average of the two values
                this->solution = (even_solution + odd_solution)/(Scalar)2.;

                // not overlapping portion of the solution b
                this->solution.head(overlap) = even_solution.head(overlap);

                // not overlapping end portion of the solution b
                this->solution.tail(overlap) = nInverses%2 ?
                                               even_solution.tail(overlap) : odd_solution.tail(overlap);


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
            return this->solution;
        }

    protected:
        ulonglong blockSize;
        std::vector<Index> blocks;

        ulonglong overlap;

        void splitter() {
            for (ulonglong i = 0; i < this->A.cols()-overlap; i += (blockSize-overlap))
                blocks.emplace_back(Index(i, std::min(blockSize, (ulonglong) this->A.cols() - i),
            i, std::min(blockSize, (ulonglong) this->A.rows() - i)));
        }


    private:


    };

}


#endif //PARALLELITERATIVE_SPARSEOPRIMIZEDOVERLAPPINGJACOBI_H
