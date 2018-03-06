//
// Created by mbarb on 05/02/2018.
//

#ifndef PARALLELITERATIVE_SPARSEOVERLAPPINGJACOBI_H
#define PARALLELITERATIVE_SPARSEOVERLAPPINGJACOBI_H



#include "Eigen"
#include "utils.h"
#include "sparseParallelJacobi.h"
#include <typeinfo>


namespace Iterative {

    template <typename Scalar>
    class sparseOverlappingJacobi : public sparseParallelJacobi<Scalar> {
    public:

        explicit sparseOverlappingJacobi(
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
				solver.compute(block);
                if(I.size() != block.size()){
                    I.resize(block.rows(), block.cols());
                    I.setIdentity();
                }
                inverses[i].first = i;
                inverses[i].second = solver.solve(I);
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

            std::vector<int> index;


            for (this->iteration=0L; this->iteration < this->iterations; ++this->iteration) {

                // Calculate the solution in parallel
                #pragma omp parallel for firstprivate(oldSolution) schedule(dynamic)
                for (int i = 0; i < inverses.size(); ++i) {

                    Eigen::ColumnVector<Scalar, Eigen::Dynamic> oldBlock = inverses[i].first%2 ?
                           odd_solution.segment(blocks[i].startCol, blocks[i].cols) :
                                 even_solution.segment(blocks[i].startCol, blocks[i].cols);

                    auto zeroBlock = oldSolution.segment(blocks[i].startCol, blocks[i].cols);

                    zeroBlock.setZero();

                    auto block = inverses[i].first%2 ? odd_solution.segment(blocks[i].startCol, blocks[i].cols) :
                                 even_solution.segment(blocks[i].startCol, blocks[i].cols);

                    block = inverses[i].second * (this->b - (this->A * oldSolution)).segment(blocks[i].startCol,
                                                                                           blocks[i].cols);

                    if ((oldBlock - block).template lpNorm<1>() <= this->tolerance*block.size()) {
                        #pragma omp critical
                        index.emplace_back(i);
                    }

                    zeroBlock = block;

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
            std::cout << this->iteration << std::endl;
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


#endif //PARALLELITERATIVE_DENSEOVERLAPPINGJACOBI_H
