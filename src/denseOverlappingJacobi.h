//
// Created by mbarb on 05/02/2018.
//

#ifndef PARALLELITERATIVE_DENSEOVERLAPPINGJACOBI_H
#define PARALLELITERATIVE_DENSEOVERLAPPINGJACOBI_H



#include "Eigen"
#include "utils.h"
#include "denseParallelJacobi.h"


namespace Iterative {

    template <typename Scalar, long long SIZE>
    class denseOverlappingJacobi : public denseParallelJacobi<Scalar, SIZE> {
    public:

        explicit denseOverlappingJacobi(
                const Eigen::Matrix<Scalar, SIZE, SIZE>& A,
                const Eigen::ColumnVector<Scalar, SIZE>& b,
                const ulonglong iterations,
                const Scalar tolerance,
                const ulong workers=0L,
                const ulonglong blockSize = 0L,
                const ulonglong overlap = 0L) :
                denseParallelJacobi<Scalar,SIZE>::denseParallelJacobi(A, b, iterations, tolerance, workers) {

            this->blockSize = blockSize;
            if (blockSize == 0)
                this->blockSize = std::max(ulong(this->A.cols() / workers), (ulong) 1L);
            if (overlap == 0)
                this->overlap = blockSize/2;
            splitter();
        }


        Eigen::ColumnVector<Scalar, SIZE> solve() {

            Eigen::ColumnVector<Scalar, SIZE> oldSolution(this->solution);
            Scalar error = this->tolerance - this->tolerance;
            std::vector<std::pair<ulonglong, Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>>> inverses(blocks.size());

            Eigen::ColumnVector<Scalar, SIZE> even_solution(this->solution);
            Eigen::ColumnVector<Scalar, SIZE> odd_solution(this->solution);

            // Compute the inverses in parallel
            #pragma omp parallel for
            for (long i = 0; i < blocks.size(); ++i) {
                inverses[i] = std::pair<ulonglong, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>(i, 
					this->A.block(blocks[i].startCol, blocks[i].startRow, blocks[i].cols, blocks[i].rows).inverse());
            }
            auto nInverses = blocks.size();

//            Eigen::ColumnVector<Scalar, SIZE> buffer(this->solution);


            auto iteration = 0L;
            std::vector<int> index;


            for (iteration; iteration < this->iterations; ++iteration) {

                // Calculate the solution in parallel
                #pragma omp parallel for firstprivate(oldSolution) schedule(dynamic)
                for (int i = 0; i < inverses.size(); ++i) {

                    Eigen::ColumnVector<Scalar, Eigen::Dynamic> oldBlock = oldSolution.segment(blocks[i].startCol,
                                                                                               blocks[i].cols);

                    auto zeroBlock = oldSolution.segment(blocks[i].startCol, blocks[i].cols);

                    zeroBlock.setZero();

                    auto block = inverses[i].first%2 ? odd_solution.segment(blocks[i].startCol, blocks[i].cols) :
                                 even_solution.segment(blocks[i].startCol, blocks[i].cols);

                    block = inverses[i].second * (this->b - (this->A * oldSolution)).segment(blocks[i].startCol,
                                                                                           blocks[i].cols);

                    if ((oldBlock - block).template lpNorm<1>() / block.size() <= this->tolerance) {
                        #pragma omp critical
                        index.emplace_back(i);
                    }

                    zeroBlock = oldBlock;

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
            return Eigen::ColumnVector<Scalar, SIZE>(this->solution);
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
