//
// Created by mbarb on 05/02/2018.
//

#ifndef PARALLELITERATIVE_DENSEOVERLAPPINGJACOBI_H
#define PARALLELITERATIVE_DENSEOVERLAPPINGJACOBI_H



#include "Eigen"
#include "utils.h"
#include "jacobi.h"
#include <typeinfo>


namespace Iterative {

    template <typename Scalar, long long SIZE>
    class denseOverlappingJacobi : public jacobi<Scalar, SIZE> {
    public:

        explicit denseOverlappingJacobi(
                const Eigen::Matrix<Scalar, SIZE, SIZE>& matrix,
                const Eigen::ColumnVector<Scalar, SIZE>& vector,
                const ulonglong iterations,
                const Scalar tolerance,
                const ulong workers=0L,
                const ulonglong blockSize = 0L,
                const ulonglong overlap = 0L) :
                jacobi<Scalar,SIZE>::jacobi(matrix, vector, iterations, tolerance, workers) {

            this->blockSize = blockSize;
            if (blockSize == 0)
                this->blockSize = std::max(ulong(this->matrix.cols() / workers), (ulong) 1L);
            if (overlap == 0)
                this->overlap = blockSize/2;
            splitter();
        }


        Eigen::ColumnVector<Scalar, SIZE> solve() {
            Eigen::ColumnVector<Scalar, SIZE> old_solution(this->solution);
            Scalar error = this->tolerance - this->tolerance;
            std::vector<Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>> inverses (blocks.size());

            Eigen::ColumnVector<Scalar, SIZE> even_solution(this->solution);
            Eigen::ColumnVector<Scalar, SIZE> odd_solution(this->solution);


            // Compute the inverses in parallel
            #pragma omp parallel for
            for (int i = 0; i < blocks.size(); ++i) {
                inverses[i] = this->matrix.block(blocks[i]->startCol, blocks[i]->startRow, blocks[i]->cols,
                                                 blocks[i]->rows).inverse();
            }

            Eigen::ColumnVector<Scalar, SIZE> buffer(this->solution);

            for (auto iteration = 0; iteration < this->iterations; ++iteration) {

                // Calculate the solution in parallel
                #pragma omp parallel for firstprivate(buffer)
                for (int i = 0; i < inverses.size(); ++i) {

                    buffer.segment(blocks[i]->startCol, blocks[i]->cols).setZero();

                    // the odd indexes updates the odd vector and the even updates the even vector
                    if (i%2) {
                        auto block = odd_solution.segment(blocks[i]->startCol, blocks[i]->cols);
                        block = inverses[i] * (this->vector - (this->matrix * buffer)).segment(blocks[i]->startCol,
                                                                                               blocks[i]->cols);
                    } else {
                        auto block = even_solution.segment(blocks[i]->startCol,blocks[i]->cols);
                        block = inverses[i] * (this->vector - (this->matrix * buffer)).segment(blocks[i]->startCol,
                                                                                               blocks[i]->cols);

                    }

                }

                // average of the two values
                this->solution = (even_solution + odd_solution)/(Scalar)2.;

                // not overlapping portion of the solution vector
                this->solution.head(overlap) = even_solution.head(overlap);

                // not overlapping end portion of the solution vector
                this->solution.tail(overlap) = inverses.size()%2 ?
                                               even_solution.tail(overlap) : odd_solution.tail(overlap);


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

        ulonglong overlap;

        void splitter() {
            for (ulonglong i = 0; i < this->matrix.cols()-overlap; i += (blockSize-overlap))
                blocks.emplace_back(new Index(i, std::min(blockSize, (ulonglong) this->matrix.cols() - i),
                                              i, std::min(blockSize, (ulonglong) this->matrix.rows() - i)));

//            for (auto &block : blocks) std::cout << *block;
        }


    private:


    };

}


#endif //PARALLELITERATIVE_DENSEOVERLAPPINGJACOBI_H
