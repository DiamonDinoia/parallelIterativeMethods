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
                const ulong workers,
                const ulonglong blockSize = 0L) :
                jacobi<Scalar,SIZE>::jacobi(matrix, vector, iterations, tolerance, workers) {

            this->blockSize = blockSize;
            if (blockSize == 0)
                this->blockSize = std::max(ulong(this->matrix.cols() / workers), (ulong) 1L);
            splitter();
        }


        Eigen::ColumnVector<Scalar, SIZE> solve() {
            Eigen::ColumnVector<Scalar, SIZE> old_solution(this->solution);
            Scalar error = this->tolerance - this->tolerance;
            std::vector<Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>> inverses (blocks.size());


            #pragma omp parallel for
            for (int i = 0; i < blocks.size(); ++i) {
                inverses[i] = this->matrix.block(blocks[i]->startCol, blocks[i]->startRow, blocks[i]->cols,
                                                 blocks[i]->rows).inverse();
            }

            Eigen::ColumnVector<Scalar, SIZE> buffer(this->solution);

            for (auto iteration = 0; iteration < this->iterations; ++iteration) {


                #pragma omp parallel for firstprivate(buffer)
                for (int i = 0; i < inverses.size(); ++i) {
                    buffer.segment(blocks[i]->startCol, blocks[i]->cols).setZero();
                    auto block = this->solution.segment(blocks[i]->startCol,blocks[i]->cols);
                    block = inverses[i]*(this->vector-(this->matrix*buffer)).segment(blocks[i]->startCol,
                                                                                     blocks[i]->cols);

                }

                //compute the error
                error += (this->solution - old_solution).template lpNorm<1>();
                // check the error
                error /= this->solution.size();
                if (error <= this->tolerance) break;

                swap(this->solution, old_solution);
                buffer = this->solution;

            }
            return Eigen::ColumnVector<Scalar, SIZE>(this->solution);
        }

    protected:
        ulonglong blockSize;
        std::vector<Index*> blocks;

        void splitter() {
            for (ulonglong i = 0; i < this->matrix.cols(); i += blockSize) {
                blocks.emplace_back(new Index(i, std::min(blockSize, (ulonglong) this->matrix.cols()),
                                              i, std::min(blockSize, (ulonglong) this->matrix.rows())));
//                std::cout << "block: " << i << ' ' << blockSize << '\n';
            }
        }


    private:


    };

}


#endif //PARALLELITERATIVE_DENSEOVERLAPPINGJACOBI_H
