//
// Created by mbarb on 14/02/2018.
//

#ifndef PARALLELITERATIVE_ASYNCJACOBI_H
#define PARALLELITERATIVE_ASYNCJACOBI_H




#include "Eigen"
#include "utils.h"
#include "jacobi.h"


namespace Iterative {

    template <typename Scalar, long long SIZE>
    class asyncBlocksJacobi : public jacobi<Scalar, SIZE> {

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
                const ulonglong blockSize = 0L,
                const unsigned int async = 4):
                jacobi<Scalar,SIZE>::jacobi(matrix, vector, iterations, tolerance, workers), async(async) {

            this->blockSize = blockSize;

            if (blockSize == 0)
                this->blockSize = std::max(ulong(this->A.cols() / workers), (ulong) 1L);
            splitter();
        }


        Eigen::ColumnVector<Scalar, SIZE> solve() {

            Eigen::ColumnVector<Scalar, SIZE> old_solution(this->solution);
            std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> inverses (blocks.size());

            // compute the inverses of the blocks and memorize it
            #pragma omp parallel for
            for (int i = 0; i < blocks.size(); ++i) {
                inverses[i] = this->A.block(blocks[i].startCol, blocks[i].startRow, blocks[i].cols,
                                            blocks[i].rows).inverse();
            }

            // start iterations
            auto iteration = 0L;
            std::vector<int> index;

            for (iteration; iteration < this->iterations; iteration+=async) {

                #pragma omp parallel for firstprivate(old_solution) schedule(dynamic)
                for (int i = 0; i < inverses.size(); i++) {

                    // set zero the components of the solution b that corresponds to the inverse
                    Eigen::ColumnVector<Scalar,Eigen::Dynamic> oldBlock = old_solution.segment(blocks[i].startCol,
                                                                                               blocks[i].cols);

                    old_solution.segment(blocks[i].startCol, blocks[i].cols).setZero();

                    for (int async_step = 0; async_step < async; ++async_step) {
                        // the segment of the solution b that this inverse approximates
                        auto block = this->solution.segment(blocks[i].startCol, blocks[i].cols);
                        // approximate the solution using the inverse and the solution at the previous iteration
                        block = inverses[i]*(this->b-(this->A*old_solution)).segment(blocks[i].startCol,
                                                                                               blocks[i].cols);

                        if((oldBlock-block).template lpNorm<1>()/block.size()<=this->tolerance) {
                            #pragma omp critical
                            index.emplace_back(i);
                        }

                    }

                }

                if(!index.empty()) {
                    std::sort(index.rbegin(), index.rend());
                    for (auto i: index) {
                        blocks.erase(blocks.begin() + i);
                        inverses.erase(inverses.begin() + i);
                    }
                    index.clear();
                    if (inverses.empty()) break;
                }

//                old_solution=this->solution;
                std::swap(this->solution, old_solution);
            }
            std::cout << iteration << std::endl;
            return Eigen::ColumnVector<Scalar, SIZE>(this->solution);
        }

    protected:
        ulonglong blockSize;
        std::vector<Index> blocks;
        const unsigned int async;

        void splitter() {
            for (ulonglong i = 0; i < this->A.cols(); i += blockSize) {
                blocks.emplace_back(Index(i, std::min(blockSize, (ulonglong) this->A.cols() - i),
                                          i, std::min(blockSize, (ulonglong) this->A.rows() - i)));
            }
        }


    private:
        template<typename Cont, typename It>
        auto ToggleIndices(Cont &cont, It beg, It end) -> decltype(std::end(cont))
        {
            int helpIndx(0);
            return std::stable_partition(std::begin(cont), std::end(cont),
                                         [&](typename Cont::value_type const& val) -> bool {
                                             return std::find(beg, end, helpIndx++) != end;
                                         });
        }

    };

}



#endif //PARALLELITERATIVE_ASYNCJACOBI_H
