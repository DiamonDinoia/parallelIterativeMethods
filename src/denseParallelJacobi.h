//
// Created by mbarb on 24/01/2018.
//

#ifndef PARALLELITERATIVE_DENSEPARALLELJACOBI_H
#define PARALLELITERATIVE_DENSEPARALLELJACOBI_H

#include <omp.h>

namespace Iterative {

    template <typename Scalar, long long SIZE>
    class denseParallelJacobi {

    public:
        /**
         *
         * @param A linear system matrix of max rank
         * @param b known terms vector
         * @param iterations  max number of iterations
         * @param tolerance min error tolerated
         * @param workers  number of threads
         */
        explicit denseParallelJacobi(
                const Eigen::Matrix<Scalar, SIZE, SIZE>& A,
                const Eigen::ColumnVector<Scalar, SIZE>& b,
                const ulonglong iterations,
                const Scalar tolerance,
                const ulong workers=0L) :
                A(A), b(b), iterations(iterations), tolerance(tolerance),
                workers(workers), solution(b) {

            solution.fill((Scalar)1/solution.size());
            omp_set_num_threads(workers);
        }

        const Eigen::ColumnVector<Scalar, SIZE> solve() {


            Eigen::ColumnVector<Scalar, SIZE> oldSolution(solution);

            std::vector<ulonglong> index(solution.size());

            for (ulonglong i = 0; i < solution.size(); ++i)
                index[i]=i;

            std::vector<ulonglong> remove;

            auto iteration = 0L;

            for (iteration = 0; iteration < iterations; ++iteration) {
                //calculate solutions parallelizing on rows
                #pragma omp parallel for schedule(dynamic)
                for (auto i = 0; i < index.size(); ++i){
                    auto el = index[i];
                    solution[el] = solution_find(b[el], el, oldSolution);
                    Scalar error = std::abs(solution[el]-oldSolution[el]);
                    if(error <= tolerance){
                        #pragma omp critical
                        remove.emplace_back(i);
                    }
                }

                if(!remove.empty()){
                    std::sort(remove.rbegin(), remove.rend());
                    for (auto i : remove) {
                        index.erase(index.begin() + i);
                    }
                    remove.clear();
                    if (index.empty()) break;
                }

                std::swap(solution, oldSolution);
            }
            std::cout << iteration << std::endl;
            return this->solution;
        }

        const Eigen::ColumnVector<Scalar, SIZE> &getSolution() const {
            return solution;
        }


    protected:

        const Eigen::Matrix<Scalar, SIZE, SIZE>& A;
        const Eigen::ColumnVector<Scalar, SIZE>& b;


        const ulonglong iterations;
        const Scalar tolerance;
        const ulong workers;

        Eigen::ColumnVector<Scalar, SIZE> solution;

    private:
        /**
        * utility function implementing the jacobi method in order to find one solution
        * @param row coeffiient row
        * @param solutions vector solution
        * @param term right term vector
        * @param index index of the solution
        * @return solution component
        */
        inline Scalar solution_find(Scalar term, const ulonglong index, Eigen::ColumnVector<Scalar,SIZE>& oldSolution) {
            term -= A.row(index) * oldSolution;
            return (term + A(index, index) * oldSolution[index]) / A(index, index);
        }

    };
};


#endif //PARALLELITERATIVE_JACOBI_H
