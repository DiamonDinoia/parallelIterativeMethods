//
// Created by mbarb on 24/01/2018.
//

#ifndef PARALLELITERATIVE_UTILS_H
#define PARALLELITERATIVE_UTILS_H

#include <chrono>
#include "Eigen"

namespace Eigen {


	template <typename Scalar, int SIZE>
	using ColumnVector = Matrix<Scalar, SIZE, 1>;

};

namespace Iterative {

	using ulong = unsigned long;
	using ulonglong = unsigned long long;

	class Index {
        public:
            explicit Index(const ulonglong startCol, const ulonglong cols, const ulonglong startRow,
						   const ulonglong rows) :
					startCol(startCol), cols(cols), startRow(startRow), rows(rows){}

			ulonglong startCol;
			ulonglong cols;
            ulonglong startRow;
            ulonglong rows;

		friend std::ostream& operator<< (std::ostream &out, const Index &index){

			out << index.startRow << ' ';
			out << index.startCol << ' ';
			out << index.cols << ' ';
			out << index.rows << ' ';

			return out << std::endl;
		}

	};

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<double> dsec;


}

class InputParser{
public:
	InputParser (int &argc, char **argv){
		for (int i=1; i < argc; ++i)
			this->tokens.push_back(std::string(argv[i]));
	}
	/// @author iain
	const std::string& getCmdOption(const std::string &option) const{
		std::vector<std::string>::const_iterator itr;
		itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
		if (itr != this->tokens.end() && ++itr != this->tokens.end()){
			return *itr;
		}
		static const std::string empty_string("");
		return empty_string;
	}
	/// @author iain
	bool cmdOptionExists(const std::string &option) const{
		return std::find(this->tokens.begin(), this->tokens.end(), option)
			   != this->tokens.end();
	}
private:
	std::vector <std::string> tokens;
};

#endif //PARALLELITERATIVE_UTILS_H
