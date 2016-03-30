#ifndef HEADER_BINARYMATADAPTER
#define HEADER_BINARYMATADAPTER
#include "opencv2/core/core.hpp"
namespace BayesianModelNs{
	class BinaryMatAdapter
	{
	public:
		BinaryMatAdapter(void);
		~BinaryMatAdapter(void);
		static Mat GetCvMat(double* datum,int rows,int cols)
		{
			return Mat(rows,cols,CV_64FC1,datum);
		}
	};

}
#endif
