#pragma once
#include "opencv2/core/core.hpp"
#include <fstream>
#include <sstream>
namespace BayesianModelNs{
	class BinaryDataReader
	{
	public:
		BinaryDataReader(void);
		~BinaryDataReader(void);
		static double* GetData(std::ifstream& fid,int size)
		{
			double* re = (double*)malloc(sizeof(double)*size);
			 fid.read((char*)re,sizeof(double)*size);
			// fid.seekg(size,std::ios::cur);
			 //fid.seekg()
			 return re;
		}
	};

}