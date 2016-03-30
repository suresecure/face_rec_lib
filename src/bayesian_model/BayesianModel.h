#ifndef HEADER_BAYESIANMODEL
#define HEADER_BAYESIANMODEL
#include "opencv2/core/core.hpp"
using namespace  cv;

#ifndef BVLENGTH
#define BVLENGTH 160
#endif
namespace BayesianModelNs
{


	class BayesianModel
	{
	public:
		BayesianModel(const char *modelFile);
		~BayesianModel(void);
		void LoadFrom(const char* modelFile);
		double CalcSimilarity(double* feature1,double* feature2,int dimNumber);
	private:
		int _featureDim;
		Mat _PCAModel,_mapping_A,_mapping_c,_mapping_G,_mapping_mean,_mapping_Su,_mapping_Sw,_mapping_threshold;
	};

}
#endif
