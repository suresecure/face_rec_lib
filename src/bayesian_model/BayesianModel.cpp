#include "BayesianModel.h"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include <fstream>
#include <sstream>
#include "BinaryDataReader.h"
#include "BinaryMatAdapter.h"
using namespace BayesianModelNs;
using namespace cv;


BayesianModelNs::BayesianModel::BayesianModel(const char *modelFile)
{
	this->LoadFrom(modelFile);

}



BayesianModel::~BayesianModel(void)
{
	int num = 8;
	Mat a[]={_PCAModel,_mapping_A,_mapping_c,_mapping_G,_mapping_mean,_mapping_Su,_mapping_Sw,_mapping_threshold};
	for (int i=0;i<num;i++)
	{
		a[i].release();
	}

}

double BayesianModelNs::BayesianModel::CalcSimilarity(double* feature1,double* feature2,int dimNumber)
{
	Mat feature1_mat = Mat(1,dimNumber,CV_64FC1,feature1);
	Mat feature2_mat = Mat(1,dimNumber,CV_64FC1,feature2);
	Mat f1norm = feature1_mat-this->_mapping_mean;
	Mat f2norm = feature2_mat-this->_mapping_mean;
	f1norm = f1norm*this->_PCAModel.t();
	f2norm = f2norm*this->_PCAModel.t();
	
	//Mat t1=(f1norm-f2norm)*this->_mapping_A*(f1norm-f2norm).t();
	//Mat t2 = 2*f1norm*(this->_mapping_A-this->_mapping_G)*f2norm.t();

	Mat similarityMat =  (f1norm-f2norm)*this->_mapping_A*(f1norm-f2norm).t()+2*f1norm*(this->_mapping_A-this->_mapping_G)*f2norm.t();
	double similarity = similarityMat.at<double>(0);
	return similarity;
	

}

void BayesianModelNs::BayesianModel::LoadFrom(const char* modelFile)
{
	std::ifstream sf_model(modelFile,std::ios::binary);
	if (!sf_model){
		printf("%s read failed.",modelFile);
		return;
	}
	sf_model.seekg(0);
	double *pcaModel,*mappingA,*mappingG,*mappingMean,*mappingSu,*mappingSw,*mappingThreshold;
	//double *a[] = {pcaModel,mappingA,mappingG,mappingMean,mappingSu,mappingSw,mappingThreshold};
	pcaModel= BinaryDataReader::GetData(sf_model,BVLENGTH*BVLENGTH);
	mappingA = BinaryDataReader::GetData(sf_model,BVLENGTH*BVLENGTH);
	mappingG = BinaryDataReader::GetData(sf_model,BVLENGTH*BVLENGTH);
	mappingMean = BinaryDataReader::GetData(sf_model,1*BVLENGTH);
	mappingSu =  BinaryDataReader::GetData(sf_model,BVLENGTH*BVLENGTH);
	mappingSw = BinaryDataReader::GetData(sf_model,BVLENGTH*BVLENGTH);
	mappingThreshold = BinaryDataReader::GetData(sf_model,1*1);

	this->_PCAModel = BinaryMatAdapter::GetCvMat(pcaModel,BVLENGTH,BVLENGTH);
	this->_mapping_A= BinaryMatAdapter::GetCvMat(mappingA,BVLENGTH,BVLENGTH);
	this->_mapping_G= BinaryMatAdapter::GetCvMat(mappingG,BVLENGTH,BVLENGTH);
	this->_mapping_mean= BinaryMatAdapter::GetCvMat(mappingMean,1,BVLENGTH);
	this->_mapping_Su= BinaryMatAdapter::GetCvMat(mappingSu,BVLENGTH,BVLENGTH);
	this->_mapping_Sw= BinaryMatAdapter::GetCvMat(mappingSw,BVLENGTH,BVLENGTH);
	this->_mapping_threshold= BinaryMatAdapter::GetCvMat(mappingA,1,1);

	//FileStorage fs("mat.yml", FileStorage::WRITE);
	//fs << "pcamodel" << this->_PCAModel;
	//fs << "mapping_a" << this->_mapping_A;
	//fs << "mapping_g" << this->_mapping_G;
	//fs << "mapping_mean" << this->_mapping_mean;
	//fs << "mapping_su" << this->_mapping_Su;
	//fs << "mapping_sw" << this->_mapping_Sw;
	//fs << "mapping_threshold" << this->_mapping_threshold;
	//fs.release();

	sf_model.close();

}
