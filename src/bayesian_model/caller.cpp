#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "BayesianModel.h"
using namespace BayesianModelNs;
using namespace cv;
int main() {
  BayesianModel ttt("bayesianModel.bin");
  double *d1 = (double *)malloc(BVLENGTH * 8);
  double *d2 = (double *)malloc(BVLENGTH * 8);
  for (int i = 0; i < BVLENGTH; i++) {
    d2[i] = d1[i] = 10.f;
  }

  double simi = ttt.CalcSimilarity(d1, d2, BVLENGTH);
  std::cout<<simi<<std::endl;
  return 0;
}
