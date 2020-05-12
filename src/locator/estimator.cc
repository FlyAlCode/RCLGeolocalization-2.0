#include "estimator.h"
#include "cross_ratio.h"
#include <opencv2/calib3d/calib3d.hpp>

// just for debug
#include <iostream>

namespace rcll {


cv::Mat HomographyEstimator::findHomography(const std::vector<cv::Point3d> &src_lines, 
                       const std::vector<cv::Point3d> &dst_lines){
    if(src_lines.size()!=dst_lines.size() || src_lines.size()<4 ||dst_lines.size()<4)
        return cv::Mat();
    
    cv::Mat A(src_lines.size()*2, 9, CV_64F);
   
    cv::Point3d X, X_;
    for(int i=0; i<src_lines.size(); i++){
        X = dst_lines[i];
        X_ = src_lines[i];
        A.at<double>(i*2, 0) = 0;
        A.at<double>(i*2, 1) = 0;
        A.at<double>(i*2, 2) = 0;
        A.at<double>(i*2, 3) = -X_.z * X.x;
        A.at<double>(i*2, 4) = -X_.z * X.y;
        A.at<double>(i*2, 5) = -X_.z * X.z;
        A.at<double>(i*2, 6) = X_.y * X.x;
        A.at<double>(i*2, 7) = X_.y * X.y;
        A.at<double>(i*2, 8) = X_.y * X.z;
      
        
        A.at<double>(i*2+1, 0) = X_.z * X.x;
        A.at<double>(i*2+1, 1) = X_.z * X.y;
        A.at<double>(i*2+1, 2) = X_.z * X.z;
        A.at<double>(i*2+1, 3) = 0;
        A.at<double>(i*2+1, 4) = 0;
        A.at<double>(i*2+1, 5) = 0;
        A.at<double>(i*2+1, 6) = -X_.x * X.x;
        A.at<double>(i*2+1, 7) = -X_.x * X.y;
        A.at<double>(i*2+1, 8) = -X_.x * X.z;   
       
    }
    
    // method one (svd)
    cv::Mat h;
    cv::SVD::solveZ(A, h);
    cv::Mat H(3, 3, CV_64F, h.data);

    cv::Mat H_inv = H.t();
    
    return H_inv/H_inv.at<double>(2,2);
    
}




}   //namespace rcll