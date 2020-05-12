#include "homography.h"
#include "glog/logging.h"

namespace rcll{
    
double CalHomography(const std::vector<cv::Point2d> &src_points,
                     const std::vector<cv::Point2d> &dst_points,
                     cv::Mat &homography_mat ){
    if(src_points.size()!=dst_points.size() || src_points.size()<4){
        return -1;
    }
    
    int N = src_points.size();
    cv::Mat A(N*2, 8, CV_64FC1);
    cv::Mat b(N*2, 1, CV_64FC1);
    
    for(int i=0; i<N; i++){
        A.at<double>(i, 0) = src_points[i].x;
        A.at<double>(i, 1) = src_points[i].y;
        A.at<double>(i, 2) = 1;
        A.at<double>(i, 3) = 0;
        A.at<double>(i, 4) = 0;
        A.at<double>(i, 5) = 0;
        A.at<double>(i, 6) = -src_points[i].x * dst_points[i].x;
        A.at<double>(i, 7) = -src_points[i].y * dst_points[i].x;
        b.at<double>(i, 0) = dst_points[i].x;
        
        A.at<double>(i+N, 0) = 0;
        A.at<double>(i+N, 1) = 0;
        A.at<double>(i+N, 2) = 0;
        A.at<double>(i+N, 3) = src_points[i].x;
        A.at<double>(i+N, 4) = src_points[i].y;
        A.at<double>(i+N, 5) = 1;
        A.at<double>(i+N, 6) = -src_points[i].x * dst_points[i].y;
        A.at<double>(i+N, 7) = -src_points[i].y * dst_points[i].y;
        b.at<double>(i+N, 0) = dst_points[i].y;
    }
    
    cv::Mat h_vector_tmp;
    cv::solve(A, b, h_vector_tmp, cv::DECOMP_SVD);
    
    cv::Mat h_mat_tmp(3, 3, CV_64FC1);
    memcpy(h_mat_tmp.data, h_vector_tmp.data, sizeof(double)*8);
    h_mat_tmp.at<double>(2, 2) = 1;
    h_mat_tmp.copyTo(homography_mat);
    
    //calculate the residual
    double residual = 0;
    cv::Mat src_pt(3,1, CV_64FC1);
    cv::Mat dst_pt(3,1, CV_64FC1);
    for(int i=0; i<src_points.size(); i++){
        src_pt.at<double>(0,0) = src_points[i].x;
        src_pt.at<double>(1,0) = src_points[i].y;
        src_pt.at<double>(2,0) = 1;
         
        dst_pt = h_mat_tmp * src_pt;
        double x_tmp = dst_pt.at<double>(0,0)/dst_pt.at<double>(2,0);
        double y_tmp = dst_pt.at<double>(1,0)/dst_pt.at<double>(2,0);
        residual += (dst_points[i].x - x_tmp)*(dst_points[i].x - x_tmp)
                   +(dst_points[i].y - y_tmp)*(dst_points[i].y - y_tmp);
    }
    return residual;
}


double CalHomographyOpencv(const std::vector<cv::Point2d> &src_points,
                           const std::vector<cv::Point2d> &dst_points,
                           cv::Mat &homography_mat ){
    if( (src_points.size()!=dst_points.size()) || src_points.size()<4){
        return -1;
    }
    
    cv::Mat h_mat_tmp = cv::findHomography( src_points, dst_points, cv::RANSAC,  1);
    double residual = 0;
    cv::Mat src_pt(3,1, CV_64FC1);
    cv::Mat dst_pt(3,1, CV_64FC1);
    for(int i=0; i<src_points.size(); i++){
        src_pt.at<double>(0,0) = src_points[i].x;
        src_pt.at<double>(1,0) = src_points[i].y;
        src_pt.at<double>(2,0) = 1;
         
        dst_pt = h_mat_tmp * src_pt;
        double x_tmp = dst_pt.at<double>(0,0)/dst_pt.at<double>(2,0);
        double y_tmp = dst_pt.at<double>(1,0)/dst_pt.at<double>(2,0);
        double error_x = dst_points[i].x - x_tmp;
        double error_y = dst_points[i].y - y_tmp;
        residual += error_x * error_x + error_y * error_y;
    }
    h_mat_tmp.copyTo(homography_mat);
    return residual;
}

}   // namespace rcll
