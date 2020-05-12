#ifndef RCLL_HOMOGRAPHY_H_
#define RCLL_HOMOGRAPHY_H_

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

namespace rcll{


    double CalHomography(const std::vector<cv::Point2d> &src_points,
                         const std::vector<cv::Point2d> &dst_points,
                         cv::Mat &homography_mat );
    
    /*  @Calculate a homography transform, so that
     *  @[dst_points; 1] = k*H*[src_points; 1]
     *  @Here, "findHomography" in opencv is used to calculate the mat,
     *  @and we just wrap it to calculate the residual
     *  @input: 
     *  @ src_points/dst_points --- inhomogeneous point
     *  @output:
     *  @ homography_mat --- the transform mat 
     *  @return:
     *  @ -1 --- if the method fail, eg. no enough points are provided
     *  @ residual --- the total transform residual
     */
    double CalHomographyOpencv(const std::vector<cv::Point2d> &src_points,
                               const std::vector<cv::Point2d> &dst_points,
                               cv::Mat &homography_mat );
}  //  namespace rcll

#endif
