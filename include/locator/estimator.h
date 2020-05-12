#ifndef RCLL_ESTIMATOR_H_
#define RCLL_ESTIMATOR_H_
#include <vector>
#include "cross_point.h"

namespace rcll{
    
class HomographyEstimator{
public:
 /* Given two sets of cross point with ordered tangents, calculate all the 
 * possible homography transformation between them. If more than 4 tangents 
 * correspondence exist, a ransac method will be perform.
 * input:
 *   src_pt/dst_pt --- here we suppose only two cross points contained in them
 * output:
 *   HMats --- all the possible homography transformation
 */
// bool EstimatorHomographyFromCrossLine(const std::vector<CrossPointPtr> &src_pt, 
//                                       const std::vector<CrossPointPtr> &dst_pt, 
//                                       std::vector<cv::Mat> &HMats); 

/* Given matching line pairs, estimation the homography transformation
 */
cv::Mat findHomography(const std::vector<cv::Point3d> &src_lines, 
                       const std::vector<cv::Point3d> &dst_lines);  

    
};  // class HomographyEstimator



}   //namespace rcll

#endif