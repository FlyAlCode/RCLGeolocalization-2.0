#ifndef RCLL_CROSS_POINT_FEATURE_CREATOR_H_
#define RCLL_CROSS_POINT_FEATURE_CREATOR_H_

#include <opencv2/core/core.hpp>
#include "cross_points_detector.h"
#include "cross_line_extractor.h"
#include "cross_point.h"

namespace rcll{
struct CrossPointFeatureParam{
    int threshold;                                  // the threshold for binary image
    int min_cross_point_distance;                   // used to merge points within this distance
    int branch_length;                              // how many pixels should be contained in one branch
    double merge_angle_threshold;
};

/* Given a road image, convert it into a thined binary image,
 * and then detect all cross points on it, and then extract tangent line 
 * for every cross point to from a cross point feature 
 * input:
 *   input_img --- the image to detect
 *   param --- some params for the function
 * output:
 *   cross_points --- the created cross point features
 */ 
bool CreateCrossPointFeatures(const cv::Mat &input_img,
                             const CrossPointFeatureParam &param,
                             std::vector<CrossPointPtr> &cross_points, 
                             cv::Mat &thined_img,
                             bool img_thined = false );

/*  @fill points whose left and right, or up and down is 1
 *  @input:
 *  @ input_img --- must be binary image
 *  @output:
 *  @ output_img
 */
void FillImageHole(const cv::Mat &input_img, cv::Mat &output_img);

/* Find the skeleton image for the given image
 * input:
 *   src --- the image to be thined
 * output 
 *   thined_img --- the thined image
 */
void ThinImage(const cv::Mat & src,
               cv::Mat &thined_img,
               const int maxIterations = -1);

// debug
void ShowCrossDetectorResult(const cv::Mat &img,
                             std::vector<CrossPointPtr> &cross_points);

}   // namespace rcll


#endif
