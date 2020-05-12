#ifndef RCLL_CROSS_RATIO_H_
#define RCLL_CROSS_RATIO_H_

#include <opencv2/core/core.hpp>

namespace rcll{
    
/* pts ---nonhomogeneous coordination, the collineation will not be checked
 */
double CalCrossRatioFromPoints(const std::vector<cv::Point2d> &pts);

/* lines --- express in (a, b, c)
 */
double CalCrossRatioFromLines(const std::vector<cv::Point3d> &lines);

/* normal_tangents --- the normalized tangents 
 */
double CalCrossRatioFromNormalTangents(const std::vector< cv::Point2d >& normal_tangents);

void CalAllCrossRatios(const std::vector< cv::Point2d >& normal_tangents, 
                         std::vector<double> &cross_ratios);

/* Given the directions of some lines, performing a cross ratio check to 
 * test whether the line pairs are equal under a homograpyh transformation.
 * 1. If the number of lines passing through a line is less than 4, which means
 * there are enough lines to calculate a cross ratio, we just return true.
 * 2. When abs(cross_ratio1 - cross_ratio2)/cross_ratio1 < threshold, return true, 
 * else return false. 
 * 3. When more than 4 lines exist for each point, every possible combination of 
 *    4 lines will be used to perform the check.
 * 
 * PS: Here, we suppose tangents in x and y are normalized
 */
bool CheckCrossRatioConsistency(const std::vector<cv::Point2d> &x, 
                                const std::vector<cv::Point2d> &y, 
                                const double threshold);

/* Find all the possible combination of m data for the total n data*/
void Combine(int data[],
             int n,
             int m,
             int temp[],
             const int M,
             std::vector<std::vector<int> > &vec_res);

}   // namespace rcll

#endif
