#ifndef CROSS_POINTS_DETECTOR_H_
#define CROSS_POINTS_DETECTOR_H_
#include <vector>
#include <opencv2/core/core.hpp>


namespace rcll{
/* calculate how many disconnected points are around a certain point.
 * We suppose a binary image is provided where 1 is for existance, 0 is for 
 * no existence. So input image will not be checked for its type.
 */
class CrossPointDetector{
public:
    CrossPointDetector(int min_cross_point_distance);
    ~CrossPointDetector();
    /*  @the interface for all the functions. It is used to threhold the src image,
     *  @and calculate the neighbor image
     *  @input:
     *  @ input_img --- the input image
     *  @output:
     *  @ cross_points_img --- the output image which represent how many disconnected 
     *  @                      neighbor points around the center point
     */
    void Run(const cv::Mat &input_img, cv::Mat &cross_points_img, std::vector<cv::Rect> &keypoint_area);
    
    void ColorNeighborImage(const cv::Mat &neighbor_img, cv::Mat &colored_neighbor_img);
    
    
private:
    /*  @merge adjacent cross points together to form a cross points, and redefine
     *  @the type(how many lines connect to the point) by searching the around of 
     *  @the new formed points
     *  @input:
     *  @ src --- the orginal binary image
     *  @ neighbor_img --- image get from CountDisconnectPoints()
     *  @output:
     *  @ output_img --- image after merge adjacent cross points 
     *  @ keypoint_area --- where the keypoints are, and how large the are
     */
    void MergeCrossPoints(const cv::Mat &src, const cv::Mat &neighbor_img, cv::Mat &output_img, std::vector<cv::Rect> &keypoint_area);
    
    /*  @given the image around the center point, calculate the number of disconnected
     *  @neighbor points for the center point. Just walk around the boundary, and check 
     *  @every point on it
     *  @input:
     *  @ local_image --- image around the centre point 
     *  @output:
     *  @ return value --- number of disconnected points 
     */
    int CountDisconnectPoints(const cv::Mat &local_image);
    
    /*  @calculate the number of disconnected neighbor points for every point,
     *  @here, neighbor means the 8 points around the centre point 
     *  @input:
     *  @ input_img --- the input image, must be binary image where 1 is for 
     *                 road-existence, 0 is for road-no-existence
     *  @output:
     *  @ output_img --- the value of the output_img is the number of disconnected 
     *                  neighbor points
     */ 
    void CountDisconnectPointsForAllPoints(const cv::Mat &input_img, cv::Mat &output_img);
    
//     cv::Mat neighbor_img_;
//     cv::Mat src_;
    int min_cross_point_distance_;               // points within this distance are merged into one
    
};
    
}   //namespace road_match

#endif
