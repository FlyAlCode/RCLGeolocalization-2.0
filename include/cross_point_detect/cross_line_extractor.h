#ifndef RCLL_CROSS_LINE_EXTRACTOR_H_
#define RCLL_CROSS_LINE_EXTRACTOR_H_
#include <opencv2/core/core.hpp>

namespace rcll{
    
class CrossLineExtractor{
public:
    CrossLineExtractor(int line_length);
    
    /* This function should be called every time you change the operating image
     */
    void SetNeighborImage(const cv::Mat &neighbor_img);
    
    bool FindAllBranches(const cv::Rect &center_area, 
                         std::vector<std::vector<cv::Point> > &branches);
private:
    /*  @used to provide start points for line segment expansion algrithom
     *  @input:
     *  @ neighbor_img --- neighbor_img get from CrossPointDetector
     *  @ kp --- the keypoint to calculate
     *  @output:
     *  @ start_rects --- start areas for line segment expansion
     */
    void FindAllStartRectsAroundKeypoint(const cv::Rect &center_area, 
                                         std::vector<cv::Rect> &start_rects);
    
    /*  @search around the centre points to find next points to expand
     *  @input:
     *  @ neighbor_img --- neighbor image after merging adjacent keypoints
     *  @ centre --- points area to search around
     *  @output:
     *  @ next_rect --- next points to expand
     */
    int ExpandToNextPoint(const cv::Rect &centre, 
                          const cv::Rect &last_centre, 
                          cv::Rect &next_rect);
    
    bool ExpandOneBranch(const cv::Rect &center,                        // the cross point
                         const cv::Rect &start,                         // the first area for the branch
                         std::vector<cv::Point> &branch );              // the expanded branch    

    int line_length_;
    cv::Mat visited_map_;                                               // used to recored the visited stated of points
    cv::Mat neighbor_img_;                                              // set by run
};

}   // namespace rcll

#endif