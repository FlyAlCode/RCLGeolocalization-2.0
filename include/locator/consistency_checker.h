#ifndef RCLL_CONSISTENCY_CHECKER_H_
#define RCLL_CONSISTENCY_CHECKER_H_
#include <string>
#include <vector>
#include <memory>
#include <opencv2/core/core.hpp>
#include "icp.h"

namespace rcll {
class ConsistencyChecker{
public:
    bool Init(const std::string &shp_file,
              double icp_inlier_threshold,
              double icp_precision_threshold,
              int icp_max_iter_num,
              double sample_pt_max_distance,
              double min_inlier_rate);
    
    bool Init(const std::vector<cv::Point2f> &map_pts,
              double icp_inlier_threshold,
              double icp_precision_threshold,
              int icp_max_iter_num,
              double sample_pt_max_distance,
              double min_inlier_rate);

    bool Run(const std::vector<cv::Point2f> &query_pts,
             const cv::Mat &initial_h,
             cv::Mat &final_h,
             double &final_inlier_ratio);

    bool Run(const cv::Mat &binary_img,
             const cv::Mat &initial_h,
             cv::Mat &final_h,
             double &final_inlier_ratio);

    bool BootstrapRun(const std::vector<cv::Point2f> &query_pts,
             const cv::Point2f &query_anchor_pt,
             float max_distance,                                     
             const cv::Mat &initial_h,
             cv::Mat &final_h,
             double &final_inlier_ratio);

    bool BootstrapRun(const cv::Mat &query_img,
             const cv::Point2f &query_anchor_pt,    
             const cv::Mat &initial_h,
             cv::Mat &final_h,
             double &final_inlier_ratio);

    double GetInlierRatio(const std::vector<cv::Point2f> &query_pts,
                          const cv::Mat &h,
                          double threshold);

    double GetAverageMatchDistance(const std::vector<cv::Point2f> &query_pts,
                          const cv::Mat &h,
                          double inlier_rate = 0.7);                     

private:
    bool GetRefPointsFromShpFile(const std::string &shp_file);

    // icp params
    double icp_inlier_threshold_;
    double icp_precision_threshold_;
    int icp_max_iter_num_;
    
    double min_inlier_rate_;
    double sample_pt_max_distance_;

    std::vector<cv::Point2f> ref_pts_;
    std::unique_ptr<ICPOptimizer> icp_optimizer_;
};
} // namespace rcll

#endif