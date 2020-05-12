#ifndef RCLL_ICP_H_
#define RCLL_ICP_H_

#include <vector>
#include <math.h>
#include <time.h>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/flann.hpp>

namespace rcll{


class ICPOptimizer{

public:
    void SetParams(double inlier_threshold,
                   double precision_threshold,
                   int max_iter_num);

    
    void Init(const std::vector<cv::Point2f> &ref_pts,
              double inlier_threshold,
              double precision_threshold,
              int max_iter_num);

   
    bool Run(const std::vector<cv::Point2f> &query_pts,
             const cv::Point2f &query_anchor_pt,
             const cv::Mat &initial_h,
             cv::Mat &result_h,
             double &inlier_rate);

   
    void GetMatches(const std::vector<cv::Point2f> &query_pts,
                    const cv::Mat &h,                                       
					double inlier_threshold,
                    std::vector<cv::Point2d> &match_ref_pts,
                    std::vector<cv::Point2d> &match_query_pts);

    double GetInlierRatio(const std::vector<cv::Point2f> &query_pts,
                        const cv::Mat &h, 
                        double inlier_threshold);

    double GetAverageMatchError(const std::vector<cv::Point2f> &query_pts,
                                const cv::Mat &h,
                                double inlier_rate = 0.7);

private:
    void GetMatches(const std::vector<cv::Point2f> &query_pts,
                    const std::vector<double> &weights,
                    const cv::Mat &h,                                       
					double inlier_threshold,
                    std::vector<cv::Point2d> &match_ref_pts,
                    std::vector<cv::Point2d> &match_query_pts,
                    std::vector<double> &match_weights);
    // params
    double inlier_threshold_;                               
    double precision_threshold_;                            
    int max_iter_num_;                                      

    // 中间变量
    std::vector<cv::Point2f> ref_pts_;
    cv::Mat ref_pts_mat_;
    std::unique_ptr<cv::flann::Index> ref_pt_index_;
};

}

#endif

