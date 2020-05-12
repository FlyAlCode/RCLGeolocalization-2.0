#ifndef RCLL_LOCATOR_H_
#define RCLL_LOCATOR_H_

#include <memory>
#include <list>
#include "cross_point.h"
#include "sampler.h"
#include "similarity_evaluator.h"
#include "estimator.h"
#include "cross_ratio.h"
#include "cross_point_feature_creator.h"
#include "random_grid.h"
#include "consistency_checker.h"

namespace rcll{
struct LocatorParam{
    double max_sample_distance;                             // MapPtSampler
    double max_inliner_cpt_distance;                            // within this distance, a point is treated as a inliner 
    double max_inliner_voi_distance;  
    double threshold_diatance;                              // min distance for two requry_pts
    double cos_angle_distance;                              // 
    double inliner_rate_threshold;                          // min inliner rate for a success location
    double min_inliner_point_num;                           // minimun inliner points number for a success location
    int max_iterate_num;                                    // the max try num 
    int max_requry_sample_try_num;                          // max sample number for requry points
    int min_requry_pt_num;                                  // min requry points to perform location
    double max_cross_ratio_relative_error;                  // threshold for performing cross ratio check
    double max_tangent_error;                               // the max allowed tangent error for a cross point in query image to be allowed to generate a hypothesis
    
    double query_location_success_possibility;              // the possibility for finding the transformation with a sample query tuple
    double query_location_confidence;
    double location_success_confidence;                    // 
    
    double query_sample_min_diatance;
    double query_sample_max_distance;
    double query_sample_min_cos_angle_distance;

    double random_grid_size;                                // the grid size for random grid
    double random_grid_h_error_threshold;                   // the max allowed error between two homography to be regarded the same

    // icp
    double icp_inlier_threshold_;
    double icp_precision_threshold_;
    int icp_max_iter_num_;
    
    double min_inlier_rate_;
    double sample_pt_max_distance_;
};

class Locator{
public:

    void Init(const std::vector<CrossPointPtr> &reference_map, 
              VoronoiMap *ref_vor_map,
              const std::vector<cv::Point2f> & map_pts,
              const LocatorParam &param,
              const CrossPointFeatureParam &cross_feature_detector_param );

    void Init(const std::vector<CrossPointPtr> &reference_map,
              const std::string &ref_vor_map_file, 
              const std::string &shp_file,
              const LocatorParam &param,
              const CrossPointFeatureParam &cross_feature_detector_param);
    
    // return value: -1 no enough cross points; 0 fail; 1 success
    int Locate( const cv::Mat &query_road_map,
                cv::Mat &best_H,
                double &best_similarity);
    
    // used just for debug
    void SetRequryImg(const cv::Mat &img);
    void SetMapImg(const cv::Mat &img);
    const cv::Mat & GetQueryImg() {return requry_img_;}
    const cv::Mat & GetMapImg() {return map_img_;}

    inline double min_3(double x1, double x2, double x3){
        double x = x1<x2 ? x1 : x2;
        x = x<x3 ? x : x3;
        return x;
    }
    
private:
    bool IsHVaild(const cv::Mat &H, double img_w, double img_h, double threshold);

    bool InsertModel(const cv::Mat &new_model, double new_similarity);

    bool CalBestProjectCorner(const cv::Mat &H, int img_w, int img_h, cv::Point2d &corner);
    
    std::unique_ptr<HomographyEstimator> H_estimator_;
    std::unique_ptr<RequryPtSampler> requry_sampler_;
    std::unique_ptr<MapPtSampler> map_sampler_;
    std::unique_ptr<SimilarityEvaluator> similarity_evaluator_;
    std::unique_ptr<RandomGrid2D> rand_grid_;
    std::unique_ptr<ConsistencyChecker> consistency_checker_;
    
    LocatorParam locator_params_; 
    CrossPointFeatureParam cross_feature_detector_param_;
    
    std::vector<int> priority_index_;

    std::list<double> best_similaritys_;
    std::list<cv::Mat> best_models_;
    const int MODEL_NUM_ = 2;
    
    // used just for debug
    /* kind=0 --- map 
     * kind=1 --- requry 
     */
    void ShowSample(const std::string &window_name, 
                    const std::vector<CrossPointPtr> &samplers, 
                    int kind=0,
                    int wait_time = 0);
    cv::Mat requry_img_;
    cv::Mat map_img_;

    const double lamda_threshold = -0.7;
};

}   // namespace rcll

#endif
