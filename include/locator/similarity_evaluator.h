#ifndef RCLL_SIMILARITY_EVALUATOR_H_
#define RCLL_SIMILARITY_EVALUATOR_H_

#include <memory>
#include <vector>
#include <opencv2/core/core.hpp>

#include "voronoi_map.h"

#include "cross_point.h"
#include "cross_point_tree.h"

namespace rcll{
class SimilarityEvaluator{
public:
    /* Initial the similarity with the voronoi surface map
     */
    void Init(VoronoiMap *ref_vor_map, const std::vector<CrossPointPtr> &reference_pts);
    /* Initial the similarity with the voronoi surface map reading from the ref_vor_map_file
     */
    void Init(const std::string &ref_vor_map_file, const std::vector<CrossPointPtr> &reference_pts);


    /* Evaluate the similarity between query road map and reference road map under 
     * given transformation models, and return similarity.
     * input:
     *   query_img --- the binary query road map(a thined image is performed)
     *   threshold --- a max distance for a points to regard as a inliner
     */
    double Evaluate(const cv::Mat &query_binary_img,               // query road map     
                    const cv::Mat H,                        // 
                    const double threshold);                // points within such threshld are treated as inlier

    double InverseVorEvaluate(const cv::Mat &query_vor_img,     // query road map     
                    const cv::Mat H,                        // 
                    const double threshold);                // points within such threshld are treated as inlier

    double Evaluate(const std::vector<CrossPointPtr> &requry_points,              
                    const cv::Mat &H,                        
                    double threshold);                // points within such threshld are treated as inlier

    double Evaluate(const std::vector<CrossPointPtr> &requry_points,              
                    const cv::Mat &H, 
                    int img_w, int img_h,                       
                    double threshold);                // points within such threshld are treated as inlier
    double EvaluateSum(const cv::Mat &query_binary_img,
                       const cv::Mat &query_vor_img,
                       const std::vector<CrossPointPtr> &requry_points,  
                       const cv::Mat &H,
                       double cpt_inlier_distance_threshold,
                       double vor_inlier_distance_threshold, 
                       double half_inlier_rate_threshold);

    inline cv::Vec4f get_area_bound() const {return ref_vor_map_->get_geo_bound();}
    

    const int VALIDATE_WITH_INTRESECTION = 40;
private:
    /* 1. Transform the requry_point to destination coordinate with H
     * 2. Search for points within threshold using search_tree_
     * 3. Check whether a point with the same type as requry_point exists
     */
    bool IsInliner(const CrossPointPtr &requry_point, 
                   const cv::Mat &H, 
                   const double threshold, 
                   CrossPointPtr &matching_point);
    bool IsInsidePoint(const std::vector<cv::Point2d> &corners,
                        const cv::Point2d &query_pt);
    bool GetProjectArea(const cv::Mat &H, 
                        int img_w, int img_h,
                        std::vector<cv::Point2d> &map_corners,
                        cv::Vec4d &bound);
    
    std::unique_ptr<RoadMapTree> search_tree_;
    std::vector<CrossPointPtr> cross_points_;

    std::shared_ptr<VoronoiMap> ref_vor_map_;
    const int down_sample_rate = 4;
};    
    

}   // namespace rcll

#endif