#ifndef RCLL_CROSS_PT_TUPLE_H_
#define RCLL_CROSS_PT_TUPLE_H_
#include "cross_point.h"
#include "search_tree.h"

namespace rcll{
    
class CrossPtTuple{
public:
    CrossPtTuple() {  }
    bool Init(const std::vector< CrossPointPtr >& data);
    bool Init(const CrossPointPtr &pt1, const CrossPointPtr &pt2);
    
     /* Check whether pt1 coincide with pt2,
     * and then check whether tangents of pt1 are collinear with those of pt2 .
     * input:
     *   threshold_distance --- if the distance between the centers for the two points 
     *                           is within threshold_distance, return false.
     *   cos_angle_distance --- The cos of angle between the line passing through the two points and
     *                          the tangents of the two points should be smaller than cos_angle_distance
     */
    bool CheckValidity(const double threshold_diatance,
                       const double cos_angle_distance);
    /* Calculate cross ratio for all line combination for current tuple
     * input:
     *   line_order: 0-(clockwise,clockwise), 1-(clockwise, counterclockwise),
     *               2-(counterclockwise,clockwise), 3-(counterclockwised,counterclockwise)
     */
    void CalCrossRatios(const int line_order, std::vector<double> &cross_ratios) const;
    std::vector<CrossPointPtr> get_pts() { return tuple_data_; }
    void GetSortedTangents(int line_order, std::vector< cv::Point3d >& sorted_tangents) const;
    
    inline int hash_key() const {return hash_key_; }
    inline cv::Vec4d get_centers() const {
        return cv::Vec4d(tuple_data_[0]->get_center().x, tuple_data_[0]->get_center().y,
                         tuple_data_[1]->get_center().x, tuple_data_[1]->get_center().y); }
    
    // debug
    void Print() const;
private:
    inline int CalculateHashKey(){return ( feature_[0]-3)*27 + (feature_[1]-3)*9
                                            +(feature_[2]-2)*3 + feature_[3]-2;} 
    
    /* Sort the tangents of one point of the tuple in clockwise order
     */
    void SortTangentsFromBaseLine(const CrossPointPtr& pt, 
                                  const cv::Point2d& base_line, 
                                  std::vector< cv::Point2d >& clockwise_sorted_lines) const;
                                  
    cv::Point3d EstimateLine(const cv::Point2d &point, 
                                        const cv::Point2d &tangent) const;                              
    
    std::vector<CrossPointPtr> tuple_data_;
    int feature_[4];                // pt1_branch(3,4,5), pt2_branch, pt1_tangent(2,3,4), pt2_tangent
    int hash_key_;
    
};


typedef std::vector<CrossPtTuple> CrossPtTupleSet;
typedef std::shared_ptr<CrossPtTupleSet> CrossPtTupleSetPtr;

class CrossPtTupleHashList{
public:
    struct Param{
        double max_tuple_pts_distance;
        double min_tuple_pts_distance;
        double min_cos_angle_distance;
    };
    
    CrossPtTupleHashList();
    const CrossPtTupleSetPtr & Search(const int feature[]) const;
    const CrossPtTupleSetPtr & Search(const int hash_key) const;
    
    int SearchCrossRatioConsistencyPts(const CrossPtTuple &query_tuple,             
                                       const double cross_ratio_relative_error_threshold, 
                                       std::vector<CrossPtTupleSet> &result) const;                                   
    void Build(const std::vector<CrossPointPtr> &cross_pts, 
               const Param &params,
               bool build_search_tree = true, 
               bool allow_symmetric = true);            // 是否允许一对点组成两个tuple
    inline std::vector<int> get_priority_index() const { return priority_index_;}
    
    int get_tuple_num(const int hash_key) const; 
    
    void Print() const;
    
private:
    void BuildSearchTree(const CrossPtTupleSetPtr &pts, 
                         SearchTree &tree);
    int SearchCrossRatioConsistencyPts(const int hash_key,
                                       std::vector<double> cross_ratios,               
                                       const double cross_ratio_distance_threshold, 
                                       CrossPtTupleSet &result) const;
    void BuildAllSearchTrees();
    void CalSamplePriority();
    std::vector<CrossPtTupleSetPtr> data_;
    std::vector<SearchTree> cross_ratio_search_trees_;
    std::vector<int> priority_index_;
    // std::vector<CrossPointPtr> raw_cross_points_;
    const int SIZE = 81;
};


}   // namespace rcll

#endif
