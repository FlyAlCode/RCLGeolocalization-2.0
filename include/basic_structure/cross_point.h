#ifndef RCLL_CROSS_POINT_H_
#define RCLL_CROSS_POINT_H_
#include <vector>
#include <memory>
#include <ostream>
#include <opencv2/core/core.hpp>

namespace rcll{

class CrossPoint{
    
    friend std::ostream & operator << (std::ostream &os, const CrossPoint &pt); 
public:
    bool FillData(const cv::Rect &center_area, 
                  const std::vector<std::vector<cv::Point> > &branches, 
                  double merge_angle_threshold);
    void Draw(cv::Mat &draw_img, const cv::Scalar &color);
    
    void ThinInit(const cv::Point2d &center,                        // here, "thin" means only center/branch number/tangent number/tangents are initilized 
                  const int branch_num, 
                  const std::vector<cv::Point2d> &tangents,         // tangents are normalized   
                  const double tangent_error = 0);                  // max average error for tangents estimation
    
    CrossPoint() { id_ = ID_++; tangents_estimate_error_ = 0;}
    inline cv::Point2d get_center() {return center_;}
    inline const std::vector<cv::Point2d> &get_tangents() const{ return tangents_; }
    inline int get_tangents_num() { return tangents_.size(); }
    inline int get_braches_num() { return branches_num_; }
    inline int get_id() { return id_; }
    inline double get_tangent_error() const {return tangents_estimate_error_;}
    
    
    
private:
    void set_center_area(const cv::Rect &center_area);
    bool set_branches(const std::vector<std::vector<cv::Point> > &branches, double merge_angle_threshold);
    cv::Point2d CalculateTangent(const std::vector<cv::Point> &branch);
    cv::Point3d CalculateTangentLine(const std::vector<cv::Point> &branch);
    double CalculateTangentLine(const std::vector<cv::Point> &branch, cv::Point3d &tangent_line);
    
    cv::Point2d RefineCenterByTangentLines(const std::vector<cv::Point3d> &tangent_lines);
    cv::Rect center_area_;
    cv::Point2d center_;                                        // center coordination for current cross point
    std::vector<std::vector<cv::Point> > branches_;             // container for pixel in each branch
    std::vector<cv::Point2d> tangents_;                         // normalized tangent for every line pass the point
    std::vector<cv::Point3d> tangent_lines_;
    int id_;                                                    // the unique id for the point
    // int branch_length_;                                       // indidate how many pixels contained in every branch
    int branches_num_;                                          // when we load map from file, their branches points are not aviliable,just number of branch is aviliable
    bool is_good_pt_;
    static int ID_;

    double tangents_estimate_error_;                            // 切线估计误差：所有切线拟合的最大平均误差
    const double MAX_TANGENT_ERROR = 100;                       // used to indicate that there are not enough points to calculate the tangent
};



typedef std::shared_ptr<CrossPoint> CrossPointPtr;

}   // namespace rcll

#endif
