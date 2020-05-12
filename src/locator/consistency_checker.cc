#include "consistency_checker.h"
#include "shp.h"

namespace rcll {

bool ConsistencyChecker::Init(const std::string &shp_file,
                              double icp_inlier_threshold,
                              double icp_precision_threshold,
                              int icp_max_iter_num,
                              double sample_pt_max_distance,
                              double min_inlier_rate){
    icp_inlier_threshold_ = icp_inlier_threshold;
    icp_max_iter_num_ = icp_max_iter_num;
    icp_precision_threshold_ = icp_precision_threshold;
    sample_pt_max_distance_ = sample_pt_max_distance;
    min_inlier_rate_ = min_inlier_rate;

    if(!GetRefPointsFromShpFile(shp_file))
        return false;

    icp_optimizer_.reset(new ICPOptimizer());
    icp_optimizer_->Init(ref_pts_, icp_inlier_threshold_, icp_precision_threshold_, icp_max_iter_num_);

    return true;
}

bool ConsistencyChecker::Init(const std::vector<cv::Point2f> &map_pts,
              double icp_inlier_threshold,
              double icp_precision_threshold,
              int icp_max_iter_num,
              double sample_pt_max_distance,
              double min_inlier_rate){
    icp_inlier_threshold_ = icp_inlier_threshold;
    icp_max_iter_num_ = icp_max_iter_num;
    icp_precision_threshold_ = icp_precision_threshold;
    sample_pt_max_distance_ = sample_pt_max_distance;
    min_inlier_rate_ = min_inlier_rate;

    ref_pts_ = map_pts;

    icp_optimizer_.reset(new ICPOptimizer());
    icp_optimizer_->Init(ref_pts_, icp_inlier_threshold_, icp_precision_threshold_, icp_max_iter_num_);

    return true;
}

bool ConsistencyChecker::Run(const std::vector<cv::Point2f> &query_pts,
                             const cv::Mat &initial_h,
                             cv::Mat &final_h,
                             double &final_inlier_ratio){    
    icp_optimizer_->Run(query_pts, cv::Point2f(0,0), initial_h, final_h, final_inlier_ratio);

    return final_inlier_ratio >= min_inlier_rate_;
}


bool ConsistencyChecker::Run(const cv::Mat &binary_img,
             const cv::Mat &initial_h,
             cv::Mat &final_h,
             double &final_inlier_ratio){
    std::vector<cv::Point2f> query_pts;
    for(int i=0; i<binary_img.rows; i+=5){
        for(int j=0; j<binary_img.cols; j+=5){
            if(binary_img.at<uchar>(i,j)==1)
                query_pts.push_back(cv::Point2f(j, i));
        }
    }
    icp_optimizer_->Run(query_pts, cv::Point2f(0,0), initial_h, final_h, final_inlier_ratio);

    return final_inlier_ratio >= min_inlier_rate_;
}


bool ConsistencyChecker::BootstrapRun(const std::vector<cv::Point2f> &query_pts,
                             const cv::Point2f &query_anchor_pt,
                             float max_distance,
                             const cv::Mat &initial_h,
                             cv::Mat &final_h,
                             double &final_inlier_ratio){    
    std::vector<cv::Point2f> query_pts_0, query_pts_1, query_pts_2;
    float distance;
    for (int i = 0; i < query_pts.size(); ++i){
        distance = cv::norm(query_pts[i] - query_anchor_pt);
        if(distance<max_distance/4){
            query_pts_0.push_back(query_pts[i]);
            query_pts_1.push_back(query_pts[i]);
            query_pts_2.push_back(query_pts[i]);
        }
        else if(distance<max_distance/2){
            query_pts_1.push_back(query_pts[i]);
            query_pts_2.push_back(query_pts[i]);
        }
        else{
            query_pts_2.push_back(query_pts[i]);
        }
    }

    bool is_converge;
    is_converge = icp_optimizer_->Run(query_pts_0, query_anchor_pt, initial_h, final_h, final_inlier_ratio);
    if(!is_converge)
        return false;
    is_converge = icp_optimizer_->Run(query_pts_1, query_anchor_pt, final_h, final_h, final_inlier_ratio);
    if(!is_converge)
        return false;
    
    is_converge = icp_optimizer_->Run(query_pts_2, query_anchor_pt, final_h, final_h, final_inlier_ratio);
    if(!is_converge)
        return false;

    return final_inlier_ratio >= min_inlier_rate_;
}


bool ConsistencyChecker::BootstrapRun(const cv::Mat &query_img,
             const cv::Point2f &query_anchor_pt,
             const cv::Mat &initial_h,
             cv::Mat &final_h,
             double &final_inlier_ratio){  
    final_inlier_ratio = 0;
    if(initial_h.empty()){
        return false;
    }
    initial_h.copyTo(final_h);
    std::vector<cv::Point2f> query_pts;
    double d_scale[3] = {0.25, 0.5, 1};
    for(int i=0; i<3; ++i){
        query_pts.clear();
        int min_x = query_anchor_pt.x - query_img.cols * d_scale[i];
        min_x = min_x<0 ? 0 : min_x;
        int min_y = query_anchor_pt.y - query_img.rows * d_scale[i];
        min_y = min_y<0 ? 0 : min_x;
        int max_x = query_anchor_pt.x + query_img.cols * d_scale[i];
        max_x = max_x> query_img.cols ? query_img.cols : max_x;
        int max_y = query_anchor_pt.y + query_img.rows * d_scale[i];
        max_y = max_y>query_img.rows ? query_img.rows : max_y;

        for(int x = min_x; x<max_x; x+=3){
            for(int y = min_y; y<max_y; y+=3){
                if(query_img.at<uchar>(y, x)==1)
                    query_pts.push_back(cv::Point2f(x,y));
            }
        }
        if(query_pts.size()<50)
            continue;
        bool is_converge;
        is_converge = icp_optimizer_->Run(query_pts, query_anchor_pt, final_h, final_h, final_inlier_ratio);
        if(!is_converge || final_inlier_ratio<min_inlier_rate_)
            return false;
    }

    return final_inlier_ratio >= min_inlier_rate_;
}




double ConsistencyChecker::GetInlierRatio(const std::vector<cv::Point2f> &query_pts,
                          const cv::Mat &h,
                          double threshold){
    return icp_optimizer_->GetInlierRatio(query_pts, h, threshold);
}

double ConsistencyChecker::GetAverageMatchDistance(const std::vector<cv::Point2f> &query_pts,
                          const cv::Mat &h,
                          double inlier_rate){
    return icp_optimizer_->GetAverageMatchError(query_pts, h, inlier_rate);
}

// private
bool ConsistencyChecker::GetRefPointsFromShpFile(const std::string &shp_file){
    Shp my_shp;
    if(!my_shp.Init(shp_file))
        return false;

    std::vector<std::vector<cv::Point2d>> polylines;
    polylines = my_shp.get_all_polylines();

    ref_pts_.clear();
    cv::Point2d last_pt;
    for (int i = 0; i < polylines.size(); ++i){
        if(polylines[i].size()<2){
            ref_pts_.push_back(polylines[i][0]);
            continue;
        }

        last_pt = polylines[i][0];
        ref_pts_.push_back(last_pt);
        for (int j = 1; j < polylines[i].size(); ++j) {
            double line_length = cv::norm(polylines[i][j] - last_pt);
            int step_num = std::floor(line_length / sample_pt_max_distance_);

            if(step_num==0){                            
                ref_pts_.push_back(polylines[i][j]);
            }
            else{
                cv::Point2d tangent((polylines[i][j].x - last_pt.x) / line_length,
                                    (polylines[i][j].y - last_pt.y) / line_length);
                for (int k = 1; k <= step_num; ++k) {
                    ref_pts_.push_back(last_pt + k * tangent);
                }
                ref_pts_.push_back(polylines[i][j]);
            }

            last_pt = polylines[i][j];
        }
    }

    return true;
}

} // namespace rcll 