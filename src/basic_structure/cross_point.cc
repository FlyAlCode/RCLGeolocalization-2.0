#include "cross_point.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#define rad_ratio 3.1415927/180

namespace rcll{
    
int CrossPoint::ID_ = 0;

bool CrossPoint::FillData(const cv::Rect& center_area, 
                          const std::vector< std::vector< cv::Point > >& branches, 
                          double merge_angle_threshold){
    set_center_area(center_area);
    return set_branches(branches, merge_angle_threshold);
}

void CrossPoint::ThinInit(const cv::Point2d &center,                        // here, "thin" means only center/branch number/tangent number/tangents are initilized 
              const int branch_num, 
              const std::vector<cv::Point2d> &tangents, 
              const double tangent_error){
    center_ = center;
    branches_num_ = branch_num;
    tangents_ = tangents;
    tangents_estimate_error_ = tangent_error;
}

void CrossPoint::Draw(cv::Mat& draw_img, const cv::Scalar& color){
    if(draw_img.channels()==1)
        cv::cvtColor(draw_img, draw_img, CV_GRAY2RGB);
    
    // draw branch
    // for(int i=0; i<branches_.size(); i++){
    //     for(int j=0; j<branches_[i].size(); j++){
    //         draw_img.at<cv::Vec3b>(branches_[i][j]) = cv::Vec3b(color[1], color[0], color[2]);
    //     }
    // }
    
    // draw tangents
    for(int i=0; i<tangents_.size(); i++){
        cv::line(draw_img, 
                 cv::Point(center_.x+20*tangents_[i].x, center_.y+20*tangents_[i].y), 
                 cv::Point(center_.x-20*tangents_[i].x, center_.y-20*tangents_[i].y),
                 color, 2);
    }
    
    // draw center
    //     for(int x = center_area_.x; x<center_area_.x+center_area_.width; x++){
    //         for(int y=center_area_.y; y<center_area_.y+center_area_.height; y++){
    //             draw_img.at<cv::Vec3b>(y, x) = cv::Vec3b(color[2], color[1], color[0]);
    //         }
    //     }
    cv::Point2d center_int (center_.x, center_.y);
    draw_img.at<cv::Vec3b>(center_int) = cv::Vec3b(color[2], color[1], color[0]);
}

    
void CrossPoint::set_center_area(const cv::Rect& center_area) {
    center_area_ = center_area;
    center_.x = center_area_.x + center_area_.width/2.0;
    center_.y = center_area_.y + center_area_.height/2.0;
}

bool CrossPoint::set_branches(const std::vector< std::vector< cv::Point > >& branches, 
                              double merge_angle_threshold){
    if(branches.size()<3)
        return false;
    branches_ = branches;
    branches_num_ = branches_.size();
    
    tangents_.clear();
    
    // calculate tangents
    std::vector<cv::Point2d> tangent_tmp;
    cv::Point3d tangent_line_tmp;
    std::vector<int> flag;                          // 0-no colliner | -x --- opposite  | x --- same direction (all indexs start form 1)
    for(int i=0; i<branches_.size(); i++){
        if(branches_[i].size()<10) {                          // we require at least 10 points to fit the line
            tangents_estimate_error_ = MAX_TANGENT_ERROR;
            return true;
        }
        double error = CalculateTangentLine(branches_[i], tangent_line_tmp);
        if(error>tangents_estimate_error_)
            tangents_estimate_error_ = error;
        
        tangent_tmp.push_back(cv::Point2d(tangent_line_tmp.y, -tangent_line_tmp.x));
        tangent_lines_.push_back(tangent_line_tmp);
        flag.push_back(0);
    }
    if(tangent_tmp.size()<2)
        return false;

    // check collinear tangents
    double cos_merge_angle_threshold = cos(merge_angle_threshold * rad_ratio);
    for(int i=0; i<tangent_tmp.size(); i++){
        for(int j=i+1; j<tangent_tmp.size(); j++){
            double cos_angle = tangent_tmp[i].dot(tangent_tmp[j]);
            if(cos_angle>cos_merge_angle_threshold){                     // cos(20°)
                flag[i] = j+1;
                flag[j] = i+1;
            }
            else if(cos_angle<-cos_merge_angle_threshold){
                flag[i] = -j-1;
                flag[j] = -i-1;
            }
        }
    }
    
    // merges collinear tangents
    for(int i=0; i<flag.size(); i++){
        if(flag[i]<0){
            if(-flag[i]-1>=i)                               // because all indexs in flag start form 1, here we substract 1
                tangents_.push_back((tangent_tmp[i]-tangent_tmp[-flag[i]-1])/2.0);
        }
        else if(flag[i]==0){
            tangents_.push_back(tangent_tmp[i]);
        }
        else{
            if(flag[i]-1>=i)
                tangents_.push_back((tangent_tmp[i]+tangent_tmp[flag[i]-1])/2.0);
        }
        
    }
    
    // refine center
    center_ = RefineCenterByTangentLines(tangent_lines_);
    return true;
}

/* center point is supposed to be set before call this function, no check will be done
 */
cv::Point2d CrossPoint::CalculateTangent(const std::vector< cv::Point >& branch){
    cv::Mat A(branch.size(), 2, CV_64F);
    for(int i=0; i<A.rows; i++){
        A.at<double>(i, 0) = branch[i].x - center_.x;
        A.at<double>(i, 1) = branch[i].y - center_.y;
    }
    
    cv::Mat ATA = A.t() * A;
    cv::Mat eigenvalues;
    cv::Mat eigenvectors;
    cv::eigen(ATA, eigenvalues, eigenvectors);
    
    return cv::Point2d(eigenvectors.at<double>(1,1), -eigenvectors.at<double>(1,0));   // tangent = (b, -a)
}

cv::Point3d CrossPoint::CalculateTangentLine(const std::vector< cv::Point >& branch){
    // cv::Point2d mean(0,0);
    // for(int i=0; i<branch.size(); i++){
    //     mean.x = mean.x + branch[i].x;
    //     mean.y = mean.y + branch[i].y;
    // }
    // mean.x /= branch.size();
    // mean.y /= branch.size();
    
    // cv::Mat A(branch.size(), 2, CV_64F);
    // for(int i=0; i<A.rows; i++){
    //     A.at<double>(i, 0) = branch[i].x - center_.x;
    //     A.at<double>(i, 1) = branch[i].y - center_.y;
    // }
    
    // cv::Mat ATA = A.t() * A;
    // cv::Mat eigenvalues;
    // cv::Mat eigenvectors;
    // cv::eigen(ATA, eigenvalues, eigenvectors);
    
    // std::vector<cv::Point> data;
    // for(int i=2; i<branch.size(); i++){                     // remove the first two points
    //     data.push_back(branch[i]);
    // }
    
    cv::Vec4f line_para; 
    cv::fitLine(branch, line_para, cv::DIST_L2, 0, 1e-2, 1e-2);
    double a = line_para[1];
    double b = -line_para[0];
    double c = -a * line_para[2] -b * line_para[3];
    
    return cv::Point3d(a, b, c );
    
    // if(b*(branch[branch.size()-1].x-branch[0].x)>=0)
    //     return cv::Point3d(a, b, c );
    // else
    //     return cv::Point3d(-a, -b, -c );
    
}

double CrossPoint::CalculateTangentLine(const std::vector< cv::Point >& branch, 
                                        cv::Point3d& tangent_line){
    cv::Vec4f line_para; 
    std::vector< cv::Point > branch_used(branch.begin()+3, branch.end());
    cv::fitLine(branch_used, line_para, cv::DIST_L2, 0, 1e-2, 1e-2);
    double a = line_para[1];
    double b = -line_para[0];
    double c = -a * line_para[2] -b * line_para[3];
    
    tangent_line = cv::Point3d(a, b, c );
    
    double error_sum_L2 = 0;
    for(int i=0; i<branch_used.size(); ++i){
        double error = a * branch_used[i].x + b * branch_used[i].y + c;
        error_sum_L2 += error * error;
    }
    return std::sqrt(error_sum_L2/branch_used.size());
}


cv::Point2d CrossPoint::RefineCenterByTangentLines(const std::vector< cv::Point3d >& tangent_lines){
    CV_Assert(tangent_lines.size()>=2);
    cv::Mat A(tangent_lines.size(), 2, CV_64F);
    cv::Mat b(tangent_lines.size(), 1, CV_64F);
    
    for(int i=0; i<tangent_lines.size(); i++){
        A.at<double>(i, 0) = tangent_lines[i].x;
        A.at<double>(i, 1) = tangent_lines[i].y;
        b.at<double>(i, 0) = -tangent_lines[i].z;
    }
    
    cv::Mat x;
    cv::solve(A, b, x, cv::DECOMP_SVD);
    return cv::Point2d(x.at<double>(0,0), x.at<double>(1,0));
}

std::ostream & operator << (std::ostream &os, const CrossPoint &pt){
    os<<pt.get_tangent_error()<<" ";
    os<<pt.center_.x<<" "<<pt.center_.y<<" "
        <<pt.branches_num_<<" "<<pt.tangents_.size();
    for(int i=0; i<pt.tangents_.size(); ++i){
        os<<" "<<pt.tangents_[i].x<<" "<<pt.tangents_[i].y;
    }
    os<<std::endl;
}



}   // namespace rcll
