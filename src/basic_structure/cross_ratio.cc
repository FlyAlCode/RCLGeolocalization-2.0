#include "cross_ratio.h"
// debug
#include <iostream>

namespace rcll{
    
double CalCrossRatioFromPoints(const std::vector< cv::Point2d >& pts){
    CV_Assert(pts.size() == 4);
    
    double x1x2 = cv::norm(pts[1]-pts[0]);
    double x3x4 = cv::norm(pts[3]-pts[2]);
    double x1x3 = cv::norm(pts[2]-pts[0]);
    double x2x4 = cv::norm(pts[3]-pts[1]);
    
    return x1x2 * x3x4 / x1x3 /x2x4;
}


double CalCrossRatioFromNormalTangents(const std::vector< cv::Point2d >& normal_tangents){
    CV_Assert(normal_tangents.size() == 4);
    
    // here the sign is not considered
    double sin_12 = normal_tangents[0].cross(normal_tangents[1]);
    double sin_34 = normal_tangents[2].cross(normal_tangents[3]);
    double sin_13 = normal_tangents[0].cross(normal_tangents[2]);
    double sin_24 = normal_tangents[1].cross(normal_tangents[3]);
    
    return cv::abs(sin_12 * sin_34 / sin_13 /sin_24);
}


double CalCrossRatioFromLines(const std::vector< cv::Point3d >& lines){
    CV_Assert(lines.size() == 4);
    
    std::vector<cv::Point2d> normal_tangents;
    cv::Point2d tangent_tmp;
    for(int i=0; i<lines.size(); i++){
        tangent_tmp.x = lines[i].y;
        tangent_tmp.y = -lines[i].x;
        // normal_tangents.push_back(tangent_tmp/cv::norm(tangent_tmp));
        normal_tangents.push_back(tangent_tmp);                             // here we suppose tangent_tmp has been normalized
    }
    
    return CalCrossRatioFromNormalTangents(normal_tangents);
}

bool CheckCrossRatioConsistency(const std::vector< cv::Point2d >& x, 
                                const std::vector< cv::Point2d >& y, 
                                const double threshold){
    if(x.size()<4)
        return true;
    
    int n = x.size();
    int m = 4;
    int M = 4;
    int *temp = new int [n];
    int *index = new int [n];
    for(int i=0; i<n; i++)
        index[i] = i;
    
    std::vector<std::vector<int> > vec_res;
    Combine(index, n, m, temp, M, vec_res);
    
    std::vector<cv::Point2d> x_sub, y_sub;
    for(int i=0; i<vec_res.size(); i++){
        x_sub.clear();
        y_sub.clear();
        
        for(int j=0; j<vec_res[i].size(); j++){
            x_sub.push_back(x[vec_res[i][j]]);
            y_sub.push_back(y[vec_res[i][j]]);
        }
        
        double x_cross_ratio = CalCrossRatioFromNormalTangents(x_sub);
        double y_cross_ratio = CalCrossRatioFromNormalTangents(y_sub);
        
        if(cv::abs(x_cross_ratio - y_cross_ratio)/x_cross_ratio > threshold)
            return false;
    }
    
    delete [] temp;
    delete [] index;
    return true;
}

void CalAllCrossRatios(const std::vector< cv::Point2d >& normal_tangents, 
                       std::vector<double> &cross_ratios){
    cross_ratios.clear();
    if(normal_tangents.size()<4)
        return;
    
    int n = normal_tangents.size();
    int m = 4;
    int M = 4;
    int *temp = new int [n];
    int *index = new int [n];
    for(int i=0; i<n; i++)
        index[i] = i;
    
    std::vector<std::vector<int> > vec_res;
    Combine(index, n, m, temp, M, vec_res);
    
    std::vector<cv::Point2d> normal_tangents_sub;
    for(int i=0; i<vec_res.size(); i++){
        normal_tangents_sub.clear();
        
        for(int j=0; j<vec_res[i].size(); j++){
            normal_tangents_sub.push_back(normal_tangents[vec_res[i][j]]);
        }
        
        cross_ratios.push_back(CalCrossRatioFromNormalTangents(normal_tangents_sub)); 
    }
    
    delete [] temp;
    delete [] index;
}

void Combine(int data[],
             int n,
             int m,
             int temp[],
             const int M,
             std::vector<std::vector<int> > &vec_res){
    for(int i=n; i>=m; i--)  
    {
        temp[m-1] = i - 1;
        if (m > 1)
            Combine(data,i-1,m-1,temp,M,vec_res);
        else                   
        {
            std::vector<int > vec_temp;
            for(int j=M-1; j>=0; j--){
                vec_temp.push_back(data[temp[j]]);
            }
            vec_res.push_back(vec_temp);
        }
    }
}


    
}   // namespace rcll
