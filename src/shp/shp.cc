#include "shp.h"
#include "get_shp_elements.h"
#include "opencv2/imgproc/imgproc.hpp"

// debug
#include<iostream>

bool Shp::Init(const std::string &shp_file_name){
    // 读取shp文件，获取所有折线的地理坐标、边界
    if(!GetShpElements(shp_file_name, geo_bound_, geo_polylines_))
        return false;

    // 将边界和所有折线转换到地理坐标系下
    pixel_bound_.x = 0;
    pixel_bound_.y = 0;
    pixel_bound_.height = geo_bound_.height;
    pixel_bound_.width = geo_bound_.width;

    pixel_polylines_.resize(geo_polylines_.size());
    for (int i = 0; i < geo_polylines_.size(); i++){
        pixel_polylines_[i].clear();
        for (int j = 0; j < geo_polylines_[i].size(); j++){
            pixel_polylines_[i].push_back(GeoToImage(geo_polylines_[i][j]));
        }
    }

    return true;
}

// 将指定区域的矢量图转换为栅格图
// area_geo_bound --- 地理坐标系下的转换区域的边界
// scale --- 地理坐标系到像素坐标系的尺度（每个像素代表多少m）
// line_width --- 线的宽度
// result_img --- 转换得到的图像
// 返回值 --- 整个区域都超出边界，返回false；否则取边界内的区域，并返回true
bool Shp::ConvertAreaToImage(const cv::Rect_<double> &area_geo_bound,
                             double scale,
                             cv::Mat &result_img,
                             cv::Rect2d &real_geo_bound,
                             int line_width,
                             const cv::Scalar &color,
                             bool black_background) const {
    // 判断区域是否在地图内
    if(area_geo_bound.x>=geo_bound_.x+geo_bound_.width ||
        area_geo_bound.x+area_geo_bound.width<=geo_bound_.x ||
        area_geo_bound.y>=geo_bound_.y+geo_bound_.height ||
        area_geo_bound.y+area_geo_bound.height<=geo_bound_.y)
        return false;

    // 确定地图内的边界
    double min_x = area_geo_bound.x;
    double min_y = area_geo_bound.y;
    double max_x = min_x + area_geo_bound.width;
    double max_y = min_y + area_geo_bound.height;

    if (min_x<geo_bound_.x)
        min_x = geo_bound_.x;
    if(max_x>geo_bound_.x+geo_bound_.width)
        max_x = geo_bound_.x + geo_bound_.width;
    if(min_y<geo_bound_.y)
        min_y = geo_bound_.y;
    if(max_y>geo_bound_.y+geo_bound_.height)
        max_y = geo_bound_.y + geo_bound_.height;

    cv::Rect2d vaild_geo_bound(min_x, min_y, max_x-min_x, max_y-min_y);
    real_geo_bound = vaild_geo_bound;

    // 创建地图图片
    if(black_background)
        result_img = cv::Mat::zeros(vaild_geo_bound.height / scale, vaild_geo_bound.width / scale, CV_8UC3);
    else{
        result_img = cv::Mat::ones(vaild_geo_bound.height / scale, vaild_geo_bound.width / scale, CV_8U)*255;
        cv::cvtColor(result_img, result_img, CV_GRAY2RGB);
    }

    cv::Point offset = GeoToImage(cv::Point2d(vaild_geo_bound.x, vaild_geo_bound.y + vaild_geo_bound.height));
    for (int i = 0; i < pixel_polylines_.size(); i++)  {
        for (int j = 1; j < pixel_polylines_[i].size(); j++){
            cv::line(result_img,
                     cv::Point((pixel_polylines_[i][j - 1].x - offset.x) / scale, (pixel_polylines_[i][j - 1].y - offset.y) / scale),
                     cv::Point((pixel_polylines_[i][j].x - offset.x) / scale, (pixel_polylines_[i][j].y - offset.y) / scale),
                     color,
                     line_width);
        }
    }
    return true;
}

// 将show_img显示在地图上
// T --- 图片坐标系到地理坐标系的转换关系
// padding_size --- 投影图片之外需要添加的背景边界大小
// color --- 道路图显示的颜色
// result_show_img --- 显示的结果图
// 如果图片在给定转换关系下与地图有重合区域，返回true；否则返回false
bool Shp::ShowOnMap(const cv::Mat &show_img,
                    const cv::Mat &T,
                    int padding_size,
                    const cv::Scalar &color,
                    cv::Mat &result_show_img,
                    bool black_background) const {
    // 计算show_img在地图上的投影区域
    cv::Point2d p[4];
    p[0] = GetTransformedPt(cv::Point2d(0, 0), T);
    p[1] = GetTransformedPt(cv::Point2d(0, show_img.cols), T);
    p[2] = GetTransformedPt(cv::Point2d(show_img.rows, 0), T);
    p[3] = GetTransformedPt(cv::Point2d(show_img.cols, show_img.rows), T);

    double min_x = p[0].x;
    double min_y = p[0].y;
    double max_x = p[0].x;
    double max_y = p[0].y;
    for (int i = 1; i < 4; i++){
        if(p[i].x<min_x)
            min_x = p[i].x;
        if(p[i].y<min_y)
            min_y = p[i].y;
        if(p[i].x>max_x)
            max_x=p[i].x;
        if(p[i].y>max_y)
            max_y = p[i].y;
    }
    min_x -= padding_size;
    min_y -= padding_size;
    max_x += padding_size;
    max_y += padding_size;

    if(min_x<geo_bound_.x)
        min_x = geo_bound_.x;
    if(min_y<geo_bound_.y)
        min_y = geo_bound_.y;
    if(max_x>geo_bound_.x+geo_bound_.width)
        max_x = geo_bound_.x + geo_bound_.width;
    if(max_y>geo_bound_.y+geo_bound_.height)
        max_y = geo_bound_.y + geo_bound_.height;

    // 判断是否存在合理区域
    if(min_x>=max_x||min_y>=max_y)
        return false;

    // 计算show_img到局部地图的转换关系
    cv::Mat T_geo_2_pixel = GetTransformationFromImageToGeo().inv();
    cv::Point offset = GeoToImage(cv::Point2d(min_x, max_y));
    cv::Mat T_offset = cv::Mat::zeros(3, 3, CV_32F);
    T_offset.at<float>(0, 0) = 1;
    T_offset.at<float>(0, 2) = -offset.x;
    T_offset.at<float>(1, 1) = 1;
    T_offset.at<float>(1, 2) = -offset.y;
    T_offset.at<float>(2, 2) = 1;

    cv::Mat T_32f;
    if(T.type()!=CV_32F)
        T.convertTo(T_32f, CV_32F);
    else
        T.copyTo(T_32f);

    cv::Mat T_img_2_local_map = T_offset * T_geo_2_pixel * T_32f;

    // 创建局部地图，并投影
    cv::Mat T_local_map_2_img = T_img_2_local_map.inv();
    float src_weight, dst_weight;
    cv::Rect2d real_geo_bound;
    if(black_background){
        ConvertAreaToImage(cv::Rect2d(min_x, min_y, max_x-min_x, max_y-min_y),1, result_show_img,real_geo_bound, 5, color, black_background);
        src_weight = 0.8;
    }
    else{
        ConvertAreaToImage(cv::Rect2d(min_x, min_y, max_x-min_x, max_y-min_y),1, result_show_img,real_geo_bound, 5, color, black_background);
        src_weight = 0.2;
    }
    dst_weight = 1 - src_weight;

    cv::Mat color_src_img;
    if(show_img.channels()==3)
        show_img.copyTo(color_src_img);
    else
        cv::cvtColor(show_img, color_src_img, CV_GRAY2RGB);

    cv::Point2d transformed_pt;
    cv::Rect src_bound(0, 0, show_img.cols, show_img.rows);
    for (int x = 0; x < result_show_img.cols; x++)  {
        for (int y = 0; y < result_show_img.rows; y++){
            transformed_pt = GetTransformedPt(cv::Point2d(x, y), T_local_map_2_img);
            transformed_pt.x = std::floor(transformed_pt.x);
            transformed_pt.y = std::floor(transformed_pt.y);
            if(transformed_pt.inside(src_bound)){
                // debug
                // std::cout << transformed_pt << std::endl;

                result_show_img.at<cv::Vec3b>(y, x) 
                        = src_weight * color_src_img.at<cv::Vec3b>(transformed_pt) + dst_weight * result_show_img.at<cv::Vec3b>(y, x);
            }
        }
    }
    return true;
}

// 获取将图片坐标转换为地理坐标的转换矩阵
cv::Mat Shp::GetTransformationFromImageToGeo() const{
    cv::Mat tmp = cv::Mat::zeros(3, 3, CV_32F);
    tmp.at<float>(0, 0) = 1;
    tmp.at<float>(0, 2) = geo_bound_.x;
    tmp.at<float>(1, 1) = -1;
    tmp.at<float>(1, 2) = geo_bound_.y + geo_bound_.height;
    tmp.at<float>(2, 2) = 1;

    return tmp;
}

/***************************private ******************************/
// 图像坐标转换到地理坐标
cv::Point2d Shp::ImageToGeo(const cv::Point &img_pt) const{
    return cv::Point2d(img_pt.x + geo_bound_.x, geo_bound_.y + geo_bound_.height - img_pt.y);
}

// 图片坐标系到地理坐标系
cv::Point Shp::GeoToImage(const cv::Point2d &geo_pt) const{
    return cv::Point(geo_pt.x - geo_bound_.x, geo_bound_.height - (geo_pt.y - geo_bound_.y));
}

cv::Point2d Shp::GetTransformedPt(const cv::Point2d &src_pt, const cv::Mat &H) const{
    cv::Mat corner_h(3, 1, CV_32F);
    corner_h.at<float>(0,0) = src_pt.x;
    corner_h.at<float>(1,0) = src_pt.y;
    corner_h.at<float>(2,0) = 1;

    cv::Mat H_32f;
    if (H.type() != CV_32F)
        H.convertTo(H_32f, CV_32F);
    else
        H.copyTo(H_32f);

    cv::Mat corner_h_ted = H_32f * corner_h;
    corner_h_ted = corner_h_ted/corner_h_ted.at<float>(2,0);

    cv::Point2d dst_pt;
    dst_pt.x = corner_h_ted.at<float>(0,0);
    dst_pt.y = corner_h_ted.at<float>(1,0);

    return dst_pt;
}
