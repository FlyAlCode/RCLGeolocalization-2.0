#include "draw_shp_elements.h"
#include "get_shp_elements.h"
#include "cross_point_feature_creator.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

/************************************************************
*                              public                       *
*************************************************************/
ShpDrawer::ShpDrawer(double resolution, 
          double area_width, 
          double area_height){
    scale_ = resolution;
    area_height_ = area_height;
    area_width_ = area_width;
   
}

void ShpDrawer::ShpToImg(const std::string &shp_file_name, const std::string &offset_file_name){
    cv::Rect2d bound;
    std::vector<pts> elements;
    GetShpElements(shp_file_name, bound, elements);
    
    // convert to pixel
    std::vector<std::vector<cv::Point> > pixels;
    ShpPtsToPixels(elements, bound, pixels);
    
    // create grid to seprate the whole image and output all the offset to a file
    std::vector<cv::Point> start_pts;
    CreateGrid(start_pts);
    std::ofstream fout(offset_file_name.c_str());
    for(int i=0; i<start_pts.size(); ++i){
        fout<<start_pts[i].x<<" "<<start_pts[i].y<<std::endl;
    }
    fout.close();
    
    // create images and draw shp elements for all grid
    std::ostringstream osm;
    for(int i=0; i<start_pts.size(); ++i){
        cv::Mat result_map;     
        DrawShpInOneGrid(start_pts[i], pixels, result_map);
        
        std::string img_name = std::string("./img/map");
        osm<<"_"<<start_pts[i].x<<"_"<<start_pts[i].y<<".png";
        img_name += osm.str();
        osm.str("");
        cv::imwrite(img_name, result_map);
        std::cout<<"Finish deal with image: "<<img_name<<std::endl;
    }
    
}

void ShpDrawer::MakeAreaMap(const std::string &shp_file_name,
                            const cv::Vec4d &area_bound,
                            cv::Mat &result_map){         // area_bound = {min_x, min_y, max_x, max_y} 
    cv::Rect2d shp_bound;
    std::vector<pts> elements;
    GetShpElements(shp_file_name, shp_bound, elements);
    
    // convert to pixel
    std::vector<std::vector<cv::Point> > pixels;
    ShpPtsToPixels(elements, shp_bound, pixels);
    
    DrawShpOfCertainArea(shp_bound, area_bound, pixels, result_map); 
} 

void ShpDrawer::DetectCrossPoints(const std::string &shp_file_name, 
                        std::vector<rcll::CrossPointPtr> &cross_pts,            // express in corrected geo coordinate
                        const cv::Point2d &geo_offset ){
    cross_pts.clear();

    cv::Rect2d bound;
    std::vector<pts> elements;
    GetShpElements(shp_file_name, bound, elements);
    
    // convert to pixel
    std::vector<std::vector<cv::Point> > pixels;
    ShpPtsToPixels(elements, bound, pixels);
    
    // detect all cross points in the area, 
    // Considering that the area is too large to draw in a single image,
    // we divide the area into several sub_area
    std::vector<cv::Point> start_pts;
    CreateGrid(start_pts);

    std::cout<<"Divide the area into "<<start_pts.size()<<" grids"<<std::endl;
    
    // create images and draw shp elements for all grid
    // then, detect cross points in every grid
    rcll::CrossPointFeatureParam param;
    param.branch_length = 30;
    param.min_cross_point_distance = 4;
    param.threshold = 30;
    param.merge_angle_threshold = 20;
    
    cv::Mat result_map; 
    std::vector<rcll::CrossPointPtr> cross_pt_tmp;
    std::vector<cv::Point2d> geo_tangents_tmp;
    for(int i=0; i<start_pts.size(); ++i){
        std::cout<<"Deal with "<<i<<"th tile......"<<std::endl;
        DrawShpInOneGrid(start_pts[i], pixels, result_map, 1);
       
        
        // detector cross points
        rcll::CreateCrossPointFeatures(result_map, param, cross_pt_tmp, result_map, true);
        std::cout<<cross_pt_tmp.size()<<" cross points found!!!"<<std::endl;

        // converse detected cross points into geo coordiante
        for(int k=0; k<cross_pt_tmp.size(); k++){
            // tangents
            std::vector<cv::Point2d> tangents = cross_pt_tmp[k]->get_tangents();
            geo_tangents_tmp.clear();
            for(int j=0; j<tangents.size(); j++){
                geo_tangents_tmp.push_back(ImgVectorToGeoVector(tangents[j]));
            }

            // center
            cv::Point2d center_tmp = PixelToShpPt(cross_pt_tmp[k]->get_center() + cv::Point2d(start_pts[i].x, start_pts[i].y),
                                        bound, geo_offset);

            // new cross point
            rcll::CrossPointPtr tmp(new rcll::CrossPoint);
            tmp->ThinInit(center_tmp, 
                            cross_pt_tmp[k]->get_braches_num(), 
                            geo_tangents_tmp, 
                            cross_pt_tmp[k]->get_tangent_error());
            cross_pts.push_back(tmp);
        }
    }
}

void ShpDrawer::ShowImageOnMap(const std::string &shp_file_name,                // the shp file name
                        const std::vector<cv::Mat> &src_imgs,                                 // image to draw on the map
                        const std::vector<cv::Mat> &Hs,                                       // 将图像点转换到地理坐标系点的单应矩阵
                        const std::vector<cv::Point> &padding_sizes,                          // 图像外部的地图区域
                        std::vector<cv::Mat> &draw_imgs){
    cv::Rect2d shp_bound;
    std::vector<pts> elements;
    GetShpElements(shp_file_name, shp_bound, elements);
    
    // convert to pixel
    std::vector<std::vector<cv::Point> > pixels;
    ShpPtsToPixels(elements, shp_bound, pixels);
    
    for(int img_count=0; img_count<src_imgs.size(); img_count++){
        cv::Mat query_img = src_imgs[img_count];

        // 计算图片的区域
        std::vector<cv::Point2d> corners;
        corners.push_back(cv::Point2d(0,0));
        corners.push_back(cv::Point2d(0,query_img.rows));
        corners.push_back(cv::Point2d(query_img.rows,query_img.rows));
        corners.push_back(cv::Point2d(query_img.rows,0));
        std::vector<cv::Point2d> corners_ted;
        double min_x, min_y, max_x, max_y;
        for(int i=0; i<4; i++){
            // 计算地理边界
            cv::Point2d corner_ted = GetHTransformedPt(corners[i],  Hs[img_count]);            
            corners_ted.push_back(corner_ted);

            if(i==0){
                min_x = max_x = corner_ted.x;
                min_y = max_y = corner_ted.y;
            }
            else{
                if(min_x>corner_ted.x)
                    min_x = corner_ted.x;
                if(min_y>corner_ted.y)
                    min_y = corner_ted.y;
                if(max_x<corner_ted.x)
                    max_x = corner_ted.x;
                if(max_y<corner_ted.y)
                    max_y = corner_ted.y;
            }
        }

        // 计算加padding后的图片区域
        cv::Vec4d area_with_padding;    // minx miny maxx maxy
        area_with_padding[0] = min_x - padding_sizes[img_count].x * scale_;
        area_with_padding[1] = min_y - padding_sizes[img_count].y * scale_;
        area_with_padding[2] = max_x + padding_sizes[img_count].x * scale_;
        area_with_padding[3] = max_y + padding_sizes[img_count].y * scale_;

        cv::Mat result_img;
        cv::Point2d start_pt;
        DrawShpOfCertainArea(shp_bound, area_with_padding, pixels, result_img, start_pt);
        result_img = 255 - result_img;
        // debug
        // cv::imshow("result_img", result_img);
        // cv::waitKey(0);

        // 计算query_img到局部图片的坐标转换关系
        double shp_to_pixel_data[9] = {1, 0, -shp_bound.x,
                                        0, -1, shp_bound.y + shp_bound.height,
                                        0, 0, scale_};

        cv::Mat H_shp_to_pixel(3, 3, CV_64F, shp_to_pixel_data);
        double whole_img_to_local_img_data[9] = {1, 0, -start_pt.x,
                                                0, 1, -start_pt.y,
                                                0, 0, 1};
        cv::Mat H_whole_img_to_local_img(3, 3, CV_64F, whole_img_to_local_img_data);     

        cv::Mat H_query_img_to_local_map_img = H_whole_img_to_local_img * H_shp_to_pixel * Hs[img_count];

        cv::Mat draw_img;                  
        // MergeWithMap(query_img, result_img, H_query_img_to_local_map_img, cv::Point(400,200), draw_img);
        MergeWithMap(query_img, result_img, H_query_img_to_local_map_img, padding_sizes[img_count], draw_img);
        draw_imgs.push_back(draw_img);
    }

}




/************************************************************
*                        private                            *
*************************************************************/
void ShpDrawer::ShpPtsToPixels(const std::vector<pts> &shp_pts,
                               const cv::Rect2d &bound,
                                std::vector<std::vector<cv::Point> >&pixels){
    pixels.clear();

    whole_img_width_ = bound.width / scale_;
    whole_img_height_ = bound.height / scale_;
    
    cv::Point pixel_tmp;
    std::vector<cv::Point> pixels_tmp;
    for(int i=0; i<shp_pts.size(); ++i){
        pixels_tmp.clear();
        for(int j=0; j<shp_pts[i].size(); ++j){
            pixel_tmp.x = (shp_pts[i][j].x - bound.x) / scale_;
            pixel_tmp.y = (bound.y + bound.height - shp_pts[i][j].y)/scale_;
            pixels_tmp.push_back(pixel_tmp);
        }
        pixels.push_back(pixels_tmp);
    }
}

cv::Point2d ShpDrawer::PixelToShpPt(const cv::Point2d & pixel_pt,
                               const cv::Rect2d &bound,
                               const cv::Point2d &geo_offset){
    cv::Point2d tmp;
    tmp.x = bound.x + scale_ * pixel_pt.x;
    tmp.y = bound.y + bound.height - scale_ * pixel_pt.y;

    return tmp + geo_offset;
}

int ShpDrawer::CreateGrid(std::vector<cv::Point> &start_pts){
    start_pts.clear();
    
    area_height_piexl_ = area_height_ / scale_;
    area_width_piexl_ = area_width_ / scale_;
    
    for(int x=0; x<whole_img_width_; x+=area_width_piexl_){
        for(int y=0; y<whole_img_height_; y+=area_height_piexl_){
            start_pts.push_back(cv::Point(x,y));
        }
    }
}

void ShpDrawer::DrawShpInOneGrid(const cv::Point &start_pts,                         // image coordiante in the whole image
                                 const std::vector<std::vector<cv::Point> >&pixels,
                                 cv::Mat &result_map,
                                 int line_width) {
    cv::Mat img = cv::Mat::zeros(area_height_piexl_ , 
                                 area_width_piexl_ , CV_8U);
    
    cv::Point offset =  - start_pts;
    cv::Rect bound(0, 0, img.cols, img.rows);
    cv::Point line_start, line_end;
    bool start_set;
    cv::Point tmp;
    for(int i=0; i<pixels.size(); ++i){
        start_set = false;
        for(int j=0; j<pixels[i].size(); ++j){
            tmp = pixels[i][j] + offset;
            if(!start_set){
                line_start = tmp;
                start_set = true;
            }   
            else{
                line_end = tmp;
                cv::line(img, line_start, line_end, cv::Scalar(255), line_width);               // line out of the img will not be drawed
                line_start = line_end;
            }
        }
    }
    img.copyTo(result_map);
}

void ShpDrawer::DrawShpOfCertainArea(const cv::Rect2d &shp_bound,
                                    const cv::Vec4d &area_bound,           // area_bound = {min_x, min_y, max_x, max_y}
                                    const std::vector<std::vector<cv::Point> >&pixels,
                                    cv::Mat &result_map) {
    pts area_corners;
    area_corners.push_back(cv::Point2d(area_bound[0], area_bound[3]));
    area_corners.push_back(cv::Point2d(area_bound[2], area_bound[1]));
    std::vector<pts> area_pts;
    area_pts.push_back(area_corners);
    std::vector<std::vector<cv::Point> > area_pixels;
    ShpPtsToPixels(area_pts, shp_bound, area_pixels);
    
    cv::Point start_pts(area_pixels[0][0]);
    cv::Point end_pts(area_pixels[0][1]);
    
    cv::Mat img = cv::Mat::zeros( end_pts.y - start_pts.y, 
                                  end_pts.x - start_pts.x , CV_8U);
    
    cv::Point offset =  - start_pts;
    cv::Rect bound(0, 0, img.cols, img.rows);
    cv::Point line_start, line_end;
    bool start_set;
    cv::Point tmp;
    for(int i=0; i<pixels.size(); ++i){
        start_set = false;
        for(int j=0; j<pixels[i].size(); ++j){
            tmp = pixels[i][j] + offset;
            if(!start_set){
                line_start = tmp;
                start_set = true;
            }   
            else{
                line_end = tmp;
                cv::line(img, line_start, line_end, cv::Scalar(255), 5);               // line out of the img will not be drawed
                line_start = line_end;
            }
        }
    }
    img.copyTo(result_map);
}

void ShpDrawer::DrawShpOfCertainArea(const cv::Rect2d &shp_bound,
                                    const cv::Vec4d &area_bound,           // area_bound = {min_x, min_y, max_x, max_y}
                                    const std::vector<std::vector<cv::Point> >&pixels,
                                    cv::Mat &result_map, 
                                    cv::Point2d &img_start_pt) {
    pts area_corners;
    area_corners.push_back(cv::Point2d(area_bound[0], area_bound[3]));
    area_corners.push_back(cv::Point2d(area_bound[2], area_bound[1]));
    std::vector<pts> area_pts;
    area_pts.push_back(area_corners);
    std::vector<std::vector<cv::Point> > area_pixels;
    ShpPtsToPixels(area_pts, shp_bound, area_pixels);
    
    cv::Point start_pt(area_pixels[0][0]);
    cv::Point end_pt(area_pixels[0][1]);
    
    cv::Mat img = cv::Mat::zeros( end_pt.y - start_pt.y, 
                                  end_pt.x - start_pt.x , CV_8U);
    
    cv::Point offset =  - start_pt;
    cv::Rect bound(0, 0, img.cols, img.rows);
    cv::Point line_start, line_end;
    bool start_set;
    cv::Point tmp;
    for(int i=0; i<pixels.size(); ++i){
        start_set = false;
        for(int j=0; j<pixels[i].size(); ++j){
            tmp = pixels[i][j] + offset;
            if(!start_set){
                line_start = tmp;
                start_set = true;
            }   
            else{
                line_end = tmp;
                cv::line(img, line_start, line_end, cv::Scalar(255),5);               // line out of the img will not be drawed
                line_start = line_end;
            }
        }
    }
    img.copyTo(result_map);
    img_start_pt = start_pt;
}

cv::Point2d ShpDrawer::GetHTransformedPt(const cv::Point2d &src_pt, const cv::Mat &H){
    cv::Mat corner_h(3, 1, CV_64F);
    corner_h.at<double>(0,0) = src_pt.x;
    corner_h.at<double>(1,0) = src_pt.y;
    corner_h.at<double>(2,0) = 1;

    cv::Mat corner_h_ted = H * corner_h;
    corner_h_ted = corner_h_ted/corner_h_ted.at<double>(2,0);

    cv::Point2d dst_pt;
    dst_pt.x = corner_h_ted.at<double>(0,0);
    dst_pt.y = corner_h_ted.at<double>(1,0);

    return dst_pt;
}


bool ShpDrawer::MergeWithMap(const cv::Mat &query_img,               
               const cv::Mat &map_img,                            
               const cv::Mat &H, 
               const cv::Point &padding_size,                        
               cv::Mat &draw_img){
    // calculate four transformed corners
    double corner_data[12] = {0, query_img.cols, 0, query_img.cols, 
                              0, 0, query_img.rows, query_img.rows,
                              1, 1, 1, 1};
    cv::Mat corners(3, 4, CV_64F, corner_data);
    cv::Mat transformed_corners = H * corners;
    for(int i=0; i<4; ++i){
        transformed_corners.at<double>(0,i) /= transformed_corners.at<double>(2,i);
        transformed_corners.at<double>(1,i) /= transformed_corners.at<double>(2,i);
    }
    
    // std::cout<<corners<<std::endl;
    // std::cout<<transformed_corners<<std::endl;
    
    int min_x=transformed_corners.at<double>(0,0);
    int max_x=transformed_corners.at<double>(0,0);
    int min_y=transformed_corners.at<double>(1,0);
    int max_y=transformed_corners.at<double>(1,0);
    for(int i=1; i<4; ++i){
        if(transformed_corners.at<double>(0,i)>max_x)
            max_x = transformed_corners.at<double>(0,i);
        if(transformed_corners.at<double>(0,i)<min_x)
            min_x = transformed_corners.at<double>(0,i);
        
        if(transformed_corners.at<double>(1,i)>max_y)
            max_y = transformed_corners.at<double>(1,i);
        if(transformed_corners.at<double>(1,i)<min_y)
            min_y = transformed_corners.at<double>(1,i);
    }
    if(max_x<=min_x || max_y<min_y || max_x<=0 
        || max_y<=0 || min_x>=map_img.cols-1 || min_y>=map_img.rows-1)
        return false;
    
    if(min_x<0)
        min_x=0;
    if(min_y<0)
        min_y=0;
    if(max_x>map_img.cols-1)
        max_x = map_img.cols-1;
    if(max_y>map_img.rows-1)
        max_y=map_img.rows-1;

    // draw on map_img
    double min_x_with_padding = min_x - padding_size.x;
    double min_y_with_padding = min_y - padding_size.y;
    double max_x_with_padding = max_x + padding_size.x;
    double max_y_with_padding = max_y + padding_size.y;

    if(min_x_with_padding<0)
        min_x_with_padding=0;
    if(min_y_with_padding<0)
        min_y_with_padding=0;
    if(max_x_with_padding>map_img.cols-1)
        max_x_with_padding = map_img.cols-1;
    if(max_y_with_padding>map_img.rows-1)
        max_y_with_padding=map_img.rows-1;

    cv::Mat roi_img, color_query_img;
    cv::Rect roi(min_x_with_padding, min_y_with_padding, max_x_with_padding-min_x_with_padding+1, max_y_with_padding-min_y_with_padding+1);
    std::cout<<"roi = "<<roi<<std::endl;
    map_img(roi).copyTo(roi_img);
    cv::Mat binary_roi_img;
    if(roi_img.channels()!=1)
        cv::cvtColor(roi_img, binary_roi_img, CV_RGB2GRAY);
    else
        roi_img.copyTo(binary_roi_img);
    cv::threshold(binary_roi_img, binary_roi_img, 30, 255, cv::THRESH_BINARY);
    
    // cv::namedWindow("roi_img");
    // cv::imshow("roi_img", roi_img);
    // cv::waitKey();
        
    if(roi_img.channels()!=3)
        cv::cvtColor(roi_img, roi_img, CV_GRAY2RGB);
    
    if(query_img.channels()!=3)
        cv::cvtColor(query_img, color_query_img, CV_GRAY2RGB);
    else
        query_img.copyTo(color_query_img);
    
    cv::Mat H_inv = H.inv();
    for(double x=min_x; x<max_x; x++){
        for(double y=min_y; y<max_y; y++){
            double pt_data[3] = {x, y, 1};
            cv::Mat pt(3, 1, CV_64F, pt_data);
            cv::Mat transformed_pt_h = H_inv * pt;
            cv::Point transformed_pt(transformed_pt_h.at<double>(0,0)/transformed_pt_h.at<double>(2,0), 
                                       transformed_pt_h.at<double>(1,0)/transformed_pt_h.at<double>(2,0));
            if(transformed_pt.x<0 || transformed_pt.y<0 
                || transformed_pt.x>query_img.cols-1 || transformed_pt.y>query_img.rows-1)
                continue;
            cv::Vec3b color = query_img.at<cv::Vec3b>(transformed_pt);
            if(color == cv::Vec3b(0,0,0))
                continue;
                
            int roi_x = x-min_x_with_padding;
            int roi_y = y-min_y_with_padding;
            if(roi_x<0 || roi_y<0 || roi_x>roi_img.cols-1 || roi_y>roi_img.rows-1)
                continue;
            roi_img.at<cv::Vec3b>(roi_y, roi_x) = color;
        }
    }

    // redraw road map
    for(int y=0; y<roi_img.rows; ++y){
        for(int x=0; x<roi_img.cols; ++x){
            if(binary_roi_img.at<uchar>(y, x)==0)
                roi_img.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0);
        }
    }
    
    roi_img.copyTo(draw_img);
    return true;
}


