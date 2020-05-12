#include "cross_line_extractor.h"

namespace rcll {

CrossLineExtractor::CrossLineExtractor(int line_length){
    line_length_ = line_length;
}

void CrossLineExtractor::FindAllStartRectsAroundKeypoint(const cv::Rect& center_area, 
                                                         std::vector< cv::Rect> & start_rects){
    start_rects.clear();
    std::vector<cv::Point> around_points;
    std::vector<uchar> around_points_value;
    
    // recored all around points
    cv::Point tmp_pt;
    for(int j=center_area.x-1; j<=center_area.x+center_area.width; j++){            // up
        tmp_pt.x = j;
        tmp_pt.y = center_area.y-1;
        around_points.push_back(tmp_pt);
        around_points_value.push_back(neighbor_img_.at<uchar>(tmp_pt));
    }
    
    for(int i=center_area.y; i<=center_area.y+center_area.height; i++){
        tmp_pt.x = center_area.x + center_area.width;
        tmp_pt.y = i;
        around_points.push_back(tmp_pt);
        around_points_value.push_back(neighbor_img_.at<uchar>(tmp_pt));
    }
    
    for(int j=center_area.x+center_area.width-1; j>=center_area.x-1; j--){
        tmp_pt.x = j;
        tmp_pt.y = center_area.y + center_area.height;
        around_points.push_back(tmp_pt);
        around_points_value.push_back(neighbor_img_.at<uchar>(tmp_pt));
    }
    
    for(int i=center_area.y+center_area.height-1; i>=center_area.y-1; i--){         // The up-left point is count twice to loop the circle
        tmp_pt.x = center_area.x - 1;
        tmp_pt.y = i;
        around_points.push_back(tmp_pt);
        around_points_value.push_back(neighbor_img_.at<uchar>(tmp_pt));
    }
    
    bool mask = false;
    int min_x, min_y, max_x, max_y;
    for(int i=0; i<around_points.size(); i++){
        if(mask){
            if(around_points_value[i]==2){
                if(around_points[i].x>max_x)
                    max_x = around_points[i].x;
                if(around_points[i].x<min_x)
                    min_x = around_points[i].x;
                if(around_points[i].y>max_y)
                    max_y = around_points[i].y;
                if(around_points[i].y<min_y)
                    min_y = around_points[i].y;
            }
            else{
                start_rects.push_back(cv::Rect(min_x, min_y, max_x-min_x+1, max_y-min_y+1));
                mask = false;
            }            
        }
        else{
            if(around_points_value[i]==2){
                min_x = around_points[i].x;
                max_x = around_points[i].x;
                min_y = around_points[i].y;
                max_y = around_points[i].y;
                mask = true;
            }
        }
    }  
}

int CrossLineExtractor::ExpandToNextPoint(const cv::Rect& centre, 
                                          const cv::Rect& last_centre, 
                                          cv::Rect& next_rect){
    int min_i;
    int max_i;
    int min_j;
    int max_j;
    bool rect_inited = false;
    cv::Rect last_centre_around = cv::Rect(last_centre.x-1, last_centre.y-1, last_centre.width+2, last_centre.height+2);
    for(int i=centre.y-1; i<=centre.y+centre.height; i++){
        for(int j=centre.x-1; j<=centre.x+centre.width; j++){
            bool last_centre_around_contain = last_centre_around.contains(cv::Point(j,i));
            if(!last_centre_around_contain && ((neighbor_img_.at<uchar>(i,j)>2) || neighbor_img_.at<uchar>(i,j)==1)){            // cross or terminal point found
                return -1;
            }
            if(!last_centre_around_contain && visited_map_.at<uchar>(i,j)==0 && neighbor_img_.at<uchar>(i,j)==2 ){
                if(rect_inited){
                    if(i < min_i)
                        min_i = i;
                    if(i > max_i)
                        max_i = i;
                    if(j < min_j)
                        min_j = j;
                    if(j > max_j)
                        max_j = j;
                }
                else{
                    min_i = i;
                    max_i = i;
                    min_j = j;
                    max_j = j;
                    rect_inited = true;
                }
//                 visited_map_.at<uchar>(i,j) = 1; 
            }
            
        }
    }

    // The rect is not initialized, which indicate no rect can be expanded to. 
    // This cocours when encounting the boundary of the image 
    if(!rect_inited){                      
        return 0;
    }          
    else{
        // set the new rect as visited
        for(int i=min_i; i<=max_i; i++){
            for(int j=min_j; j<=max_j; j++){
                visited_map_.at<uchar>(i, j) = 1;
            }
        }
        next_rect.x = min_j;
        next_rect.y = min_i;
        next_rect.width = max_j - min_j + 1;
        next_rect.height = max_i- min_i + 1;
        return 1;
    }
}

bool CrossLineExtractor::ExpandOneBranch(const cv::Rect& center, 
                                         const cv::Rect& start_rect, 
                                         std::vector< cv::Point >& branch){
     // 0. set the start_rect as visited    
    for(int x=start_rect.x; x<start_rect.x+start_rect.width; x++){
        for(int y=start_rect.y; y<start_rect.y+start_rect.height; y++){
            visited_map_.at<uchar>(y, x) = 1;
        }
    }
    
    // 1. push back the start rect
    branch.clear();
    for(int i=start_rect.x; i<start_rect.x+start_rect.width; i++){
        for(int j=start_rect.y; j<start_rect.y+start_rect.height; j++){
            branch.push_back(cv::Point(i,j));
        }
    }
     
    // 2. expand 
    cv::Rect last_centre = center;
    cv::Rect current_rect = start_rect;
    cv::Rect next_rect;
    int expand_line_flag;

    int N = this->line_length_;
    while(N--){
        expand_line_flag = ExpandToNextPoint(current_rect, last_centre, next_rect);
        if(expand_line_flag ==1){
            last_centre = current_rect;
            current_rect = next_rect;        
            // push all the points in the expanded rect
            for(int i=next_rect.x; i<next_rect.x+next_rect.width; i++){
                for(int j=next_rect.y; j<next_rect.y+next_rect.height; j++){
                    branch.push_back(cv::Point(i,j));
                }
            }
        }
        else{
            return false;
        }
    }
    return true;
}


void CrossLineExtractor::SetNeighborImage(const cv::Mat& neighbor_img){
    neighbor_img.copyTo(neighbor_img_);
    visited_map_ = cv::Mat::zeros(neighbor_img_.rows, neighbor_img_.cols, neighbor_img_.type());
}

bool CrossLineExtractor::FindAllBranches(const cv::Rect& center_area,
                                         std::vector< std::vector< cv::Point > >& branches){
    // reset visited_map_ to deal with two jacent points
    memset(visited_map_.data, 0, visited_map_.rows*visited_map_.step*sizeof(char));
    
    branches.clear();
    std::vector< cv::Rect>  start_rects;
    FindAllStartRectsAroundKeypoint(center_area, start_rects);
    
    std::vector< cv::Point > branch;
    for(int i=0; i<start_rects.size(); i++){        
        ExpandOneBranch(center_area, start_rects[i], branch);
        branches.push_back(branch);
    }
    
    return true;
}




    
}   // namespace rcll