#include "cross_points_detector.h"

// debug
#include <iostream>

namespace rcll{
    
CrossPointDetector::CrossPointDetector(int min_cross_point_distance){
    min_cross_point_distance_ = min_cross_point_distance;
}

CrossPointDetector::~CrossPointDetector(){
    
}

int CrossPointDetector::CountDisconnectPoints(const cv::Mat& local_image){
    int w = local_image.cols;
    int h = local_image.rows;
    std::vector<uchar> around_pixel;
    for(int i = 0; i<w; i++)
        around_pixel.push_back(local_image.at<uchar>(0,i));
    for(int i=1; i<h; i++)
        around_pixel.push_back(local_image.at<uchar>(i,w-1));
    for(int i=w-2; i>=0; i--)
        around_pixel.push_back(local_image.at<uchar>(h-1, i));
    for(int i=h-2; i>0; i--)
        around_pixel.push_back(local_image.at<uchar>(i,0));
    
    const int max_step = 1;                             // max number of points to skip to avoid adjacent point
    int num = 0;
    int connected_points_num = 0;
    int points_num = around_pixel.size();
    for(int i=0; i<points_num-1; i++){
        if(around_pixel[i]==1){
            if(connected_points_num == 0){
                ++num;                        
            }
            ++connected_points_num;
            if(connected_points_num > max_step)
                connected_points_num = 0;
        }
        else{
            connected_points_num = 0;
        }
    }
    
    if(around_pixel[points_num-1]==1){                  // deal with the last which is special for adjance with first point
        if(connected_points_num==0 && around_pixel[0]==0)     
            ++num;
    }
    
    return num;
}

void CrossPointDetector::CountDisconnectPointsForAllPoints(const cv::Mat &input_img, cv::Mat &output_img){  
    cv::Mat tmp = cv::Mat::zeros(input_img.size(), input_img.type());
    for(int i=1; i<tmp.rows-1; i++){
        for(int j=1; j<tmp.cols-1; j++){
            if(input_img.at<uchar>(i,j)==1)
                tmp.at<uchar>(i,j) = CountDisconnectPoints(input_img(cv::Rect(j-1,i-1,3,3)));
            else
                tmp.at<uchar>(i,j) = 0;
        }
    }
    tmp.copyTo(output_img);
}

/*  @ 1. When encountering a point with value larger than 2, a neighbor area with radius r(3)
 *  @   is search to find all points with value larger than 2; after than, the max/min in x/y
 *  @   are calculated to find the boundary, and a rect area is formed, which is treated as the keypoint. 
 *  @ 2. Then a new search around the area is performed,to find the how many lines connected to the keypoint. 
 */
void CrossPointDetector::MergeCrossPoints(const cv::Mat &src, const cv::Mat &neighbor_img, cv::Mat &output_img, std::vector<cv::Rect> &keypoint_area){
    const int max_keypoint_radius = min_cross_point_distance_;                                  // the radius to search to form a keypoint
    neighbor_img.copyTo(output_img);
    keypoint_area.clear();
    cv::Mat point_visited_map = cv::Mat::zeros(neighbor_img.size(), neighbor_img.type());
    for(int i=0; i<neighbor_img.rows; i++){
        for(int j=0; j<neighbor_img.cols; j++){
            if(point_visited_map.at<uchar>(i,j)==0 && neighbor_img.at<uchar>(i,j)>2){       // unvisited cross points
                int max_i = i;
                int min_i = i;
                int max_j = j;
                int min_j = j;
                for(int di=0; di<=max_keypoint_radius; di++){      // points above the current are all visited, thus are not need to search                         
                    for(int dj=-max_keypoint_radius; dj<=max_keypoint_radius; dj++){
                        int search_i = i+di;
                        int search_j = j+dj;
                        if(search_i<0 || search_i>neighbor_img.rows-1 || search_j<0 ||search_j >neighbor_img.cols-1)
                            continue;
                        if(neighbor_img.at<uchar>(search_i, search_j)>2){
                            if(search_i>max_i)                      
                                max_i = search_i;
                            if(search_j>max_j)
                                max_j = search_j;
                            if(search_j<min_j)
                                min_j = search_j;
                        }
                    }
                }
                
                // record the cross point
                keypoint_area.push_back(cv::Rect(min_j, min_i, max_j-min_j+1, max_i-min_i+1));
                // redefine the type of the point
                int type = CountDisconnectPoints(src(cv::Rect(min_j-1, min_i-1, max_j-min_j+1+2, max_i-min_i+1+2)));
                for(int di=min_i; di<=max_i; di++){
                    for(int dj=min_j; dj<=max_j; dj++){
                        point_visited_map.at<uchar>(di, dj) = 1;                    // mark all points belong the keypoint as visited to avoid forming more than once
                        output_img.at<uchar>(di, dj) = type;                        // refine the type for the point
                    }
                }
            }
            
        }
    }
}

void CrossPointDetector::Run(const cv::Mat &input_img, cv::Mat &cross_points_img, std::vector<cv::Rect> &keypoint_area){
    cv::Mat tmp;
    CountDisconnectPointsForAllPoints(input_img, tmp);
    MergeCrossPoints(input_img, tmp, cross_points_img, keypoint_area);
}

/*  @Used just for debug
 */
void CrossPointDetector::ColorNeighborImage(const cv::Mat &neighbor_img, cv::Mat &colored_neighbor_img){
    cv::Mat show_img(neighbor_img.size(), CV_8UC3);
    for(int i=0; i<neighbor_img.rows; i++){
        for(int j=0; j<neighbor_img.cols; j++){
            switch(neighbor_img.at<uchar>(i,j)){
                case 0:
                    show_img.at<cv::Vec3b>(i,j) = cv::Vec3b(0, 0, 0);
                    break;
                case 1:
                    show_img.at<cv::Vec3b>(i,j) = cv::Vec3b(255, 0, 0);
                    break;
                case 2:
                    show_img.at<cv::Vec3b>(i,j) = cv::Vec3b(0, 255, 0);
                    break;
                case 3:
                    show_img.at<cv::Vec3b>(i,j) = cv::Vec3b(0, 0, 255);
                    break;
                case 4:
                    show_img.at<cv::Vec3b>(i,j) = cv::Vec3b(255,255,0);
                    break;
                default:
                    show_img.at<cv::Vec3b>(i,j) = cv::Vec3b(255,255,255);
                    
            }
        }
    }
    show_img.copyTo(colored_neighbor_img);
}





    
}   // road_match

