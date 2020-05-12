#include "random_grid.h"
#include <cmath>
#include <ctime>
#include <cstdlib> 
#include <memory.h>
#include <opencv2/core/core.hpp>

// debug
#include <iostream>
#include <glog/logging.h>

namespace rcll{

void RandomGrid2D::Init(int grid_size, 
                        float grid_min_x, 
                        float grid_max_x, 
                        float grid_min_y, 
                        float grid_max_y){
    // 1. Set the parameters
    grid_size_  = grid_size;
    grid_min_x_ = grid_min_x;
    grid_min_y_ = grid_min_y;
    grid_max_x_ = grid_max_x;
    grid_max_y_ = grid_max_y;

    // 2. Compute grid_r_ and grid_c_
    grid_c_ = std::ceil((grid_max_x - grid_min_x_)/grid_size_);
    grid_r_ = std::ceil((grid_max_y - grid_min_y_)/grid_size_);

    // 3. Allocate memory for grid_data_
    grid_data_ = new int* [grid_c_ * grid_r_];
    for(int i=0; i<grid_c_ * grid_r_; ++i)
        grid_data_[i] = nullptr;

    shift_num_ = 9;
    for(int i=-1; i<2; ++i){
        for(int j=-1; j<2; ++j){
            rand_dx_.push_back(i*grid_size_);
            rand_dy_.push_back(j*grid_size_);
        }
    }

}


void RandomGrid2D::Reset(){
    // 1. Clear models
    all_models_.clear();

    // 2. Clear grid_data_
    for(int i=0; i<grid_r_*grid_c_; ++i)
        if(grid_data_[i] != nullptr){
            delete [] grid_data_[i];
            grid_data_[i] = nullptr;
        }
}


int RandomGrid2D::InsertNewModel(const std::shared_ptr<Model> & new_model, 
                                    float center_x, 
                                    float center_y,
                                    std::vector<int> &ids,
                                    double H_error_threshold){
    // 1. Compute the transformed center under the new_model
    float h_z = center_x * new_model->H_mat[6] + center_y * new_model->H_mat[7] + new_model->H_mat[8];
    float h_x = (center_x * new_model->H_mat[0] + center_y * new_model->H_mat[1] + new_model->H_mat[2])/h_z;
    float h_y = (center_x * new_model->H_mat[3] + center_y * new_model->H_mat[4] + new_model->H_mat[5])/h_z;

    all_models_.push_back(new_model);
    int current_model_index = all_models_.size() -1;
    ids.clear();
    for(int i=0; i<shift_num_; ++i){
        // 2. Shift the transformed center with the random shifts
        float shift_h_x = h_x + rand_dx_[i];
        float shift_h_y = h_y + rand_dy_[i];

        // 3. For each shifted center, quantize it into a grid
        int c = std::floor((shift_h_x - grid_min_x_)/grid_size_);
        int r = std::floor((shift_h_y - grid_min_y_)/grid_size_);
        if(c<0 || c>=grid_c_ || r<0 || r>=grid_r_)
            continue;
        int grid_index = r*grid_c_ + c;

        // insert new model
        if(rand_dx_[i]==0 && rand_dy_[i]==0){
            if(grid_data_[grid_index]==nullptr) {        // empty hash bin
                grid_data_[grid_index] = new int [MAX_PT_NUM_EACH_GRID];
                memset(grid_data_[grid_index], -1, sizeof(int)*MAX_PT_NUM_EACH_GRID);        // 初始化所有索引为非法值-1
                grid_data_[grid_index][0] = current_model_index;
            }
            else{
                for(int j=0; j<MAX_PT_NUM_EACH_GRID; ++j){                  // find first empty pointer
                    if(grid_data_[grid_index][j] == -1){
                        grid_data_[grid_index][j] = all_models_.size() -1;
                        break;
                    }
                }
            }
        }

        // 4. find collisions       
        if(grid_data_[grid_index]!=nullptr) {
            for(int j=0; j<MAX_PT_NUM_EACH_GRID; ++j){                  // find first empty pointer
                if(grid_data_[grid_index][j] == -1){                   
                    break;
                }
            
                if(all_models_[grid_data_[grid_index][j]]->parent_id_ != new_model->parent_id_
                    && CalModelDistance(all_models_[grid_data_[grid_index][j]], new_model, center_x*2, center_y*2)<H_error_threshold){
                    ids.push_back(grid_data_[grid_index][j]);
                }
            }
        }
    }


    return ids.size();      // 未检测到冲突
}


// clear memory
RandomGrid2D::~RandomGrid2D(){
    for(int i=0; i<grid_r_*grid_c_; ++i)
        if(grid_data_[i] != nullptr){
            delete [] grid_data_[i];
        }
    delete [] grid_data_;
}


double RandomGrid2D::CalModelDistance(const std::shared_ptr<Model> &_m1, 
                                    const std::shared_ptr<Model> &_m2,
                                    int img_w, int img_h){
    cv::Mat m1(3,3, CV_64F, _m1->H_mat);
    cv::Mat m2(3,3, CV_64F, _m2->H_mat);
    
	cv::Mat H_w_c = m1.inv();
	cv::Mat h1_cross_h2 = H_w_c.col(0).cross(H_w_c.col(1));
	h1_cross_h2 = -h1_cross_h2 /
				  std::abs(h1_cross_h2.at<double>(2, 0)); // 归一化, 同时反转符号
	
	double x[4] = {0, img_w, img_w, 0};
	double y[4] = {0, 0, img_h, img_h};
	double h_cross_0 = h1_cross_h2.at<double>(0, 0);
	double h_cross_1 = h1_cross_h2.at<double>(1, 0);
	double h_cross_2 = h1_cross_h2.at<double>(2, 0);
	double lamda[4];
	for (int i = 0; i < 4; ++i)	{
		lamda[i] = h1_cross_h2.at<double>(0, 0) * x[i] +
				   h1_cross_h2.at<double>(1, 0) * y[i] +
				   h1_cross_h2.at<double>(2, 0);
	}

	std::vector<cv::Point2d> intersections;
	intersections.push_back(
		cv::Point2d((lamda_threshold - h1_cross_h2.at<double>(2, 0)) / h1_cross_h2.at<double>(0, 0), 0));
	intersections.push_back(
		cv::Point2d(img_w, (lamda_threshold - h1_cross_h2.at<double>(2, 0) -
							h1_cross_h2.at<double>(0, 0) * img_w) /
							   h1_cross_h2.at<double>(1, 0)));
	intersections.push_back(
		cv::Point2d((lamda_threshold - h1_cross_h2.at<double>(2, 0) -
					 h1_cross_h2.at<double>(1, 0) * img_h) /
						h1_cross_h2.at<double>(0, 0),
					img_h));
	intersections.push_back(
		cv::Point2d(0, (lamda_threshold - h1_cross_h2.at<double>(2, 0)) /
						   h1_cross_h2.at<double>(1, 0)));

	std::vector<cv::Point2d> corners;
	if (lamda[0] < lamda_threshold)
		corners.push_back(cv::Point2d(x[0], y[0]));
	if (intersections[0].x < img_w && intersections[0].x >= 0)
		corners.push_back(intersections[0]);
	if (lamda[1] < lamda_threshold)
		corners.push_back(cv::Point2d(x[1], y[1]));
	if (intersections[1].y < img_h && intersections[1].y >= 0)
		corners.push_back(intersections[1]);
	if (lamda[2] < lamda_threshold)
		corners.push_back(cv::Point2d(x[2], y[2]));
	if (intersections[2].x < img_w && intersections[2].x >= 0)
		corners.push_back(intersections[2]);
	if (lamda[3] < lamda_threshold)
		corners.push_back(cv::Point2d(x[3], y[3]));
	if (intersections[3].y < img_h && intersections[3].y >= 0)
		corners.push_back(intersections[3]);
    
    double d = 0;
    for(int i=0; i<corners.size(); ++i){
        double z1 = m1.at<double>(2,0) * corners[i].x +  m1.at<double>(2,1) * corners[i].y + m1.at<double>(2,2);
        double x1 = (m1.at<double>(0,0) * corners[i].x + m1.at<double>(0,1) * corners[i].y + m1.at<double>(0,2))/z1;
        double y1 = (m1.at<double>(1,0) * corners[i].x + m1.at<double>(1,1) * corners[i].y + m1.at<double>(1,2))/z1;
        double z2 = m2.at<double>(2,0) * corners[i].x +  m2.at<double>(2,1) * corners[i].y + m2.at<double>(2,2);
        double x2 = (m2.at<double>(0,0) * corners[i].x + m2.at<double>(0,1) * corners[i].y + m2.at<double>(0,2))/z2;
        double y2 = (m2.at<double>(1,0) * corners[i].x + m2.at<double>(1,1) * corners[i].y + m2.at<double>(1,2))/z2;
        double dx = std::abs(x1-x2);
        double dy = std::abs(y1-y2);
        if(dx>d)
            d = dx;
        if(dy>d)
            d = dy;
    }
    return d;
}

}   // namespace rcll