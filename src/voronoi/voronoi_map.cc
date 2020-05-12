#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

#include <opencv2/highgui.hpp>

#include "voronoi_map.h"


/* Load tiles from file under the guidance of the given info file. 
    The corresponding voronoi surface image file must be provide in the specific path.
    主要操作包括：
    0、清理tiles内存
    1、设置geo bound相关信息；
    2、设置总图片大小，网格划分信息；
    3、为tiles分配内存空间
    4、读取每个tile对应的文件，并保存。 
*/
int VoronoiMap::ReadFromFile(const std::string &info_file){
    std::ifstream fin(info_file);
    if(!fin.is_open()){
        std::cout<<"Cannot find given file, please check!"<<std::endl;
        return -1;
    }

    fin>>geo_x_min_>>geo_y_min_>>geo_x_max_>>geo_y_max_;
    fin>>scale_>>tile_cols_>>tile_rows_>>tile_size_;
    fin>>vaild_tile_num_;

    img_w_ = geo_x_max_ - geo_x_min_;
    img_h_ = geo_y_max_ - geo_y_min_;

    // debug
    // std::cout<<"vaild_tile_num_ = "<<vaild_tile_num_<<std::endl;

    int tile_num = tile_rows_ * tile_cols_;
    tile_data_.resize(tile_num);
    tile_distance_scale_.resize(tile_num);

    std::string tile_img_name;
    int tile_r;
    int tile_c;
    float tile_scale;
    cv::Mat tile_o_data;
    for(int i=0; i<vaild_tile_num_; ++i){
        fin>>tile_img_name;
        tile_o_data = cv::imread(tile_img_name, CV_LOAD_IMAGE_GRAYSCALE);
        if(tile_o_data.empty()){
            std::cout<<"Cannot find tile image file: "<<tile_img_name<<std::endl;
            fin.close();
            // roll back changes
            tile_data_.clear();
            tile_distance_scale_.clear();

            return 0;
        }
        
        fin>>tile_r>>tile_c>>tile_scale;
        
        // tile_o_data.convertTo(tile_o_data, CV_32F, tile_scale);
        // tile_o_data = tile_o_data * tile_scale;

        int tile_index = tile_r * tile_cols_ + tile_c;
        tile_data_[tile_index].reset(new cv::Mat());
        tile_o_data.convertTo(*(tile_data_[tile_index]), CV_32F, tile_scale);
        // tile_o_data.copyTo(*(tile_data_[tile_index]));
        tile_distance_scale_[tile_index] = tile_scale;
    }
    fin.close();
    return 1;
}

/* Save the voronoi surface information into file.
    * @ img_file_pre --- 保存voronoi surface图片文件名的前缀，后续命名规则为_row_col.png
*/
int VoronoiMap::WriteToFile(const std::string &info_file, 
                            const std::string &img_file_pre){
    std::ofstream fout(info_file);
    if(!fout.is_open()){
        std::cout<<"Write file fail!"<<std::endl;
        return -1;
    }
    fout.precision(7);
    fout << geo_x_min_ <<" "<< geo_y_min_<<" " << geo_x_max_<<" " 
        << geo_y_max_ << " "<<scale_ << std::endl;
    fout << tile_cols_ <<" " << tile_rows_ << " " <<tile_size_ << std::endl;

    // check whether all data are set
    // for(int i=0; i<tile_cols_ * tile_rows_; ++i){
    //     if(tile_data_[i] == nullptr){
    //         std::cout<<"Empty tile data!"<<std::endl;
    //         return 0;
    //     }
    // }

    // int tile_num = tile_rows_ * tile_cols_;
    fout<<vaild_tile_num_<<std::endl;
    for(int i=0; i<tile_rows_; ++i){
        for(int j=0; j<tile_cols_; ++j){
            int tile_index = i * tile_cols_ + j;
            if(tile_distance_scale_[tile_index]==0)         // 当前区域不存在道路
                continue;

            char stmp[100];
            sprintf(stmp, "_%d_%d.png", i, j);
            std::string tile_img_name = img_file_pre + std::string(stmp);
            fout << tile_img_name <<" "<< i<<" " << j<<" " << tile_distance_scale_[tile_index] << std::endl;

            cv::Mat tile_img;
            tile_data_[tile_index]->convertTo(tile_img, CV_8U, 1.0/tile_distance_scale_[tile_index]);

            // debug
            // cv::imshow("save", tile_img);
            // cv::waitKey(0);
            // std::cout<< tile_img_name <<"： "<<tile_img.size()<<", "<<tile_distance_scale_[tile_index]<<std::endl;

            cv::imwrite(tile_img_name, tile_img);
        }
    }
    return 1;
}

// write data
void VoronoiMap::Init(float scale, int tile_rows, int tile_cols, int tile_size){
    scale_ = scale;
    tile_cols_ = tile_cols;
    tile_rows_ = tile_rows;
    tile_size_ = tile_size;

    // allocate memory
    tile_data_.resize(tile_cols_ * tile_rows_); 
    tile_distance_scale_.assign(tile_cols_ * tile_rows_, 0); 

    vaild_tile_num_ = 0;   
}

void VoronoiMap::SetGeoBound(float geo_min_x, 
                            float geo_min_y, 
                            float geo_max_x, 
                            float geo_max_y){
    geo_x_min_ = geo_min_x;
    geo_y_min_ = geo_min_y;
    geo_x_max_ = geo_max_x;
    geo_y_max_ = geo_max_y;

    img_w_ = geo_x_max_ - geo_x_min_;
    img_h_ = geo_y_max_ - geo_y_min_;
}

void VoronoiMap::SetTileData(const cv::Mat &tile_data, 
                            int row, int col, 
                            float max_distance){
    if(max_distance <=0 )
        return;

    int index = row * tile_cols_ + col;
    tile_data_[index].reset(new cv::Mat());

    tile_data.copyTo(*(tile_data_[index]));
    tile_distance_scale_[index] = max_distance / 255;
    ++vaild_tile_num_;
}

// access data
/* 给定一点的地理坐标，返回该点离最近道路点的距离
    如果给定点不在范围内，返回-1
*/
float VoronoiMap::GetVoronoiDistance(float geo_x, float geo_y){
    // to image coordination
    float img_x = geo_x - geo_x_min_;
    float img_y = geo_y_max_ - geo_y;

    if(img_x<0 || img_x >= img_w_ || img_y <0 || img_y >=img_h_)
        return -1;

    // to scale image coordination
    int s_img_x = img_x * scale_;
    int s_img_y = img_y * scale_;

    // get tile index and coordination in certain tile
    int row = s_img_y / tile_size_;
    int col = s_img_x / tile_size_;
    int tile_x = s_img_x % tile_size_;
    int tile_y = s_img_y % tile_size_;

    int tile_index = row*tile_cols_ + col;
        
    if(tile_data_[tile_index]==nullptr || tile_data_[tile_index]->empty())
        return FAR_AWAY;

    // debug
    // float d = tile_data_[tile_index]->at<float>(tile_y, tile_x);
    // float d_s = d / scale_;
    // std::cout<<"["<<row<<", "<<col<<", "<<tile_x<<", "<<tile_y<<", "
    //         <<tile_index<<", "<<d<<", "<<d_s<<"]"<<std::endl;

    return tile_data_[tile_index]->at<float>(tile_y, tile_x) / scale_;
}