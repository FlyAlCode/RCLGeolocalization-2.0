#include <iostream>
#include <stdlib.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "voronoi_map.h"
#include "voronoi_surface.h"
#include "shp.h"

#ifdef _DEBUG_
#include <stdio.h>
#include <time.h>
#endif
// #define _TEST_READ_FROM_FILE_


#ifdef _TEST_READ_FROM_FILE_
bool IsMatEqual(const cv::Mat &m1, const cv::Mat &m2){
    if(!(m1.rows==m2.rows && m1.cols==m2.cols && m1.channels()==m2.channels()))
        return 0;

    // cv::Mat dif = (m1 != m2);
    cv::Mat dif = m1 -m2;

    int count =0; 
    for(int i=0; i<m1.rows; ++i){
        for(int j=0; j<m1.cols; ++j){
            if(std::abs(m1.at<float>(i,j)-m2.at<float>(i,j))>20)
                ++count;
        }
    }
#ifdef _DEBUG_
    int count = cv::countNonZero(dif);
    cv::namedWindow("dif", CV_WINDOW_NORMAL);
    cv::namedWindow("m1", CV_WINDOW_NORMAL);
    cv::namedWindow("m2", CV_WINDOW_NORMAL);
    cv::imshow("dif", dif);
    cv::imshow("m1", m1);
    cv::imshow("m2", m2);
    cv::waitKey(0);
#endif

    std::cout<<"non_zero_num = "<<count<<std::endl;
    return count==0;
}

#endif


int main(int argc, char *argv[]){
    if(argc<6){
        std::cout<<"Usage: shp_to_voronoi [shp_file] [info_file] [tile_img_path] [gsd] [tile_size]"<<std::endl;
        exit(-1);
    }
    std::cout.precision(7);

    // create VoronoiMap instance
    VoronoiMap vor_map;

    // read shp file
    Shp my_shp;
    my_shp.Init(argv[1]);
    std::cout<<"Finishing reading shp file ..."<<std::endl;
    std::cout<<"Begin computing voronoi surface: "<<std::endl;

    // geo bound
    cv::Rect2d geo_bound = my_shp.get_geo_bound();
    vor_map.SetGeoBound(geo_bound.x, geo_bound.y, 
                        geo_bound.x+geo_bound.width, geo_bound.y+geo_bound.height);

    // tile params
    int tile_size = atoi(argv[5]);
    float gsd = atof(argv[4]);
    int tile_row = std::ceil(geo_bound.height * gsd / tile_size);
    int tile_col = std::ceil(geo_bound.width * gsd / tile_size);

    vor_map.Init(gsd, tile_row, tile_col, tile_size);

    // get each tile, and compute corresponding voronoi surface
    cv::Mat tile_img, vor_surface;
    VoronoiSurfaceGenerator vor_generator;
    float max_distance;

    //  四周增加padding_size大小的padding，以消除分割带来的影响
    int padding_size = 500;
    cv::Rect2d area_geo_bound, real_geo_bound;
    area_geo_bound.width = tile_size / gsd;          
    area_geo_bound.height = area_geo_bound.width;

    cv::Rect2d padding_geo_bound;
    padding_geo_bound.width = area_geo_bound.width + padding_size * 2;
    padding_geo_bound.height = padding_geo_bound.width;
    for(int i=0; i<tile_row; ++i){
        for(int j=0; j<tile_col; ++j){
            std::cout<<"    Dealing with tile: ["<<i<<", "<<j<<"]"<<std::endl;
            area_geo_bound.x = j * tile_size / gsd + geo_bound.x;
            area_geo_bound.y =  geo_bound.y + geo_bound.height - (i + 1) * tile_size / gsd;
            padding_geo_bound.x = area_geo_bound.x - padding_size;
            padding_geo_bound.y = area_geo_bound.y - padding_size;

            my_shp.ConvertAreaToImage(padding_geo_bound, 
                                    1/gsd, 
                                    tile_img, 
                                    real_geo_bound,
                                    1);
            if(tile_img.channels()!=1)
                cv::cvtColor(tile_img, tile_img, CV_RGB2GRAY);
            cv::threshold(tile_img, tile_img, 30, 1, CV_THRESH_BINARY);
#ifdef _DEBUG_
            cv::imshow("binary", tile_img*255);
            cv::waitKey(0);
            char binary_img_name[100];
            sprintf(binary_img_name, "binary_%d_%d.png", i, j);
            cv::imwrite(std::string(argv[3])+binary_img_name, tile_img*255);
#endif
            
            // compute the real bound 
            cv::Vec4f padding_bound;
            padding_bound[0] = real_geo_bound.y+real_geo_bound.height - (area_geo_bound.y+area_geo_bound.height);
            padding_bound[1] = area_geo_bound.y - real_geo_bound.y;
            padding_bound[2] = area_geo_bound.x - real_geo_bound.x;
            padding_bound[3] = real_geo_bound.x+real_geo_bound.width - (area_geo_bound.x + area_geo_bound.width);
            for(int i=0; i<4; ++i){
                if(padding_bound[i]<0)
                    padding_bound[i] = 0;
                padding_bound[i] *= gsd;
            }

#ifdef _DEBUG_
            std::cout<<"        area_geo_bound = "<<area_geo_bound<<std::endl;
            std::cout<<"        padding_geo_bound = "<<padding_geo_bound<<std::endl;
            std::cout<<"        real_geo_bound = "<<real_geo_bound<<std::endl;
            std::cout<<"        padding_bound = "<<padding_bound<<std::endl;
#endif
                
            max_distance = vor_generator.Generate(tile_img, padding_bound, vor_surface);

#ifdef _DEBUG_
            std::cout<<"        max_distance = "<<max_distance<<std::endl;
            cv::Mat vor_show_img;
            vor_surface.convertTo(vor_show_img, 255/max_distance);
            cv::imshow("voronoi", vor_show_img);     // /max_distance*255
            cv::waitKey(0);
#endif

            vor_map.SetTileData(vor_surface, i, j, max_distance);
           
        }
    }

    // write to file
    int write_flag = vor_map.WriteToFile(argv[2], argv[3]);
    std::cout<<"write_flag = "<<write_flag<<std::endl;


#ifdef _TEST_READ_FROM_FILE_
    {
        // test read from file
        VoronoiMap vor_map_file;
        int read_flag = vor_map_file.ReadFromFile("17_voronoi_surface/17.info");
        // vor_map_file.WriteToFile("17_vor/17.info", "17_vor/tile");
        std::cout<<"read_flag = "<<read_flag<<std::endl;
        if(!(read_flag==1 && write_flag==1))
            return -1;

        std::cout<<"geo_bound: \n";
        std::cout<<"    estimated: "<<vor_map.get_geo_bound() <<", file: "<<vor_map_file.get_geo_bound()<<std::endl;
        std::cout<<"gsd:\n    estimated: "<<vor_map.get_gsd()<<", file: "<<vor_map_file.get_gsd()<<std::endl;
        std::cout<<"tile_info:\n    estimated: "<<vor_map.get_tile_info()<<", file: "<<vor_map_file.get_tile_info()<<std::endl;
        std::cout<<"tile_data:\n";
        for(int i=0; i<vor_map.get_tile_info()[1]; ++i){
            for(int j=0; j<vor_map.get_tile_info()[2]; ++j){
                std::shared_ptr<cv::Mat> e_tile_data = vor_map.get_tile_data(i,j);
                std::shared_ptr<cv::Mat> f_tile_data = vor_map_file.get_tile_data(i,j);
                int is_equal = -2;
                if((e_tile_data!=nullptr && f_tile_data!=nullptr))
                    is_equal = IsMatEqual(*e_tile_data, *f_tile_data);
                if(e_tile_data==nullptr && f_tile_data==nullptr)
                    is_equal == -1;   

                std::cout<<"["<<i<<", "<<j<<"]: "<<is_equal<<std::endl;
            }
        }
        // 随机访问若干个位置，观察最近的值是否一致
        srand((unsigned int)time(0));
        int geo_w, geo_h;
        cv::Vec4f geo_bound_test = vor_map.get_geo_bound();
        geo_w = geo_bound_test[2] - geo_bound_test[0];
        geo_h = geo_bound_test[3] - geo_bound_test[1];
        std::cout<<"Test access distance data (within bound): "<<std::endl;
        for(int i=0; i<100; ++i){
            float x = rand() % geo_w + geo_bound_test[0];
            float y = rand() % geo_h + geo_bound_test[1];
            float e_d = vor_map.GetVoronoiDistance(x,y);
            float f_d = vor_map_file.GetVoronoiDistance(x,y);
            std::cout<<"["<<x<<", "<<y<<"]: estimated = "<< e_d <<", file = "
                    << f_d <<", "<< (e_d==f_d) <<std::endl;
        }
        
        std::cout<<"\n\nTest access distance data (beyond bound): "<<std::endl;
        for(int i=0; i<10; ++i){
            float x = rand() % geo_w - geo_bound_test[0];
            float y = rand() % geo_h + geo_bound_test[1];
            float e_d = vor_map.GetVoronoiDistance(x,y);
            float f_d = vor_map_file.GetVoronoiDistance(x,y);
            std::cout<<"["<<x<<", "<<y<<"]: estimated = "<< e_d <<", file = "
                    << f_d <<", "<< (e_d==f_d) <<std::endl;
            
            x = rand() % geo_w + geo_bound_test[0];
            y = rand() % geo_h - geo_bound_test[1];
            e_d = vor_map.GetVoronoiDistance(x,y);
            f_d = vor_map_file.GetVoronoiDistance(x,y);
            std::cout<<"["<<x<<", "<<y<<"]: estimated = "<< e_d <<", file = "
                    << f_d <<", "<< (e_d==f_d) <<std::endl;

            x = rand() % geo_w + geo_bound_test[2];
            y = rand() % geo_h + geo_bound_test[1];
            e_d = vor_map.GetVoronoiDistance(x,y);
            f_d = vor_map_file.GetVoronoiDistance(x,y);
            std::cout<<"["<<x<<", "<<y<<"]: estimated = "<< e_d <<", file = "
                    << f_d <<", "<< (e_d==f_d) <<std::endl;

            x = rand() % geo_w + geo_bound_test[0];
            y = rand() % geo_h + geo_bound_test[3];
            e_d = vor_map.GetVoronoiDistance(x,y);
            f_d = vor_map_file.GetVoronoiDistance(x,y);
            std::cout<<"["<<x<<", "<<y<<"]: estimated = "<< e_d <<", file = "
                    << f_d <<", "<< (e_d==f_d) <<std::endl;
        }

    }
#endif

    return 0;
}