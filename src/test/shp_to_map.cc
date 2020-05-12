#include "cross_point_feature_creator.h"
#include "draw_shp_elements.h"
#include <iostream>
#include <fstream>


int main(int argc, char *argv[]){
    if(argc<3){
        std::cerr<<"Usage: shp_to_map shp_file cross_file_name"<<std::endl;
        exit(0);
    }


    std::ofstream fout(argv[2]);
    if(!fout.is_open()){
        std::cout<<"Cannot open file---"<<std::string(argv[2])<<", please check!!!"<<std::endl;
        return -1;
    }
    fout.precision(7);

    ShpDrawer my_shp_drawer(1, 30000, 30000);

    std::vector<rcll::CrossPointPtr> cross_pts;
    my_shp_drawer.DetectCrossPoints(argv[1], cross_pts, cv::Point2d(0, 0));

    // save cross points to file

    for(int i=0; i<cross_pts.size(); i++){
        fout<<*(cross_pts[i]);
    }
    fout.close();

    return 0;
}