/*
    对给定文件夹中的所有图像进行定位
*/

#include <fstream>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

#include "cross_point_feature_creator.h"
#include "locator.h"

#include "shp.h"

// debug
#include <glog/logging.h>
#include <sys/stat.h>
#include <time.h>

#include <dirent.h>
#include <string.h>  //包含strcmp的头文件,也可用: #include <ctring>

#include <algorithm>
#include <sstream>

void getFileNames(const std::string path, std::vector<std::string> &filenames,
                  const std::string suffix = "") {
  DIR *pDir;
  struct dirent *ptr;
  if (!(pDir = opendir(path.c_str()))) return;

  while ((ptr = readdir(pDir)) != 0) {
    if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
      std::string file = path + "/" + ptr->d_name;
      if (opendir(file.c_str())) {
        getFileNames(file, filenames, suffix);
      }

      else {
        if (suffix == file.substr(file.size() - suffix.size())) {
          filenames.push_back(file);
        }
      }
    }
  }
  closedir(pDir);

  std::sort(filenames.begin(), filenames.end());
}

int LoadMapFromFile(const std::string &filename, const cv::Point2d &offset,
                    const double scale,
                    std::vector<rcll::CrossPointPtr> &cross_pts) {
  std::ifstream fin(filename);
  if (!fin.is_open()) {
    std::cout << "Fail to open map file, please check!!!" << std::endl;
    return 0;
  }
  cv::Point2d center;
  std::vector<cv::Point2d> tangents;
  cv::Point2d tangent_tmp;
  int branch_num;
  int tangent_num;
  double tangent_fitting_error;
  while (!fin.eof()) {
    fin >> tangent_fitting_error;
    fin >> center.x >> center.y;
    fin >> branch_num;
    fin >> tangent_num;
    tangents.clear();
    for (int i = 0; i < tangent_num; ++i) {
      fin >> tangent_tmp.x >> tangent_tmp.y;
      tangents.push_back(tangent_tmp);
    }

    if (tangent_fitting_error < 1.0 && branch_num >= 3 && branch_num <= 5 &&
        tangent_num >= 2 && tangent_num <= 4) {
      rcll::CrossPointPtr cross_pt_tmp(new rcll::CrossPoint());
      center.x = (center.x + offset.x) / scale;
      center.y = (center.y + offset.y) / scale;

      cross_pt_tmp->ThinInit(center, branch_num, tangents,
                             tangent_fitting_error);
      cross_pts.push_back(cross_pt_tmp);
    }
  }
}

int main(int argc, char *argv[]) {
  // debug:显示定位结果
  // Shp ref_shp;
  // ref_shp.Init("17/road.shp");
  // Initialize Google's logging library.
  FLAGS_log_dir = "/media/li/flight_sim/log/";
  google::InitGoogleLogging(argv[0]);

  // 1. deal with params
  if (argc < 6) {
    std::cout
        << "Usage: single_img_localization [map_file_name] [ref_vor_file]"
           "  [shp_file] [query_img_dir] [result_file]..."
        << std::endl;
    exit(-1);
  }

  std::string map_file_name(argv[1]);
  std::string query_img_dir(argv[4]);
  std::vector<std::string> requry_image_names;
  getFileNames(query_img_dir, requry_image_names, "png");

  std::cout << requry_image_names.size() << " images found......" << std::endl;


  std::string result_file(argv[5]);

  clock_t start, end;
  start = clock();
  std::vector<rcll::CrossPointPtr> map;
  LoadMapFromFile(map_file_name, cv::Point2d(0,0), 1.0, map);
  end = clock();
  std::cout << (end - start) / (CLOCKS_PER_SEC / 1000.0)
            << " ms passed in loading map file with " << map.size()
            << " cross points" << std::endl;

  rcll::CrossPointFeatureParam cross_feature_detector_param;
  cross_feature_detector_param.branch_length = 30;
  cross_feature_detector_param.min_cross_point_distance = 4;
  cross_feature_detector_param.threshold = 10;
  cross_feature_detector_param.merge_angle_threshold = 10;

  // 2. Initilize locator
  rcll::LocatorParam locator_params;
  {
    locator_params.max_iterate_num = 600000;
    locator_params.max_requry_sample_try_num = 10;

    locator_params.cos_angle_distance = 0.99;  // cos(5°)
    locator_params.inliner_rate_threshold = 0.6;
    locator_params.min_inliner_point_num = 20;

    locator_params.max_inliner_cpt_distance = 40;  // 10 piexl
    locator_params.max_inliner_voi_distance = 50;
    locator_params.max_sample_distance = 3000;
    locator_params.threshold_diatance = 300;
    locator_params.min_requry_pt_num = 5;
    locator_params.max_cross_ratio_relative_error = 0.2;
    locator_params.max_tangent_error = 0.8;

    locator_params.location_success_confidence = 0.99;
    locator_params.query_location_confidence = 0.5;
    locator_params.query_location_success_possibility = 0.2;

    locator_params.query_sample_max_distance = 1500;
    locator_params.query_sample_min_diatance = 300;
    locator_params.query_sample_min_cos_angle_distance = 0.985;

    locator_params.random_grid_h_error_threshold = 200;
    locator_params.random_grid_size = 50;

    locator_params.icp_inlier_threshold_ = 200;
    locator_params.icp_precision_threshold_ = 0.005;
    locator_params.icp_max_iter_num_ = 100;

    locator_params.min_inlier_rate_ = 0.6;
    locator_params.sample_pt_max_distance_ = 5;
  }


  Shp ref_shp;
  ref_shp.Init(argv[3]);

  start = clock();
  rcll::Locator locator;
  locator.Init(map, argv[2], argv[3], locator_params,
               cross_feature_detector_param);
  end = clock();
  std::cout << (end - start) / (CLOCKS_PER_SEC / 1000.0)
            << " ms passed in  Initilize locator" << std::endl;

  // 3. Locate
  std::ofstream fout(result_file);
  for (int i = 0; i < requry_image_names.size(); i++) {
    cv::Mat query_img;
    double similarity;
    cv::Mat H;

    // debug
    std::cout << "\n\n******************** " << i + 1 << "/"
              << requry_image_names.size() << " Start locate image "
              << requry_image_names[i] << "***************************"
              << std::endl;
    LOG(INFO) << "\n\n******************** " << i + 1 << "/"
              << requry_image_names.size() << " Start locate image "
              << requry_image_names[i] << "***************************"
              << std::endl;

    start = clock();
    query_img = cv::imread(requry_image_names[i], 0);

    int success = locator.Locate(query_img, H, similarity);

    LOG(INFO) << "locate state: " << success << std::endl;

    end = clock();
    double t = (end - start) / (CLOCKS_PER_SEC / 1000.0);
    std::cout << t << " ms passed in  localization" << std::endl;

    // 输出的H是相对于地理坐标系的
    cv::Mat H_geo;
    if (success == -1 || H.empty())
      H_geo = cv::Mat::zeros(3, 3, CV_64F);
    else
      H_geo = H;

    fout << i << "    " << success << "    " << t << "    ";
    for (int j = 0; j < 9; j++)
      fout << H_geo.at<double>(j / 3, j % 3) << "    ";
    fout << std::endl;
  }

  return 0;
}