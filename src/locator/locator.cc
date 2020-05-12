#include "locator.h"
#include <cmath>
#include <opencv2/calib3d/calib3d.hpp> 
#include "cross_point_feature_creator.h"   
#include "voronoi_surface.h"   


// used just for debug
#include <glog/logging.h> 
#include <time.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace rcll{
    
void Locator::Init(const std::vector<CrossPointPtr> &reference_map, 
                   VoronoiMap *ref_vor_map, 
                   const std::vector<cv::Point2f> & map_pts,
                   const LocatorParam &param ,
                   const CrossPointFeatureParam &cross_feature_detector_param){
    // 1. Initilize sampler
    requry_sampler_.reset(new RequryPtSampler());
    
    map_sampler_.reset(new MapPtSampler());
    map_sampler_->Initialize(reference_map,
                             param.cos_angle_distance,
                             param.max_sample_distance);
    priority_index_ = map_sampler_->GetSamplePriorityIndex();
    
    // 2. Initilize estimator
    H_estimator_.reset(new HomographyEstimator);
    
    // 3. Initialize similarity evaluator
    similarity_evaluator_.reset(new SimilarityEvaluator);
    similarity_evaluator_->Init(ref_vor_map, reference_map);

    // 4. Initilize random grid
    cv::Vec4f bound = similarity_evaluator_->get_area_bound();
    rand_grid_.reset(new RandomGrid2D);
    rand_grid_->Init(param.random_grid_size, bound[0], bound[2], bound[1], bound[3]);

    // 5. Initilize consistency checker
    consistency_checker_.reset(new ConsistencyChecker);
    consistency_checker_->Init(map_pts, 
                                param.icp_inlier_threshold_,
                                param.icp_precision_threshold_,
                                param.icp_max_iter_num_,
                                param.sample_pt_max_distance_,
                                param.min_inlier_rate_);
    
    // 6. Save all the params
    locator_params_ = param;
    cross_feature_detector_param_ = cross_feature_detector_param;
}

void Locator::Init(const std::vector<CrossPointPtr> &reference_map,
                const std::string &ref_vor_map_file, 
                const std::string &shp_file,
                const LocatorParam &param ,
                const CrossPointFeatureParam &cross_feature_detector_param ){
    std::cout<<"Initilizing......"<<std::endl;
    // 1. Initilize sampler
    requry_sampler_.reset(new RequryPtSampler());
    
    map_sampler_.reset(new MapPtSampler());
    map_sampler_->Initialize(reference_map,
                             param.cos_angle_distance,
                             param.max_sample_distance);
    priority_index_ = map_sampler_->GetSamplePriorityIndex();
    std::cout<<"    Finishing initilizing sampler..."<<std::endl;
    // 2. Initilize estimator
    H_estimator_.reset(new HomographyEstimator);
    std::cout<<"    Finishing initilizing estimator..."<<std::endl;
    
    // 3. Initialize similarity evaluator
    similarity_evaluator_.reset(new SimilarityEvaluator);
    similarity_evaluator_->Init(ref_vor_map_file, reference_map);
    std::cout<<"    Finishing initilizing similarity evaluator..."<<std::endl;
    
    // 4. Initilize random grid
    cv::Vec4f bound = similarity_evaluator_->get_area_bound();
    rand_grid_.reset(new RandomGrid2D);
    // use a big grid size to avoid missing the right model
    rand_grid_->Init(param.random_grid_size, bound[0], bound[2], bound[1], bound[3]);
    std::cout<<"    Finishing initilizing random grid..."<<std::endl;

    // 5. Initilize consistency checker
    consistency_checker_.reset(new ConsistencyChecker);
    consistency_checker_->Init(shp_file, 
                                param.icp_inlier_threshold_,
                                param.icp_precision_threshold_,
                                param.icp_max_iter_num_,
                                param.sample_pt_max_distance_,
                                param.min_inlier_rate_); 
    std::cout<<"    Finishing initilizing consistency checker..."<<std::endl;
    // 6. Save all the params
    locator_params_ = param;
    cross_feature_detector_param_ = cross_feature_detector_param;
    std::cout<<"All initilizing works have been done!!!"<<std::endl;
}

void Locator::SetRequryImg(const cv::Mat& img){
    img.copyTo(requry_img_);
}

void Locator::SetMapImg(const cv::Mat& img){
    img.copyTo(map_img_);
}



int Locator::Locate( const cv::Mat &query_road_map,
                    cv::Mat &best_H,
                    double &best_similarity){ 
    // 0. reset random grid
    double t_start = clock();
    rand_grid_->Reset();
    best_similaritys_.clear();
    best_models_.clear();

    // 1. detect cross pts and get thined road map   
    std::vector<CrossPointPtr> query_cpts;
    cv::Mat query_thined_map;
    CreateCrossPointFeatures(query_road_map, cross_feature_detector_param_, query_cpts, query_thined_map); 

    // 1.5. compute voronoi surface
    VoronoiSurfaceGenerator vor_generator;
    cv::Mat query_vor_surface;
    vor_generator.Generate(query_thined_map, cv::Vec4f(0,0,0,0), query_vor_surface);
    
    // 2. 进入循环：采样-模型估计-模型验证       
    if(query_cpts.size()<locator_params_.min_requry_pt_num)
        return -1;
    
    cv::Mat current_H;  
    std::vector<CrossPointPtr> current_matching_points;
    double current_similarity_cpts = 0;
    double current_similarity_voi = 0;
    double inverse_vor_similarity = 0;
    double current_similarity = 0;
    double current_consistent_num = 0;
    best_similarity = 0;
    int run_round = 0;
    
    std::vector<CrossPointPtr> sample_map_pts;
    std::vector<cv::Mat> HMats;
    
    ////////////////////////////
    std::vector<cv::Point3d> query_tangents;
    std::vector<cv::Point3d> map_tangnets;
    
    double p_fail_sum = 1 - locator_params_.location_success_confidence;
    double p_fail_map = 1 - locator_params_.query_location_confidence;
    double p_query_success = locator_params_.query_location_success_possibility;
    int min_query_sample_num = 1.5 * std::log10(p_fail_sum)/                    // 1.5 more than regular sample
                                            std::log10(1 - p_query_success + p_query_success * p_fail_map);    
    int query_sample_num = 0;

    // debug
    std::cout<<"min_query_sample_num = "<<min_query_sample_num<<std::endl;
    
    // debug
    clock_t start, end;
    // start = clock();
    requry_sampler_->Initilize(query_cpts, 
                               priority_index_,
                               locator_params_.query_sample_min_diatance,
                               locator_params_.query_sample_max_distance,
                               locator_params_.query_sample_min_cos_angle_distance,
                               locator_params_.max_tangent_error);
        
    CrossPtTuple query_sample_data;
    int total_round_num = 0;
    // debug
    int collision_num = 0;     
    while(query_sample_num <= min_query_sample_num){
        // 1. sample query points 
        int result = requry_sampler_->Sample(query_sample_data);
               
        if(result == -1){
            LOG(INFO)<<"Return with all query sampled"<<std::endl;
            return 0;                       // all datas have been sample with no correct datas, which means location fails
        }
            
        if(result == 0)
            continue;
        
     
        CrossPtTuple map_sample_data;
        int match_type;
        int min_map_sample_num = 100;           
        int map_sample_num = 0;
        int collision_success_every_n = 0;          // 多少次碰撞检测到一次成功碰撞
        std::vector<int> collision_ids;
        while(map_sample_num++ < min_map_sample_num){
            // 2. sample map points
            int current_match_num = map_sampler_->Sample(query_sample_data,
                                                         locator_params_.max_cross_ratio_relative_error,
                                                         map_sample_data,
                                                         match_type,
                                                         !(map_sample_num-1));

            
            if(current_match_num == 0)
                break;
            
            if(map_sample_num==1){
                min_map_sample_num = current_match_num * locator_params_.query_location_confidence;
                LOG(INFO)<<"min_map_sample_num = "<<min_map_sample_num<<std::endl;
            }
           
            
            // 3. calculate model
            map_sample_data.GetSortedTangents(0, map_tangnets);
            query_sample_data.GetSortedTangents(match_type, query_tangents);
            
            cv::Mat tmp_H = H_estimator_->findHomography(query_tangents, map_tangnets);
            
            if(tmp_H.empty() || !IsHVaild(tmp_H, query_road_map.cols, query_road_map.rows, 10))
                continue;

            // 3.5 collision check for new model
            std::shared_ptr<RandomGrid2D::Model> new_model(new RandomGrid2D::Model);
            memcpy(new_model->H_mat, tmp_H.data, sizeof(double)*9);
            cv::Vec4d pt_tmp = query_sample_data.get_centers();
            for(int pt_i = 0; pt_i<4; ++pt_i)
                new_model->match_pts[pt_i] = pt_tmp[pt_i];
            pt_tmp = map_sample_data.get_centers();
            for(int pt_i = 0; pt_i<4; ++pt_i)
                new_model->match_pts[pt_i+4] = pt_tmp[pt_i];
            new_model->parent_id_ = query_sample_num;

            cv::Point2d collision_corner;
            if(!CalBestProjectCorner(tmp_H, query_road_map.cols, query_road_map.rows, collision_corner))
                continue;                           

            int collision_id_num = rand_grid_->InsertNewModel(new_model, 
                                                             collision_corner.x,
                                                             collision_corner.y,
                                                             collision_ids,
                                                             locator_params_.random_grid_h_error_threshold);
        
            // 4. similarity evaluate            
            if(collision_id_num==0){
                // debug
                ++collision_success_every_n;
                ++total_round_num;
            }
            else{
                ++total_round_num;
                DLOG(INFO)<<"collision_success_every_n = "<<collision_success_every_n<<std::endl;
                collision_success_every_n = 0;
                                                
                collision_num += collision_ids.size();
               
                current_similarity = similarity_evaluator_->EvaluateSum(query_thined_map,
                                                                        query_vor_surface,
                                                                        query_cpts,
                                                                        tmp_H,
                                                                        locator_params_.max_inliner_cpt_distance,
                                                                        locator_params_.max_inliner_voi_distance,
                                                                        locator_params_.inliner_rate_threshold * 0.5);
                
                // ICP
                if(current_similarity>=locator_params_.inliner_rate_threshold/2.0 && 
                    current_similarity<locator_params_.inliner_rate_threshold){
            
                    cv::Point2f anchor_pt((new_model->match_pts[0]+new_model->match_pts[2])/2.0,
                                        (new_model->match_pts[1]+new_model->match_pts[3])/2.0);
                    if(!consistency_checker_->BootstrapRun(query_thined_map, anchor_pt, tmp_H, tmp_H, current_similarity))
                        continue;
                    
                    current_similarity = similarity_evaluator_->EvaluateSum(query_thined_map,
                                                                        query_vor_surface,
                                                                        query_cpts,
                                                                        tmp_H,
                                                                        locator_params_.max_inliner_cpt_distance,
                                                                        locator_params_.max_inliner_voi_distance,
                                                                        locator_params_.inliner_rate_threshold * 0.5);
                }                

                if(current_similarity>best_similarity){
                    best_similarity = current_similarity;
                    tmp_H.copyTo(best_H); 
                    if(best_similarity>locator_params_.inliner_rate_threshold){
                        LOG(INFO)<<"query_sample_num = "<<query_sample_num<<std::endl;
                        LOG(INFO)<<"collision_num = "<<collision_num<<std::endl;
                        return 1;
                    } 
                }
                            
            }

            if(total_round_num>locator_params_.max_iterate_num){
                // std::cout<<"total_round_num = "<<total_round_num<<std::endl;
                LOG(INFO)<<"best_similarity = "<<best_similarity<<std::endl;
                LOG(INFO)<<"Return with too many try"<<std::endl;
                return 0;                
            }
        }
        ++query_sample_num;
    }

    LOG(INFO)<<"Return with max query samples."<<std::endl;
    return 0; 
}


void Locator::ShowSample(const std::string& window_name, 
                            const std::vector< CrossPointPtr> &samplers, 
                            int kind, 
                            int wait_time){
    cv::Mat result_img;
    if(kind==0)
        map_img_.copyTo(result_img);
    else
        requry_img_.copyTo(result_img);
    if(samplers.size()>=2){
        samplers[0]->Draw(result_img, cv::Scalar(255,0,0));
        samplers[1]->Draw(result_img, cv::Scalar(255,255,0));
    }
    std::cout<<"Press any key to continue..."<<std::endl;
    cv::resize(result_img, result_img, cv::Size(1000, 1000));
    cv::imshow(window_name, result_img);
    cv::waitKey(wait_time);
}


bool Locator::IsHVaild(const cv::Mat &H, double img_w, double img_h,  double threshold){
	cv::Mat H_w_c = H.inv();
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

    std::vector<cv::Point2d> map_corners;
    cv::Vec4d bound;
	for (int i = 0; i < corners.size(); ++i) {
		double z = H.at<double>(2, 0) * corners[i].x +
				   H.at<double>(2, 1) * corners[i].y + H.at<double>(2, 2);
		map_corners.push_back(
			cv::Point2d((H.at<double>(0, 0) * corners[i].x +
						 H.at<double>(0, 1) * corners[i].y + H.at<double>(0, 2)) /
							z,
						(H.at<double>(1, 0) * corners[i].x +
						 H.at<double>(1, 1) * corners[i].y + H.at<double>(1, 2)) /
							z));
		if (i == 0) {
			bound[0] = map_corners[i].x;
			bound[1] = map_corners[i].x;
			bound[2] = map_corners[i].y;
			bound[3] = map_corners[i].y;
		}
		else{
			if (bound[0] > map_corners[i].x)
				bound[0] = map_corners[i].x;
			if (bound[1] < map_corners[i].x)
				bound[1] = map_corners[i].x;
			if (bound[2] > map_corners[i].y)
				bound[2] = map_corners[i].y;
			if (bound[3] < map_corners[i].y)
				bound[3] = map_corners[i].y;
		}
	}

    double dx = bound[1]-bound[0];
    double dy = bound[3]-bound[2];
    if(dx > threshold*img_w || dx <img_w/threshold || dy > threshold*img_h || dy < img_h/threshold)
        return false;
    return true; 
}


bool Locator::InsertModel(const cv::Mat &new_model, double new_similarity){
    bool is_better_model = false;
    if(best_similaritys_.empty()){
        best_similaritys_.push_back(new_similarity);
        best_models_.push_back(cv::Mat());
        new_model.copyTo(best_models_.back());
        is_better_model = true;
    }
    else{
        std::list<cv::Mat>::iterator model_it = best_models_.begin();
        for(std::list<double>::iterator sim_it = best_similaritys_.begin();    
            sim_it!=best_similaritys_.end(); ++sim_it, ++model_it){
            if(*sim_it<new_similarity){
                best_similaritys_.insert(sim_it, new_similarity);
                std::list<cv::Mat>::iterator model_tmp_it = best_models_.insert(model_it, cv::Mat());
                new_model.copyTo(*model_tmp_it);
                is_better_model = true;
                break;
            }
        }
    }
    
    if(best_models_.size()>MODEL_NUM_){
        best_similaritys_.pop_back();
        best_models_.pop_back();
    }
    return is_better_model;
}


bool Locator::CalBestProjectCorner(const cv::Mat &H, int img_w, int img_h, cv::Point2d &corner){
	cv::Mat H_w_c = H.inv();
	cv::Mat h1_cross_h2 = H_w_c.col(0).cross(H_w_c.col(1));
	h1_cross_h2 = -h1_cross_h2 /
				  std::abs(h1_cross_h2.at<double>(2, 0)); // 归一化, 同时反转符号
	
	double x[4] = {0, img_w, img_w, 0};
	double y[4] = {0, 0, img_h, img_h};
	double h_cross_0 = h1_cross_h2.at<double>(0, 0);
	double h_cross_1 = h1_cross_h2.at<double>(1, 0);
	double h_cross_2 = h1_cross_h2.at<double>(2, 0);
	double lamda[4];
    double max_lamda=0;
    int max_lamda_index = -1;
	for (int i = 0; i < 4; ++i)	{
		lamda[i] = h1_cross_h2.at<double>(0, 0) * x[i] +
				   h1_cross_h2.at<double>(1, 0) * y[i] +
				   h1_cross_h2.at<double>(2, 0);
        if(lamda[i]<=lamda_threshold && lamda[i]>max_lamda_index){
            max_lamda = lamda[i];
            max_lamda_index = i;
        }
	}
    if(max_lamda_index==-1)
        return false;

	corner.x = x[max_lamda_index];
    corner.y = y[max_lamda_index];
    return true;
}
    
}   // namespace rcll
