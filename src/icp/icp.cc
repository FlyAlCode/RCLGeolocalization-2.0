#include "icp.h"
#include "homography.h"

/*  Used just for debug */
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>

namespace rcll {
void ICPOptimizer::SetParams(double inlier_threshold,
		double precision_threshold,
		int max_iter_num){
	inlier_threshold_ = inlier_threshold;
	precision_threshold_ = precision_threshold;
	max_iter_num_ = max_iter_num;
}

void ICPOptimizer::Init(const std::vector<cv::Point2f> &ref_pts,
              double inlier_threshold,
              double precision_threshold,
              int max_iter_num){
	SetParams(inlier_threshold, precision_threshold, max_iter_num);

	// save ref_pts
	ref_pts_ = ref_pts;
	ref_pts_mat_ = cv::Mat::zeros(ref_pts.size(), 2, CV_32F);
	for (int i = 0; i < ref_pts.size(); ++i){
		ref_pts_mat_.at<float>(i, 0) = ref_pts[i].x;
		ref_pts_mat_.at<float>(i, 1) = ref_pts[i].y;
	}
	ref_pt_index_.reset(new cv::flann::Index(ref_pts_mat_, cv::flann::KDTreeIndexParams()));
}


bool ICPOptimizer::Run(const std::vector<cv::Point2f> &query_pts,
			const cv::Point2f &query_anchor_pt,
			const cv::Mat &initial_h,
			cv::Mat &result_h,
			double &inlier_rate){
	int iter_num = 0;
	double last_cost = 0, current_cost = 0, cost_change_rate = 1;
	std::vector<cv::Point2d> match_ref_pts, match_query_pts;
	std::vector<double> match_weights;
	cv::Mat last_h;
	initial_h.copyTo(last_h);
	cv::Mat current_h;
	std::vector<uchar> is_inliers;
	double current_inlier_threshold = inlier_threshold_;		
	int precision_vaild_num = 0;					
	const int max_precision_vaild_num = 3;			
	while(iter_num<=max_iter_num_  && precision_vaild_num<max_precision_vaild_num){
		if(last_h.empty())
			return false;
		GetMatches(query_pts, last_h, current_inlier_threshold, match_ref_pts, match_query_pts);

		
		current_cost = CalHomography(match_query_pts, match_ref_pts, current_h);
		current_cost /= match_query_pts.size();
		
		if(iter_num==0){
			cost_change_rate = 1;
			last_cost = current_cost;
		}
		else{
			cost_change_rate = std::abs(current_cost - last_cost) / last_cost;
		}
		
		last_cost = current_cost;
		current_h.copyTo(last_h);
		++iter_num;
		current_inlier_threshold = inlier_threshold_ ;

		if(cost_change_rate<precision_threshold_){
			++precision_vaild_num;
		}
		else{
			precision_vaild_num = 0;
		}
	}
	current_h.copyTo(result_h);

	
	inlier_rate = GetInlierRatio(query_pts, result_h, inlier_threshold_ / 4.0);
	
	return precision_vaild_num >= max_precision_vaild_num;
}


void ICPOptimizer::GetMatches(const std::vector<cv::Point2f> &query_pts,
					const cv::Mat &h_,
					double inlier_threshold,
                    std::vector<cv::Point2d> &match_ref_pts,
                    std::vector<cv::Point2d> &match_query_pts){
	double t_start = clock();
	match_query_pts.clear();
	match_ref_pts.clear();

	cv::Mat h;
	if(h_.type()!=CV_32F)
		h_.convertTo(h, CV_32F);
	else
		h_.copyTo(h);

	cv::Mat index, distance;
	cv::Mat transformed_query_pts_mat = cv::Mat::zeros(query_pts.size(), 2, CV_32F);
	for (int i = 0; i < query_pts.size(); ++i){
		float z = h.at<float>(2, 0) * query_pts[i].x + h.at<float>(2, 1) * query_pts[i].y + h.at<float>(2,2);
		if(z==0)
			continue;
		transformed_query_pts_mat.at<float>(i, 0) = (h.at<float>(0, 0) * query_pts[i].x +
									  h.at<float>(0, 1) * query_pts[i].y + h.at<float>(0, 2)) / z;
									
		transformed_query_pts_mat.at<float>(i, 1) = (h.at<float>(1, 0) * query_pts[i].x +
									  h.at<float>(1, 1) * query_pts[i].y + h.at<float>(1, 2)) / z;							 
	}

	ref_pt_index_->knnSearch(transformed_query_pts_mat, index, distance, 1);
	
	for (int i = 0; i < query_pts.size(); ++i){
		if(distance.at<float>(i,0)<inlier_threshold){
			match_query_pts.push_back(query_pts[i]);
			match_ref_pts.push_back(ref_pts_[index.at<int>(i, 0)]);
		}
	}	
}


double ICPOptimizer::GetInlierRatio(const std::vector<cv::Point2f> &query_pts,
                        const cv::Mat &h_, 
                        double inlier_threshold){
	cv::Mat h;
	if(h_.type()!=CV_32F)
		h_.convertTo(h, CV_32F);
	else
		h_.copyTo(h);

	cv::Mat index, distance;
	cv::Mat transformed_query_pts_mat = cv::Mat::zeros(query_pts.size(), 2, CV_32F);
	for (int i = 0; i < query_pts.size(); ++i){
		float z = h.at<float>(2, 0) * query_pts[i].x + h.at<float>(2, 1) * query_pts[i].y + h.at<float>(2,2);
		if(z==0)
			continue;
		transformed_query_pts_mat.at<float>(i, 0) = (h.at<float>(0, 0) * query_pts[i].x +
									  h.at<float>(0, 1) * query_pts[i].y + h.at<float>(0, 2)) / z;
									
		transformed_query_pts_mat.at<float>(i, 1) = (h.at<float>(1, 0) * query_pts[i].x +
									  h.at<float>(1, 1) * query_pts[i].y + h.at<float>(1, 2)) / z;							 
	}
	ref_pt_index_->knnSearch(transformed_query_pts_mat, index, distance, 1);

	int inlier_num = 0;
	for (int i = 0; i < query_pts.size(); ++i){
		if(distance.at<float>(i,0)<inlier_threshold){
			++inlier_num;
		}
	}
	return inlier_num / double(query_pts.size());
}

double ICPOptimizer::GetAverageMatchError(const std::vector<cv::Point2f> &query_pts,
                                const cv::Mat &h_,
								double inlier_rate){
	cv::Mat h;
	if(h_.type()!=CV_32F)
		h_.convertTo(h, CV_32F);
	else
		h_.copyTo(h);

	cv::Mat index, distance;
	cv::Mat transformed_query_pts_mat = cv::Mat::zeros(query_pts.size(), 2, CV_32F);
	for (int i = 0; i < query_pts.size(); ++i){
		float z = h.at<float>(2, 0) * query_pts[i].x + h.at<float>(2, 1) * query_pts[i].y + h.at<float>(2,2);
		if(z==0)
			continue;
		transformed_query_pts_mat.at<float>(i, 0) = (h.at<float>(0, 0) * query_pts[i].x +
									  h.at<float>(0, 1) * query_pts[i].y + h.at<float>(0, 2)) / z;
									
		transformed_query_pts_mat.at<float>(i, 1) = (h.at<float>(1, 0) * query_pts[i].x +
									  h.at<float>(1, 1) * query_pts[i].y + h.at<float>(1, 2)) / z;							 
	}
	ref_pt_index_->knnSearch(transformed_query_pts_mat, index, distance, 1);

	std::vector<double> distance_v;
	for (int i = 0; i < distance.rows; ++i)
		distance_v.push_back(distance.at<float>(i, 0));
	std::sort(distance_v.begin(), distance_v.end());

	double sum_error = 0;
	int vaild_pt_num = distance_v.size() * inlier_rate;
	for (int i = 0; i < vaild_pt_num; ++i){
		sum_error += distance_v[i];
	}
	return sum_error / vaild_pt_num;


	
}

/*************************************************
 *                    private                    *
 *************************************************/
void ICPOptimizer::GetMatches(const std::vector<cv::Point2f> &query_pts,
                    const std::vector<double> &weights,
					const cv::Mat &h_,
					double inlier_threshold,
                    std::vector<cv::Point2d> &match_ref_pts,
                    std::vector<cv::Point2d> &match_query_pts,
                    std::vector<double> &match_weights){
	match_query_pts.clear();
	match_ref_pts.clear();
	match_weights.clear();

	cv::Mat h;
	if(h_.type()!=CV_32F)
		h_.convertTo(h, CV_32F);
	else
		h_.copyTo(h);

	cv::Mat index, distance;
	cv::Mat transformed_query_pts_mat = cv::Mat::zeros(query_pts.size(), 2, CV_32F);
	for (int i = 0; i < query_pts.size(); ++i){
		float z = h.at<float>(2, 0) * query_pts[i].x + h.at<float>(2, 1) * query_pts[i].y + h.at<float>(2,2);
		if(z==0)
			continue;
		transformed_query_pts_mat.at<float>(i, 0) = (h.at<float>(0, 0) * query_pts[i].x +
									  h.at<float>(0, 1) * query_pts[i].y + h.at<float>(0, 2)) / z;
									
		transformed_query_pts_mat.at<float>(i, 1) = (h.at<float>(1, 0) * query_pts[i].x +
									  h.at<float>(1, 1) * query_pts[i].y + h.at<float>(1, 2)) / z;							 
	}
	ref_pt_index_->knnSearch(transformed_query_pts_mat, index, distance, 1);

	for (int i = 0; i < query_pts.size(); ++i)
	{
		if (distance.at<float>(i, 0) < inlier_threshold )
		{
			match_query_pts.push_back(query_pts[i]);
			match_ref_pts.push_back(ref_pts_[index.at<int>(i, 0)]);
		}
	}		
}
} // namespace rcll 

