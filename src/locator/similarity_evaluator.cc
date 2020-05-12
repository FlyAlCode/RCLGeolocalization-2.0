#include "similarity_evaluator.h"
// #include <functional>
// #include <algorithm>
// #include <ctime>
#include "voronoi_surface.h"

// debug
#include <iostream>
// #include <glog/logging.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace rcll
{
void SimilarityEvaluator::Init(
	VoronoiMap *ref_vor_map, const std::vector<CrossPointPtr> &reference_pts)
{
	// 1. init voronoi surface
	ref_vor_map_.reset(ref_vor_map);

	// 2. init reference cross points
	cross_points_ = reference_pts;
	// Get all the center point to build the search tree
	std::vector<cv::Point2d> pt_centers;
	for (int i = 0; i < reference_pts.size(); i++)
	{
		pt_centers.push_back(reference_pts[i]->get_center());
	}

	search_tree_.reset(new RoadMapTree);
	search_tree_->BuildKDTree(pt_centers);
}

void SimilarityEvaluator::Init(
	const std::string &ref_vor_map_file,
	const std::vector<CrossPointPtr> &reference_pts)
{
	// 1. init voronoi surface
	ref_vor_map_.reset(new VoronoiMap);
	ref_vor_map_->ReadFromFile(ref_vor_map_file);

	// 2. init reference cross points
	cross_points_ = reference_pts;
	// Get all the center point to build the search tree
	std::vector<cv::Point2d> pt_centers;
	for (int i = 0; i < reference_pts.size(); i++)
	{
		pt_centers.push_back(reference_pts[i]->get_center());
	}

	search_tree_.reset(new RoadMapTree);
	search_tree_->BuildKDTree(pt_centers);
}

double SimilarityEvaluator::Evaluate(const cv::Mat &query_img, const cv::Mat H,
									 const double threshold)
{
	// get all road points
	int max_pt_num = 100000;
	int k = 0;
	cv::Mat query_pts_tmp(2, max_pt_num, CV_64F);
	for (int i = 0; i < query_img.rows; i += down_sample_rate)
	{
		for (int j = 0; j < query_img.cols; j += down_sample_rate)
		{
			if (query_img.at<uchar>(i, j) == 1)
			{
				query_pts_tmp.at<double>(0, k) = j;
				query_pts_tmp.at<double>(1, k) = i;
				++k;
			}
		}
	}
	cv::Mat query_pts = cv::Mat::ones(3, k, CV_64F);
	query_pts_tmp(cv::Rect(0, 0, k, 2)).copyTo(query_pts(cv::Rect(0, 0, k, 2)));
	cv::Mat query_ted_pts_h = H * query_pts;

	int inlier_count = 0;
	double error_sum = 0;
	double sum_pt_num = 0;
	for (int i = 0; i < k; ++i)
	{
		float geo_x =
			query_ted_pts_h.at<double>(0, i) / query_ted_pts_h.at<double>(2, i);
		float geo_y =
			query_ted_pts_h.at<double>(1, i) / query_ted_pts_h.at<double>(2, i);
		float voronoi_distance = ref_vor_map_->GetVoronoiDistance(geo_x, geo_y);
		
		++sum_pt_num;

		if (voronoi_distance < threshold && voronoi_distance >= 0)
			++inlier_count;
	}
	return double(inlier_count) / sum_pt_num;
}

double SimilarityEvaluator::InverseVorEvaluate(const cv::Mat &query_vor_img,
											   const cv::Mat H,
											   const double threshold)
{
	// debug
	// double t = clock();
	if (H.empty())
		return -1;
	
	std::vector<cv::Point2d> m_pts;
	cv::Vec4d bound;
	if (!GetProjectArea(H, query_vor_img.cols, query_vor_img.rows, m_pts, bound))
		return 0;
	double min_x, min_y, max_x, max_y;
	min_x = bound[0];
	max_x = bound[1];
	min_y = bound[2];
	max_y = bound[3];

	std::vector<cv::Point2d> inside_pts;
	for (int x = min_x; x <= max_x; x += 3)
	{
		for (int y = min_y; y < max_y; y += 3)
		{
			cv::Point2f current_pt(x, y);
			double d = ref_vor_map_->GetVoronoiDistance(x, y);
			if (d >= 0 && d <= ref_vor_map_->get_gsd())
			{
				if (IsInsidePoint(m_pts, current_pt))
				{
					inside_pts.push_back(current_pt);
				}
			}
		}
	}

	
	cv::Mat H_inv = H.inv();
	
	double inlier_num = 0;
	double inside_forward_num = 0;
	for (int i = 0; i < inside_pts.size(); ++i)
	{
		double z = H_inv.at<double>(2, 0) * inside_pts[i].x +
				   H_inv.at<double>(2, 1) * inside_pts[i].y +
				   H_inv.at<double>(2, 2);
	
		double x =
			(H_inv.at<double>(0, 0) * inside_pts[i].x +
			 H_inv.at<double>(0, 1) * inside_pts[i].y + H_inv.at<double>(0, 2)) / z;
		double y =
			(H_inv.at<double>(1, 0) * inside_pts[i].x +
			 H_inv.at<double>(1, 1) * inside_pts[i].y + H_inv.at<double>(1, 2)) / z;
		
		if (x < 0 || y < 0 || x >= query_vor_img.cols || y >= query_vor_img.rows)
			continue;
		++inside_forward_num;
		if (query_vor_img.at<float>(y, x) < threshold)
			++inlier_num;
	}

	
	if (inside_forward_num == 0)
		return 0;
	return inlier_num / inside_forward_num;
}

double SimilarityEvaluator::EvaluateSum(
	const cv::Mat &query_binary_img, const cv::Mat &query_vor_img,
	const std::vector<CrossPointPtr> &requry_points, const cv::Mat &H,
	double cpt_inlier_distance_threshold, double vor_inlier_distance_threshold,
	double half_inlier_rate_threshold)
{
	double cpt_sim = Evaluate(requry_points, H, cpt_inlier_distance_threshold);
	if (cpt_sim < half_inlier_rate_threshold)
		return cpt_sim;

	double vor_sim =
		Evaluate(query_binary_img, H, vor_inlier_distance_threshold) - 0.2;
	if (vor_sim < half_inlier_rate_threshold)
		return vor_sim;
	double inverse_vor_sim =
		InverseVorEvaluate(query_vor_img, H, vor_inlier_distance_threshold) - 0.2;

	double sim = vor_sim < cpt_sim ? vor_sim : cpt_sim;
	sim = sim < inverse_vor_sim ? sim : inverse_vor_sim;
	

	return sim;
}


double SimilarityEvaluator::Evaluate(
	const std::vector<CrossPointPtr> &requry_points, const cv::Mat &H,
	double threshold)
{
	std::vector<CrossPointPtr> current_matching_point_set;
	CrossPointPtr current_matching_point;
	int inlier_num = 0;
	int vaild_cpt_num = 0;
	for (int j = 0; j < requry_points.size(); j++)
	{
		if (requry_points[j]->get_tangent_error() > 1.0)
		{
			current_matching_point_set.push_back(nullptr);
			continue;
		}
		++vaild_cpt_num;
		if (IsInliner(requry_points[j], H, threshold,
					  current_matching_point))
		{ 
			bool matching_point_existed = false;
			for (int k = 0; k < current_matching_point_set.size(); k++)
			{
				if (current_matching_point != nullptr &&
					current_matching_point_set[k] != nullptr &&
					current_matching_point->get_id() ==
						current_matching_point_set[k]->get_id())
				{
					matching_point_existed = true;
					break;
				}
			}

			if (!matching_point_existed)
				++inlier_num;
			else
				current_matching_point = nullptr;
		}
		current_matching_point_set.push_back(current_matching_point);
	}
	return double(inlier_num) / vaild_cpt_num;
}

double SimilarityEvaluator::Evaluate(
	const std::vector<CrossPointPtr> &requry_points, const cv::Mat &H,
	int img_w, int img_h, double threshold)
{
	
	double q_pt_data[15] = {0, img_w, img_w, 0, img_w / 2.0,
							0, 0, img_h, img_h, img_h / 2.0,
							1, 1, 1, 1, 1};
	cv::Mat q_pt(3, 5, CV_64F, q_pt_data);
	cv::Mat m_pt_h = H * q_pt;
	std::vector<cv::Point2d> corners(5);
	for (int i = 0; i < 5; ++i)
	{
		corners[i].x = m_pt_h.at<double>(0, i) / m_pt_h.at<double>(2, i);
		corners[i].y = m_pt_h.at<double>(1, i) / m_pt_h.at<double>(2, i);
	}
	double d_max = 0;
	for (int i = 0; i < 4; ++i)
	{
		double d = cv::norm(corners[i] - corners[4]);
		if (d > d_max)
			d_max = d;
	}
	
	std::vector<cv::Point2d> center_pts;
	center_pts.push_back(corners[4]);

	std::vector<std::vector<std::pair<size_t, double>>> ret_matches;
	search_tree_->RadiusSearch(center_pts, d_max * d_max, ret_matches);
	CrossPointPtr transformed_point_tmp;
	double map_cpt_num = 0;
	corners.pop_back();
	for (int i = 0; i < ret_matches[0].size(); i++)
	{
		transformed_point_tmp = cross_points_[ret_matches[0][i].first];
		if (IsInsidePoint(corners, transformed_point_tmp->get_center()))
			++map_cpt_num;
	}
	if (std::abs(map_cpt_num - requry_points.size()) / requry_points.size() > 0.3)
		return 0;

	// 交叉点匹配
	std::vector<CrossPointPtr> current_matching_point_set;
	CrossPointPtr current_matching_point;
	int inlier_num = 0;
	int vaild_cpt_num = 0;
	for (int j = 0; j < requry_points.size(); j++)
	{
		if (requry_points[j]->get_tangent_error() > 1.0)
		{
			current_matching_point_set.push_back(nullptr);
			continue;
		}
		++vaild_cpt_num;
		if (IsInliner(requry_points[j], H, threshold,
					  current_matching_point))
		{ 
			bool matching_point_existed = false;
			for (int k = 0; k < current_matching_point_set.size(); k++)
			{
				if (current_matching_point != nullptr &&
					current_matching_point_set[k] != nullptr &&
					current_matching_point->get_id() ==
						current_matching_point_set[k]->get_id())
				{
					matching_point_existed = true;
					break;
				}
			}

			if (!matching_point_existed)
				++inlier_num;
			else
				current_matching_point = nullptr;
		}
		current_matching_point_set.push_back(current_matching_point);
	}
	return double(inlier_num) / vaild_cpt_num;
}

bool SimilarityEvaluator::IsInliner(const CrossPointPtr &requry_point,
									const cv::Mat &H, const double threshold,
									CrossPointPtr &matching_point)
{
	// 1. Transform requry_point to destination coordination
	cv::Point2d center = requry_point->get_center();
	cv::Mat center_homograph(3, 1, CV_64F);
	center_homograph.at<double>(0, 0) = center.x;
	center_homograph.at<double>(1, 0) = center.y;
	center_homograph.at<double>(2, 0) = 1;

	cv::Mat transformed_center_homograph = H * center_homograph;
	cv::Point2d transformed_center;
	double x = transformed_center_homograph.at<double>(0, 0);
	double y = transformed_center_homograph.at<double>(1, 0);
	double z = transformed_center_homograph.at<double>(2, 0);
	transformed_center.x = x / z;
	transformed_center.y = y / z;

	// 2. search for points within threshold
	std::vector<cv::Point2d> requry_pts;
	requry_pts.push_back(transformed_center);

	std::vector<std::vector<std::pair<size_t, double>>> ret_matches;
	search_tree_->RadiusSearch(requry_pts, threshold * threshold, ret_matches);
	// 3. check type
	CrossPointPtr transformed_point_tmp;
	
	for (int i = 0; i < ret_matches[0].size(); i++)
	{
		transformed_point_tmp = cross_points_[ret_matches[0][i].first];
		if (transformed_point_tmp->get_braches_num() ==
				requry_point->get_braches_num() &&
			transformed_point_tmp->get_tangents_num() ==
				requry_point->get_tangents_num())
		{
			matching_point = transformed_point_tmp;
			return true;
		}
	}

	matching_point = nullptr;
	return false;
}

bool SimilarityEvaluator::IsInsidePoint(const std::vector<cv::Point2d> &corners,
										const cv::Point2d &query_pt)
{
	double cross = (corners[0] - corners.back()).cross(query_pt - corners.back());
	if (cross == 0)
		return true;
	for (int i = 0; i < corners.size() - 1; ++i)
	{
		double cross_tmp =
			(corners[i + 1] - corners[i]).cross(query_pt - corners[i]);
		if (cross_tmp == 0)
			return true;
		if ((cross > 0 && cross_tmp < 0) || (cross < 0 && cross_tmp > 0))
			return false;
	}
	return true;
}

bool SimilarityEvaluator::GetProjectArea(const cv::Mat &H, int img_w, int img_h,
										 std::vector<cv::Point2d> &map_corners,
										 cv::Vec4d &bound)
{
	cv::Mat H_w_c = H.inv();
	cv::Mat h1_cross_h2 = H_w_c.col(0).cross(H_w_c.col(1));
	h1_cross_h2 = -h1_cross_h2 /
				  std::abs(h1_cross_h2.at<double>(2, 0)); // 归一化, 同时反转符号

	const double lamda_threshold = -0.5;
	double x[4] = {0, img_w, img_w, 0};
	double y[4] = {0, 0, img_h, img_h};
	double h_cross_0 = h1_cross_h2.at<double>(0, 0);
	double h_cross_1 = h1_cross_h2.at<double>(1, 0);
	double h_cross_2 = h1_cross_h2.at<double>(2, 0);
	double lamda[4];
	for (int i = 0; i < 4; ++i)
	{
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

	
	map_corners.clear();
	for (int i = 0; i < corners.size(); ++i)
	{
		double z = H.at<double>(2, 0) * corners[i].x +
				   H.at<double>(2, 1) * corners[i].y + H.at<double>(2, 2);
		map_corners.push_back(
			cv::Point2d((H.at<double>(0, 0) * corners[i].x +
						 H.at<double>(0, 1) * corners[i].y + H.at<double>(0, 2)) /
							z,
						(H.at<double>(1, 0) * corners[i].x +
						 H.at<double>(1, 1) * corners[i].y + H.at<double>(1, 2)) /
							z));
		if (i == 0)
		{
			bound[0] = map_corners[i].x;
			bound[1] = map_corners[i].x;
			bound[2] = map_corners[i].y;
			bound[3] = map_corners[i].y;
		}
		else
		{
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

	if (bound[1] - bound[0] > 8 * img_w || bound[3] - bound[2] > 8 * img_h)
		return false;
	return true;
}

} // namespace rcll
