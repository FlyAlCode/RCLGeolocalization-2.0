#include "cross_pt_tuple.h"
#include "cross_ratio.h"
#include "cross_point_tree.h"
#include <algorithm> 

namespace rcll {
    
bool CrossPtTuple::Init(const CrossPointPtr& pt1, const CrossPointPtr& pt2){
    if(pt1==nullptr || pt2==nullptr)
        return false;
         
    tuple_data_.clear();
    tuple_data_.push_back(pt1);
    tuple_data_.push_back(pt2);
    feature_[0] = pt1->get_braches_num();
    feature_[1] = pt2->get_braches_num();
    feature_[2] = pt1->get_tangents_num();
    feature_[3] = pt2->get_tangents_num();
    
    if(feature_[0]<3 || feature_[0]>5 || feature_[1]<3 || feature_[1]>5 
        ||feature_[2]<2 || feature_[2]>4 || feature_[3]<2 || feature_[3]>4)
        return false;
    
    hash_key_ = CalculateHashKey();
    
    // debug
    // if(feature_[0]<3){
    //     std::cout<<feature_[0]<<"  "<<feature_[1]<<"  "<<feature_[2]<<"  "<<feature_[3]<<std::endl;
    //     std::cout<<hash_key_<<std::endl<<std::endl;
    // }

    return true;
}

bool CrossPtTuple::Init(const std::vector< CrossPointPtr >& data){
    if(data[0]==nullptr || data[1] ==nullptr)
        return false;
       
    tuple_data_.clear();
    tuple_data_.push_back(data[0]);
    tuple_data_.push_back(data[1]);
    feature_[0] = data[0]->get_braches_num();
    feature_[1] = data[1]->get_braches_num();
    feature_[2] = data[0]->get_tangents_num();
    feature_[3] = data[1]->get_tangents_num();
    
    if(feature_[0]<3 || feature_[0]>5 || feature_[1]<3 || feature_[1]>5 
        ||feature_[2]<2 || feature_[2]>4 || feature_[3]<2 || feature_[3]>4)
        return false;
    
    hash_key_ = CalculateHashKey();
    return true;
}

bool CrossPtTuple::CheckValidity(const double threshold_diatance,
                                 const double cos_angle_distance){
    CrossPointPtr pt1 = tuple_data_[0];
    CrossPointPtr pt2 = tuple_data_[1];
    cv::Point2d relative_pt(pt1->get_center()-pt2->get_center());
    double relative_pt_norm = cv::norm(relative_pt);
    if(relative_pt_norm <threshold_diatance)
        return false;
    
    std::vector<cv::Point2d > pt1_tangents = pt1->get_tangents();
    std::vector<cv::Point2d > pt2_tangents = pt2->get_tangents();
    for(int i=0; i<pt1_tangents.size(); i++){
        // cos<>, norms for pt1_tangents are 1, so no need to calculate anymore
        if(cv::abs(relative_pt.dot(pt1_tangents[i]))/relative_pt_norm > cos_angle_distance)  
            return false;
    }
    
    for(int i=0; i<pt2_tangents.size(); i++){
        if(cv::abs(relative_pt.dot(pt2_tangents[i]))/relative_pt_norm > cos_angle_distance)  
            return false;
    }
    
    return true;
}

// here, base_line is supposed to have been normalized
void CrossPtTuple::SortTangentsFromBaseLine(const CrossPointPtr& pt, 
                                       const cv::Point2d& base_line, 
                                       std::vector< cv::Point2d >& clockwise_sorted_lines) const{
    clockwise_sorted_lines.clear();
    
    clockwise_sorted_lines = pt->get_tangents();
    
    // debug
    //     if(clockwise_sorted_lines.size()>2)
    //         int end = 0;
    
    // calculate the angle between the x coordination and all the tangents
    std::vector<double> cos_angle;
    for(int i=0; i<clockwise_sorted_lines.size(); i++){
        double cross_tmp = base_line.cross(clockwise_sorted_lines[i]);
        double dot_tmp = base_line.dot(clockwise_sorted_lines[i]);
        if(cross_tmp>=0)
            cos_angle.push_back(dot_tmp);
        else
            cos_angle.push_back(-dot_tmp);
    }
    
    // optimized bubble sorting
    int start=0, end=cos_angle.size();
    while(start<end){
        int index = start;
        int current_end = start;
        while(++index<end){                             
            if(cos_angle[index-1]<cos_angle[index]){
                double tmp_cos_angle = cos_angle[index];
                cos_angle[index] = cos_angle[index-1];
                cos_angle[index-1] = tmp_cos_angle;
                
                cv::Point2d tangent_tmp = clockwise_sorted_lines[index];
                clockwise_sorted_lines[index] = clockwise_sorted_lines[index-1];
                clockwise_sorted_lines[index-1] = tangent_tmp;
                
                current_end = index;
            }
        }
        end = current_end;
    }
    
    clockwise_sorted_lines.push_back(base_line);                // base_line is always at the end 
    // debug
    //     if(clockwise_sorted_lines.size()>2)
    //         end = 0;
    
}
                                       
void CrossPtTuple::CalCrossRatios(const int line_order, 
                                  std::vector<double> &cross_ratios) const{
    cross_ratios.clear();
    
    cv::Point2d base_line = tuple_data_[0]->get_center()-tuple_data_[1]->get_center();
    double base_line_norm = cv::norm(base_line);
    base_line.x /= base_line_norm;
    base_line.y /=base_line_norm;
    
    std::vector< cv::Point2d > clockwise_sorted_lines_tmp;
    std::vector< cv::Point2d > counterclockwise_sorted_lines_tmp;
    // calculate cross ratio for first point
    if(tuple_data_[0]->get_tangents_num()>=3){
        SortTangentsFromBaseLine(tuple_data_[0], base_line, clockwise_sorted_lines_tmp);
        if(line_order==0 || line_order==1){
            CalAllCrossRatios(clockwise_sorted_lines_tmp, cross_ratios);
        }
        else{
            for(int i=clockwise_sorted_lines_tmp.size()-1;i>=0; --i){
                counterclockwise_sorted_lines_tmp.push_back(clockwise_sorted_lines_tmp[i]);
            }
            CalAllCrossRatios(counterclockwise_sorted_lines_tmp,cross_ratios);
        }
    }
    
    // calculate cross ratio for second point
    if(tuple_data_[1]->get_tangents_num()>=3){
        std::vector<double> cross_ratios2;
        SortTangentsFromBaseLine(tuple_data_[1], base_line, clockwise_sorted_lines_tmp);
        if(line_order==0 || line_order==2){
            CalAllCrossRatios(clockwise_sorted_lines_tmp, cross_ratios2);
        }
        else{
            counterclockwise_sorted_lines_tmp.clear();
            for(int i=clockwise_sorted_lines_tmp.size()-1;i>=0; --i){
                counterclockwise_sorted_lines_tmp.push_back(clockwise_sorted_lines_tmp[i]);
            }
            CalAllCrossRatios(counterclockwise_sorted_lines_tmp, cross_ratios2);
        }
        
        for(int i=0; i<cross_ratios2.size(); ++i){
            cross_ratios.push_back(cross_ratios2[i]);
        }
    }
}

void CrossPtTuple::GetSortedTangents(int line_order,
                                     std::vector< cv::Point3d >& sorted_tangents) const{
    sorted_tangents.clear();
    std::vector<cv::Point2d> tangent_pt1_clockwise;
    std::vector<cv::Point2d> tangent_pt2_clockwise;
    cv::Point2d pt1_center = tuple_data_[0]->get_center();
    cv::Point2d pt2_center = tuple_data_[1]->get_center();
    
    cv::Point2d base_line = pt1_center - pt2_center;
    double base_line_norm = cv::norm(base_line);
    base_line.x /= base_line_norm;
    base_line.y /=base_line_norm;
    
    SortTangentsFromBaseLine(tuple_data_[0], base_line, tangent_pt1_clockwise);
    SortTangentsFromBaseLine(tuple_data_[1], base_line, tangent_pt2_clockwise);
    
    // debug
    // Print();
    
    // the connection between the two points are not used
    tangent_pt1_clockwise.pop_back();
    tangent_pt2_clockwise.pop_back();
    
    switch(line_order){
        case 0:{
            for(int i=0; i<tangent_pt1_clockwise.size(); ++i)
                sorted_tangents.push_back(EstimateLine(pt1_center, tangent_pt1_clockwise[i]));
            for(int i=0; i<tangent_pt2_clockwise.size(); ++i)
                sorted_tangents.push_back(EstimateLine(pt2_center, tangent_pt2_clockwise[i]));
            break;
        }
        case 1:{
            for(int i=0; i<tangent_pt1_clockwise.size(); ++i)
                sorted_tangents.push_back(EstimateLine(pt1_center, tangent_pt1_clockwise[i]));
            for(int i=tangent_pt2_clockwise.size()-1; i>=0; --i)
                sorted_tangents.push_back(EstimateLine(pt2_center, tangent_pt2_clockwise[i]));
            break;
        }
        case 2:{
            for(int i=tangent_pt1_clockwise.size()-1; i>=0; --i)
                sorted_tangents.push_back(EstimateLine(pt1_center, tangent_pt1_clockwise[i]));
            for(int i=0; i<tangent_pt2_clockwise.size(); ++i)
                sorted_tangents.push_back(EstimateLine(pt2_center, tangent_pt2_clockwise[i]));
            break;
        }
        case 3:{
            for(int i=tangent_pt1_clockwise.size()-1; i>=0; --i)
                sorted_tangents.push_back(EstimateLine(pt1_center, tangent_pt1_clockwise[i]));
            for(int i=tangent_pt2_clockwise.size()-1; i>=0; --i)
                sorted_tangents.push_back(EstimateLine(pt2_center, tangent_pt2_clockwise[i]));
            break;
        }
    }
}

cv::Point3d CrossPtTuple::EstimateLine(const cv::Point2d &point, 
                                        const cv::Point2d &tangent) const{
    cv::Point3d line; 
    line.z = tangent.x * point.y - tangent.y * point.x;
    
    line.x = tangent.y;
    line.y = - tangent.x;
    
    return line;
}

void CrossPtTuple::Print() const{
    std::cout<<"branches:("<<tuple_data_[0]->get_braches_num()<<","<<tuple_data_[1]->get_braches_num()<<") "
    <<"tangents:("<<tuple_data_[0]->get_tangents_num()<<","<<tuple_data_[1]->get_tangents_num()<<")"<<std::endl;
}




CrossPtTupleHashList::CrossPtTupleHashList(){
    
}

void CrossPtTupleHashList::Build(const std::vector< CrossPointPtr >& cross_pts, 
                                 const CrossPtTupleHashList::Param &params, 
                                 bool build_search_tree,
                                 bool allow_symmetric){
    // 0. Allocate memory space
    data_.clear();
    for(int i=0; i<SIZE; ++i){
        CrossPtTupleSetPtr tmp(new  CrossPtTupleSet);
        data_.push_back(tmp);
    }
    std::vector<int> used_pt_indexs;
    
    // 1. build distance tree
    // Get all the center point to build the search tree 
    std::vector<cv::Point2d> pt_centers;
    for(int i=0; i<cross_pts.size(); i++){
        pt_centers.push_back(cross_pts[i]->get_center());
    }
    
    std::unique_ptr<RoadMapTree> search_tree;
    search_tree.reset(new RoadMapTree);
    search_tree->BuildKDTree(pt_centers);
    
    //2. build the hash list
    std::vector<cv::Point2d> current_pt_center;
    std::vector<std::vector<std::pair<size_t,double> > > ret_matches;
    for(int i=0; i<cross_pts.size(); i++){
        if(!allow_symmetric)
            used_pt_indexs.push_back(i);
        int current_pt_id = cross_pts[i]->get_id();
        
        // search pts within a certain distance
        current_pt_center.clear();
        current_pt_center.push_back(cross_pts[i]->get_center());
        ret_matches.clear();
        search_tree->RadiusSearch( current_pt_center,
                                    params.max_tuple_pts_distance * params.max_tuple_pts_distance,
                                    ret_matches );
        
        // build hash list
        for(int j=0; j<ret_matches[0].size(); ++j){
            int index = ret_matches[0][j].first;
            if(allow_symmetric){
                if(cross_pts[index]->get_id()!=current_pt_id ){
                    CrossPtTuple tmp;
                    if(!tmp.Init(cross_pts[i], cross_pts[index])) 
                        continue;
                    if(tmp.CheckValidity(params.min_tuple_pts_distance, params.min_cos_angle_distance))
                        data_[tmp.hash_key()]->push_back(tmp);
                }
            }
            else{                            // when symmetric is not allowed, whether the point has been used must be checked 
                std::vector<int>::iterator it;
                it = std::find(used_pt_indexs.begin(), used_pt_indexs.end(), index);
                if(it==used_pt_indexs.end()){
                    CrossPtTuple tmp;
                    if(!tmp.Init(cross_pts[i], cross_pts[index])) 
                        continue;
                    if(tmp.CheckValidity(params.min_tuple_pts_distance, params.min_cos_angle_distance))
                        data_[tmp.hash_key()]->push_back(tmp);
                }
            }
            
        }
    }
    
    if(build_search_tree){
        BuildAllSearchTrees();
        CalSamplePriority();
    }
    
    
}

void CrossPtTupleHashList::BuildSearchTree(const CrossPtTupleSetPtr &pts, 
                                           SearchTree &tree){
    if(pts->size()<1)
        return;
    
    CrossPtTuple first_pt = (*pts)[0];
    int first_pt_hash_key = first_pt.hash_key();
    int search_tree_type = first_pt_hash_key % 9;

    
    std::vector<double> cross_ratios_tmp;
    switch(search_tree_type){
        case 1:{
            PointCloud1D<double> cross_ratio_1d;
            for(int i=0; i<pts->size(); ++i){
                ((*pts)[i]).CalCrossRatios(0, cross_ratios_tmp);
                cross_ratio_1d.push_data(cross_ratios_tmp.data());
            }
            tree.BuildTree(cross_ratio_1d);
            break;
        }
        case 2:{
            PointCloud5D<double> cross_ratio_5d;
            for(int i=0; i<pts->size(); ++i){
                ((*pts)[i]).CalCrossRatios(0, cross_ratios_tmp);
                cross_ratio_5d.push_data(cross_ratios_tmp.data());
            }
            tree.BuildTree(cross_ratio_5d);
            break;
        }
        case 3:{
            PointCloud1D<double> cross_ratio_1d;
            for(int i=0; i<pts->size(); ++i){
                ((*pts)[i]).CalCrossRatios(0, cross_ratios_tmp);
                cross_ratio_1d.push_data(cross_ratios_tmp.data());
            }
            tree.BuildTree(cross_ratio_1d);
            break;
        }
        case 4:{
            PointCloud2D<double> cross_ratio_2d;
            for(int i=0; i<pts->size(); ++i){
                ((*pts)[i]).CalCrossRatios(0, cross_ratios_tmp);
                cross_ratio_2d.push_data(cross_ratios_tmp.data());
            }
            tree.BuildTree(cross_ratio_2d);
            break;
        }
        case 5:{
            PointCloud6D<double> cross_ratio_6d;
            for(int i=0; i<pts->size(); ++i){
                ((*pts)[i]).CalCrossRatios(0, cross_ratios_tmp);
                cross_ratio_6d.push_data(cross_ratios_tmp.data());
            }
            tree.BuildTree(cross_ratio_6d);
            break;
        }
        case 6:{
            PointCloud5D<double> cross_ratio_5d;
            for(int i=0; i<pts->size(); ++i){
                ((*pts)[i]).CalCrossRatios(0, cross_ratios_tmp);
                cross_ratio_5d.push_data(cross_ratios_tmp.data());
            }
            tree.BuildTree(cross_ratio_5d);
            break;
        }
        case 7:{
            PointCloud6D<double> cross_ratio_6d;
            for(int i=0; i<pts->size(); ++i){
                ((*pts)[i]).CalCrossRatios(0, cross_ratios_tmp);
                cross_ratio_6d.push_data(cross_ratios_tmp.data());
            }
            tree.BuildTree(cross_ratio_6d);
            break;
        }
        case 8:{
            PointCloud10D<double> cross_ratio_10d;
            for(int i=0; i<pts->size(); ++i){
                ((*pts)[i]).CalCrossRatios(0, cross_ratios_tmp);
                cross_ratio_10d.push_data(cross_ratios_tmp.data());
            }
            tree.BuildTree(cross_ratio_10d);
            break;
        }
    }
}

void CrossPtTupleHashList::BuildAllSearchTrees(){
    cross_ratio_search_trees_.resize(SIZE);
    for(int i=0; i<SIZE; ++i){
        BuildSearchTree(data_[i], cross_ratio_search_trees_[i]);
    }
}

const CrossPtTupleSetPtr& CrossPtTupleHashList::Search(const int feature_input[]) const{
    int feature[4];
    feature[0] =feature_input[0] - 3;
    feature[1] =feature_input[1] - 3;
    feature[2] =feature_input[2] - 2;
    feature[3] =feature_input[3] - 2;
    
    int hash_key = 27*feature[0]+9*feature[1]+3*feature[2]+feature[3];
//     std::cout<<"hash_key = "<<hash_key<<std::endl;
    return data_[hash_key];
}


const CrossPtTupleSetPtr& CrossPtTupleHashList::Search(const int hash_key) const{
    return data_[hash_key];
}

int CrossPtTupleHashList::SearchCrossRatioConsistencyPts(const int hash_key, 
                                   std::vector<double> cross_ratios,               
                                   const double cross_ratio_distance_threshold, 
                                   CrossPtTupleSet &result) const{
    result.clear();
    int cross_ratios_type = hash_key % 9; 
    
    std::vector<std::vector<std::pair<size_t,double> > > ret_matches;
     
    switch(cross_ratios_type){
        case 0:{
            result = *Search(hash_key);
            break;
        }
        case 1:{
            PointCloud1D<double> query_pts;
            query_pts.push_data(cross_ratios.data());
            ret_matches.clear();
            cross_ratio_search_trees_[hash_key].RadiusSearch(query_pts,
                                                             cross_ratio_distance_threshold,
                                                             ret_matches);
            for(int i=0; i<ret_matches[0].size(); ++i){
                result.push_back((*(data_[hash_key]))[(ret_matches[0][i].first)]);
            }
            break;
        }
        case 2:{
            PointCloud5D<double> query_pts;
            query_pts.push_data(cross_ratios.data());
            ret_matches.clear();
            cross_ratio_search_trees_[hash_key].RadiusSearch(query_pts,
                                                             cross_ratio_distance_threshold,
                                                             ret_matches);
            for(int i=0; i<ret_matches[0].size(); ++i){
                result.push_back((*(data_[hash_key]))[(ret_matches[0][i].first)]);
            }
            break;
        }
        case 3:{
            PointCloud1D<double> query_pts;
            query_pts.push_data(cross_ratios.data());
            ret_matches.clear();
            cross_ratio_search_trees_[hash_key].RadiusSearch(query_pts,
                                                             cross_ratio_distance_threshold,
                                                             ret_matches);
            for(int i=0; i<ret_matches[0].size(); ++i){
                result.push_back((*(data_[hash_key]))[(ret_matches[0][i].first)]);
            }
            break;
        }
        case 4:{
            PointCloud2D<double> query_pts;
            query_pts.push_data(cross_ratios.data());
            ret_matches.clear();
            cross_ratio_search_trees_[hash_key].RadiusSearch(query_pts,
                                                             cross_ratio_distance_threshold,
                                                             ret_matches);
            for(int i=0; i<ret_matches[0].size(); ++i){
                result.push_back((*(data_[hash_key]))[(ret_matches[0][i].first)]);
            }
            break;
        }
        case 5:{
            PointCloud6D<double> query_pts;
            query_pts.push_data(cross_ratios.data());
            ret_matches.clear();
            cross_ratio_search_trees_[hash_key].RadiusSearch(query_pts,
                                                             cross_ratio_distance_threshold,
                                                             ret_matches);
            for(int i=0; i<ret_matches[0].size(); ++i){
                result.push_back((*(data_[hash_key]))[(ret_matches[0][i].first)]);
            }
            break;
        }
        case 6:{
            PointCloud5D<double> query_pts;
            query_pts.push_data(cross_ratios.data());
            ret_matches.clear();
            cross_ratio_search_trees_[hash_key].RadiusSearch(query_pts,
                                                             cross_ratio_distance_threshold,
                                                             ret_matches);
            for(int i=0; i<ret_matches[0].size(); ++i){
                result.push_back((*(data_[hash_key]))[(ret_matches[0][i].first)]);
            }
            break;
        }
        case 7:{
            PointCloud6D<double> query_pts;
            query_pts.push_data(cross_ratios.data());
            ret_matches.clear();
            cross_ratio_search_trees_[hash_key].RadiusSearch(query_pts,
                                                             cross_ratio_distance_threshold,
                                                             ret_matches);
            for(int i=0; i<ret_matches[0].size(); ++i){
                result.push_back((*(data_[hash_key]))[(ret_matches[0][i].first)]);
            }
            break;
        }
        case 8:{
            PointCloud10D<double> query_pts;
            query_pts.push_data(cross_ratios.data());
            ret_matches.clear();
            cross_ratio_search_trees_[hash_key].RadiusSearch(query_pts,
                                                             cross_ratio_distance_threshold,
                                                             ret_matches);
            for(int i=0; i<ret_matches[0].size(); ++i){
                result.push_back((*(data_[hash_key]))[(ret_matches[0][i].first)]);
            }
            break;
        }
    }
    return result.size();   
}

int CrossPtTupleHashList::SearchCrossRatioConsistencyPts(const CrossPtTuple &query_tuple,             
                                                         const double cross_ratio_relative_error_threshold, 
                                                         std::vector<CrossPtTupleSet> &result) const{
    result.resize(4);   
    
    int hash_key = query_tuple.hash_key();
    std::vector<double> query_pts_cross_ratios;
    int sum_consistency_pt_num = 0;
    int consistency_pt_num;
    for(int j=0; j<4; ++j){
        if(j==1 || j==3)
            continue;
        query_tuple.CalCrossRatios(j,query_pts_cross_ratios);
        double query_pts_cross_ratios_norm_square = 0;
        for(int i=0; i<query_pts_cross_ratios.size(); ++i){
            query_pts_cross_ratios_norm_square += query_pts_cross_ratios[i] * query_pts_cross_ratios[i];
        }
        
        double search_threshold = cross_ratio_relative_error_threshold 
                                    * cross_ratio_relative_error_threshold 
                                    * query_pts_cross_ratios_norm_square;
        
        consistency_pt_num = SearchCrossRatioConsistencyPts(hash_key, 
                                       query_pts_cross_ratios,
                                       search_threshold,
                                       result[j]);
        sum_consistency_pt_num += consistency_pt_num;
        
    } 
    return sum_consistency_pt_num;
}

void CrossPtTupleHashList::CalSamplePriority(){
    std::vector<int> pt_nums;
    std::vector<int> index;
    for(int i=0; i<data_.size(); ++i){
        if((this->data_[i])->size()>0){
            pt_nums.push_back((this->data_[i])->size());
            index.push_back(i);
        }
    }
    
    int start=0, end = pt_nums.size();
    while(start<end){
        int current_end = start;
        int i = start;
        while(++i<end){
            if(pt_nums[i-1]>pt_nums[i]){
                int tmp_num = pt_nums[i], tmp_index = index[i];
                pt_nums[i] = pt_nums[i-1], index[i] = index[i-1];
                pt_nums[i-1] = tmp_num, index[i-1] = tmp_index;
                current_end  = i;
            }
        }
        end = current_end;
    }
    priority_index_ = index;
}

void CrossPtTupleHashList::Print() const{
    for(int i=0; i<SIZE; ++i){
        int tangent_num_1, tangent_num_2, branch_num_1, branch_num_2;
        branch_num_1 = i/27 + 3;
        branch_num_2 = (i%27)/9 + 3;
        tangent_num_1 = (i%9)/3 + 2;
        tangent_num_2 = i%3 + 2;
        
    }
}

int CrossPtTupleHashList::get_tuple_num(const int hash_key) const{
    if(hash_key<0 || hash_key>80)
        return -1;
    
    return data_[hash_key]->size();
}

    
}   // namespace rcll
