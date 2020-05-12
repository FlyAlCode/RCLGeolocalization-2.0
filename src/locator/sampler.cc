#include "sampler.h"
// just for debug
#include <iostream>
#include <glog/logging.h> 

namespace rcll{

/*****************************************
 *     RequryPtSampler implentation      *
 *****************************************/    
void RequryPtSampler::Initilize(){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    util_generator_.seed(seed);
}

void RequryPtSampler::Initilize(const std::vector< CrossPointPtr >& data, 
               const std::vector<int> &priority_index,
               const double min_diatance,
               const double max_distance,
               const double min_cos_angle_distance,
               const double max_tangent_error){
    Initilize();
    good_cross_pts_.clear();
    current_sample_index_ = 0;
    current_sample_type_ = 0;
    priority_index_ = priority_index;

    // find 'good' cross points (points with tangent estimation error less than max_tangent_error)
    for(int i=0; i<data.size(); ++i){
        if(data[i]->get_tangent_error()<max_tangent_error)
            good_cross_pts_.push_back(data[i]);
    }
   
    std::cout<<"    good point ratio = "<<good_cross_pts_.size()<<"/"<<data.size()<<std::endl;
       
    cross_pt_tuple_hash_list_.reset(new CrossPtTupleHashList);
    CrossPtTupleHashList::Param params;
    params.max_tuple_pts_distance = max_distance;
    params.min_tuple_pts_distance = min_diatance;
    params.min_cos_angle_distance = min_cos_angle_distance;
    cross_pt_tuple_hash_list_->Build(good_cross_pts_, params, false, false);
    // cross_pt_tuple_hash_list_->Print();
}

int RequryPtSampler::Sample(CrossPtTuple &sub_set){    
    if(current_sample_type_>=priority_index_.size())
        return -1;                                              // all datas have been sampled
    
    if(current_sample_index_==0){                               // start a new type points sampling
        int current_hash_key = priority_index_[current_sample_type_];
        current_tuple_set_ = cross_pt_tuple_hash_list_->Search(current_hash_key);
    }
    
    
    if(current_tuple_set_->empty()){                     // if current type tuple set is empty, jump to next type tuple set
        ++current_sample_type_;
        current_sample_index_ = 0;
        return 0;
    }
    
    sub_set = (*current_tuple_set_)[current_sample_index_];
    ++current_sample_index_;
    
    if(current_sample_index_>=current_tuple_set_->size()){
        ++current_sample_type_;
        current_sample_index_ = 0;
    }
    return 1;
}



/*****************************************************************************
 *                        MapPtSampler implentation                          *
 *****************************************************************************/ 
void MapPtSampler::Initialize(const std::vector< CrossPointPtr >& data, 
                              const double cos_angle_threshold,
                              const double max_distance){
    cross_pt_tuple_hash_list_.reset(new CrossPtTupleHashList);
    CrossPtTupleHashList::Param params;
    params.max_tuple_pts_distance = max_distance;
    params.min_tuple_pts_distance = 0;
    params.min_cos_angle_distance = cos_angle_threshold;
    cross_pt_tuple_hash_list_->Build(data, params);
    // cross_pt_tuple_hash_list_->Print();
    
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    util_generator_.seed(seed);
}

int MapPtSampler::Sample(const CrossPtTuple &reference_sampler,
                         const double cross_ratio_relative_error_threshold,
                         CrossPtTuple &sub_set,
                         int &match_type,
                         bool is_reference_changed){
    
     
    if(is_reference_changed){
        current_match_pt_num_ =  cross_pt_tuple_hash_list_->SearchCrossRatioConsistencyPts(reference_sampler,   
                                                                      cross_ratio_relative_error_threshold, 
                                                                      current_match_pt_set_ );
        current_match_pt_each_type_num_[0] = current_match_pt_set_[0].size();
        current_match_pt_each_type_num_[1] = current_match_pt_each_type_num_[0]+current_match_pt_set_[1].size();
        current_match_pt_each_type_num_[2] = current_match_pt_each_type_num_[1]+current_match_pt_set_[2].size();
        current_match_pt_each_type_num_[3] = current_match_pt_each_type_num_[2]+current_match_pt_set_[3].size();
        current_index = 0;
    }
        
    if(current_match_pt_num_==0)
        return 0;
     
    // int index = RandInt(0, current_match_pt_num_-1);
    int index = current_index;
    
    if(index<current_match_pt_each_type_num_[0]){
        sub_set = current_match_pt_set_[0][index];
        match_type = 0;
    }
    else if(index<current_match_pt_each_type_num_[1]){
        sub_set = current_match_pt_set_[1][index - current_match_pt_each_type_num_[0]];
        match_type = 1;
    }
    else if(index<current_match_pt_each_type_num_[2]){
        sub_set = current_match_pt_set_[2][index - current_match_pt_each_type_num_[1]];
        match_type = 2;
    }
    else{
        sub_set = current_match_pt_set_[3][index - current_match_pt_each_type_num_[2]];
        match_type = 3;
    }
    
    ++current_index;
    return current_match_pt_num_;
}


int MapPtSampler::SampleAllCandinates(const CrossPtTuple &reference_sampler,
                            const double cross_ratio_relative_error_threshold, 
                            std::vector<CrossPtTupleSet> &candinates){
    int candinates_num =  cross_pt_tuple_hash_list_->SearchCrossRatioConsistencyPts(reference_sampler,   
                                                                      cross_ratio_relative_error_threshold, 
                                                                      candinates );
    return candinates_num;
}

std::vector<int> MapPtSampler::GetSamplePriorityIndex(){
    return cross_pt_tuple_hash_list_->get_priority_index();
}

    
}   //namespace rcll
