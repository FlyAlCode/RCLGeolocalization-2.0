#ifndef RCLL_SAMPLER_H_
#define RCLL_SAMPLER_H_

#include <vector>
#include <chrono>
#include <random>
#include <memory>
#include <functional>
#include <algorithm>
#include <ctime>
#include <opencv2/core/core.hpp>
#include "cross_point.h"
#include "cross_point_tree.h"
#include "cross_pt_tuple.h"


namespace rcll{

/* @A sampler used for sample points from requry point set.
 */
class RequryPtSampler {
public:

    /* return value:
     *   -1 -- all datas have been sampled
     *    0 -- current type has no data
     *    1 -- sample success
     */
    int Sample(CrossPtTuple &sub_set);                                // if it is set true, then rebuild the hash list
    
    /* @Must be called before sampling
     * @Initialize the random number generator
     */
    void Initilize();
    void Initilize(const std::vector< CrossPointPtr >& data, 
                   const std::vector<int> &priority_index,
                   const double min_diatance,
                   const double max_distance,
                   const double min_cos_angle_distance, 
                   const double max_tangent_error);
    
private:
    inline int RandInt(int lower, int upper) {
        std::uniform_int_distribution<int> distribution(lower, upper);
        return distribution(util_generator_);
    }

    std::default_random_engine util_generator_;
    CrossPtTupleSet cross_pt_tuple_set_; 
    int current_sample_index_;
    int current_sample_type_;
    std::vector<int> priority_index_;
    CrossPtTupleSetPtr current_tuple_set_;
    std::unique_ptr<CrossPtTupleHashList> cross_pt_tuple_hash_list_;

    std::vector<CrossPointPtr> good_cross_pts_;
};


class MapPtSampler{
public:
    /* @Must be called before sampling
     * @1. Initialize the random number generator
     * @2. 
     * @3. Create a hash list for search
     * @input:
     *   max_distance --- distance between any two samplers should be less than max_distance
     *   cos_angle_threshold --- to ensure four different lines got form the samplers
     */
    void Initialize(const std::vector< CrossPointPtr >& data, 
                    const double cos_angle_threshold,
                    const double max_distance);
    
    /* @Search all possible matchers for given reference_sampler,
     * @and save them in current_match_pt_set_, the total number 
     * @of them are saved in current_match_pt_num_
     * @input:
     *   reference_sampler 
     *   cross_ratio_relative_error_threshold -- the relative error of cross ratio for a tuple to be considered as a matcher
     * @output:
     *   sub_set --- the sampled data
     *   match_type -- 0(++) 1(+-) 2(-+) 3(--) where + for clockwise, and - for counterclockwise
     *   is_reference_changed -- if set to true, update current_match_pt_set_
     */
    int Sample(const CrossPtTuple &reference_sampler,
               const double cross_ratio_relative_error_threshold,
               CrossPtTuple &sub_set,
               int &match_type,
               bool is_reference_changed = false);
    
    /* @Get all matching candinates that pass the cross ratio test, and return the total number
       @of candinates.
       @output: 
            candinates --- Save matching candiantes under different cross ratios
    */
    int SampleAllCandinates(const CrossPtTuple &reference_sampler,
                            const double cross_ratio_relative_error_threshold, 
                            std::vector<CrossPtTupleSet> &candinates);
    
    std::vector<int> GetSamplePriorityIndex();
    inline int GetTupleNum(const int hash_key) const{ 
        return cross_pt_tuple_hash_list_->get_tuple_num(hash_key);}
    
private:
    inline int RandInt(int lower, int upper) {
        std::uniform_int_distribution<int> distribution(lower, upper);
        return distribution(util_generator_);
    }
     
    std::default_random_engine util_generator_;
    std::vector<CrossPtTupleSet> current_match_pt_set_;
    int current_match_pt_num_;
    int current_match_pt_each_type_num_[4];
    int current_index;
//     std::unique_ptr<RoadMapTree> search_tree_;
//     std::vector<CrossPointPtr> cross_points_;
    std::unique_ptr<CrossPtTupleHashList> cross_pt_tuple_hash_list_;
};
    
}   // namespace rcll

#endif
