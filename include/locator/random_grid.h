#ifndef _RCLL_RANDOM_GRID_H_
#define _RCLL_RANDOM_GRID_H_

#include <vector>
#include <memory>
#include <assert.h>
#include <iostream>

namespace rcll{

class RandomGrid2D{
public:
    struct Model {
        double H_mat[9];
        double match_pts[8];    
        int parent_id_;        

        friend std::ostream & operator << (std::ostream &os, const Model *model){
            for(int i=0; i<9; ++i)
                os<<model->H_mat[i]<<"  ";
        }
    };

    /* 1. Set the parameters;
     * 2. Compute grid_r_ and grid_c_;
     * 3. Allocate memory for grid_data_;
     * 4. Generate random shifts and save them in rand_dx_ and rand_dy_.
    */
    void Init(int grid_size, float grid_min_x, float grid_max_x, float grid_min_y, float grid_max_y);

    /* 1. Clear model;
     * 2. Set all grid_data_ to nullptr;
     */
    void Reset();

    /* 1. Compute the transformed center under the new_model
     * 2. Shift the transformed center with the random shifts
     * 3. For each shifted center, quantize it into a grid
     * 4. If there exist already a model, check whether the two models are consistent
     * 5. insert the index of the model into the grid
     */
    int InsertNewModel(const std::shared_ptr<Model> & new_model, 
                        float center_x, 
                        float center_y,
                        std::vector<int> &ids,
                        double H_error_threshold = 50);

    inline const std::shared_ptr<Model> & get_model(int index) const {
        assert(index<all_models_.size() && index>=0);
        return all_models_[index];
    }


    // clear memory
    ~RandomGrid2D();



private:
    double CalModelDistance(const std::shared_ptr<Model> &m1, 
                            const std::shared_ptr<Model> &m2,
                            int img_w, int img_h);

    std::vector<std::shared_ptr<Model> > all_models_;
    
    // grid params: bound, size
    int grid_size_;
    float grid_min_x_, grid_max_x_, grid_min_y_, grid_max_y_;
    int grid_r_, grid_c_;

    int shift_num_;    
    std::vector<float> rand_dx_;
    std::vector<float> rand_dy_;

    int ** grid_data_;
    const int MAX_PT_NUM_EACH_GRID = 50;

    const double lamda_threshold = -0.7;   

};


}   // namespace rcll

#endif