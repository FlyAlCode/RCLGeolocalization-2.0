#ifndef RCC_SEARCH_TREE_H_
#define RCC_SEARCH_TREE_H_

#include <vector>
#include <iostream>
#include <memory>
#include <nanoflann.hpp>

namespace rcll{

/****************************** Used for cross ratio search *****************************/
// 1D tree
template <typename T>
struct PointCloud1D{
    struct Point {
        T  x;
    };
    
    std::vector<Point>  pts_;

    inline size_t kdtree_get_point_count() const { return pts_.size(); }

    inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0) return pts_[idx].x;
        else return -1;
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
    
    void push_data(const T *data){
        Point tmp;
        tmp.x = *data;
        pts_.push_back(tmp);
    }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud1D<double> >,
                                            PointCloud1D<double>,
                                            1 /* dim */ > SearchTree1D;
typedef std::unique_ptr<SearchTree1D> SearchTree1DUniquePtr;

// 2D tree
template <typename T>
struct PointCloud2D{
    struct Point {
        T  x1, x2;
    };
    
    std::vector<Point>  pts_;
    
    inline size_t kdtree_get_point_count() const { return pts_.size(); }
    
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0) return pts_[idx].x1;
        else return pts_[idx].x2;
    }
    
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
    
    void push_data(const T *data){
        Point tmp;
        tmp.x1 = *data;
        tmp.x2 = *(data+1);
        pts_.push_back(tmp);
    }
    
};
typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud2D<double> >,
                                            PointCloud2D<double>,
                                            2 /* dim */ > SearchTree2D;
typedef std::unique_ptr<SearchTree2D> SearchTree2DUniquePtr;

// 5D tree
template <typename T>
struct PointCloud5D{
    struct Point {
        T  x1, x2, x3, x4, x5;
    };
    
    std::vector<Point>  pts_;
    
    inline size_t kdtree_get_point_count() const { return pts_.size(); }
    
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
        switch(dim){
            case 0: return pts_[idx].x1;
            case 1: return pts_[idx].x2;
            case 2: return pts_[idx].x3;
            case 3: return pts_[idx].x4;
            case 4: return pts_[idx].x5;
            default: return -1;
        }
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
    
    void push_data(const T *data){
        Point tmp;
        tmp.x1 = *data;
        tmp.x2 = *(data+1);
        tmp.x3 = *(data+2);
        tmp.x4 = *(data+3);
        tmp.x5 = *(data+4);
        pts_.push_back(tmp);
    }
    
};

typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud5D<double> >,
                                            PointCloud5D<double>,
                                            5 /* dim */ > SearchTree5D;
typedef std::unique_ptr<SearchTree5D> SearchTree5DUniquePtr;

// 6D tree
template <typename T>
struct PointCloud6D{
    struct Point {
        T  x1, x2, x3, x4, x5, x6;
    };
    
    std::vector<Point>  pts_;
    
    inline size_t kdtree_get_point_count() const { return pts_.size(); }
    
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
        switch(dim){
            case 0: return pts_[idx].x1;
            case 1: return pts_[idx].x2;
            case 2: return pts_[idx].x3;
            case 3: return pts_[idx].x4;
            case 4: return pts_[idx].x5;
            case 5: return pts_[idx].x6;
            default: return -1;
        }
    }
    
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }  
    
    void push_data(const T *data){
        Point tmp;
        tmp.x1 = *data;
        tmp.x2 = *(data+1);
        tmp.x3 = *(data+2);
        tmp.x4 = *(data+3);
        tmp.x5 = *(data+4);
        tmp.x6 = *(data+5);
        pts_.push_back(tmp);
    }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud6D<double> >,
                                            PointCloud6D<double>,
                                            6 /* dim */ > SearchTree6D;
typedef std::unique_ptr<SearchTree6D> SearchTree6DUniquePtr;

// 10D tree
template <typename T>
struct PointCloud10D{
    struct Point {
        T  x1, x2, x3, x4, x5, x6, x7, x8, x9, x10;
    };
    
    std::vector<Point>  pts_;
    
    inline size_t kdtree_get_point_count() const { return pts_.size(); }
    
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
        switch(dim){
            case 0: return pts_[idx].x1;
            case 1: return pts_[idx].x2;
            case 2: return pts_[idx].x3;
            case 3: return pts_[idx].x4;
            case 4: return pts_[idx].x5;
            case 5: return pts_[idx].x6;
            case 6: return pts_[idx].x7;
            case 7: return pts_[idx].x8;
            case 8: return pts_[idx].x9;
            case 9: return pts_[idx].x10;
            default: return -1;
        }
    }
    
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }  
    
    void push_data(const T *data){
        Point tmp;
        tmp.x1 = *data;
        tmp.x2 = *(data+1);
        tmp.x3 = *(data+2);
        tmp.x4 = *(data+3);
        tmp.x5 = *(data+4);
        tmp.x6 = *(data+5);
        tmp.x7 = *(data+6);
        tmp.x8 = *(data+7);
        tmp.x9 = *(data+8);
        tmp.x10 = *(data+9);
        pts_.push_back(tmp);
    }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud10D<double> >,
                                            PointCloud10D<double>,
                                            10 /* dim */ > SearchTree10D;
typedef std::unique_ptr<SearchTree10D> SearchTree10DUniquePtr;



// search tree 
struct SearchTree{
    SearchTree1DUniquePtr search_tree_1d_;
    SearchTree2DUniquePtr search_tree_2d_;
    SearchTree5DUniquePtr search_tree_5d_;
    SearchTree6DUniquePtr search_tree_6d_;
    SearchTree10DUniquePtr search_tree_10d_;
    
    PointCloud1D<double> pts_1d_;
    PointCloud2D<double> pts_2d_;
    PointCloud5D<double> pts_5d_;
    PointCloud6D<double> pts_6d_;
    PointCloud10D<double> pts_10d_;
    
    void BuildTree(const PointCloud1D<double> &points );
    void BuildTree(const PointCloud2D<double> &points );
    void BuildTree(const PointCloud5D<double> &points );
    void BuildTree(const PointCloud6D<double> &points );
    void BuildTree(const PointCloud10D<double> &points );
    
    void RadiusSearch(  const PointCloud1D<double>  &requry_pts,
                        const double threshold_distance,
                        std::vector<std::vector<std::pair<size_t,double> > >  &ret_matches) const;
    void RadiusSearch(  const PointCloud2D<double>  &requry_pts,
                        const double threshold_distance,
                        std::vector<std::vector<std::pair<size_t,double> > > &ret_matches) const;
    void RadiusSearch(  const PointCloud5D<double>  &requry_pts,
                        const double threshold_distance,
                        std::vector<std::vector<std::pair<size_t,double> > > &ret_matches) const;
    void RadiusSearch(  const PointCloud6D<double>  &requry_pts,
                        const double threshold_distance,
                        std::vector<std::vector<std::pair<size_t,double> > > &ret_matches) const;
    void RadiusSearch(  const PointCloud10D<double>  &requry_pts,
                        const double threshold_distance,
                        std::vector<std::vector<std::pair<size_t,double> > > &ret_matches) const;
    
};
    
}   // namespace rcll


#endif
