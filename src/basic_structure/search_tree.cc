#include "search_tree.h"

namespace rcll{
/***********************************************************************
 *                            Search Tree                              *
 ***********************************************************************/
void SearchTree::BuildTree(const PointCloud1D<double> &points ){
    pts_1d_ = points;
    search_tree_1d_.reset(new SearchTree1D(1, pts_1d_, nanoflann::KDTreeSingleIndexAdaptorParams(20)));
    search_tree_1d_->buildIndex();
}

void SearchTree::BuildTree(const PointCloud2D<double> &points ){
    pts_2d_ = points;
    search_tree_2d_.reset(new SearchTree2D(2, pts_2d_, nanoflann::KDTreeSingleIndexAdaptorParams(20)));
    search_tree_2d_->buildIndex();
}
void SearchTree::BuildTree(const PointCloud5D<double> &points ){
    pts_5d_ = points;
    search_tree_5d_.reset(new SearchTree5D(5, pts_5d_, nanoflann::KDTreeSingleIndexAdaptorParams(20)));
    search_tree_5d_->buildIndex();
}
void SearchTree::BuildTree(const PointCloud6D<double> &points ){
    pts_6d_ = points;
    search_tree_6d_.reset(new SearchTree6D(6, pts_6d_, nanoflann::KDTreeSingleIndexAdaptorParams(20)));
    search_tree_6d_->buildIndex();
}
void SearchTree::BuildTree(const PointCloud10D<double> &points ){
    pts_10d_ = points;
    search_tree_10d_.reset(new SearchTree10D(10, pts_10d_, nanoflann::KDTreeSingleIndexAdaptorParams(20)));
    search_tree_10d_->buildIndex();
}

void SearchTree::RadiusSearch(  const PointCloud1D<double>  &requry_pts,
                    const double threshold_distance,
                    std::vector<std::vector<std::pair<size_t,double> > > &ret_matches)const{
    if(search_tree_1d_==nullptr)
        return;
    
    ret_matches.clear();
    ret_matches.resize(requry_pts.kdtree_get_point_count());
    nanoflann::SearchParams params;
    double query_pt;
    for(int i=0; i<requry_pts.kdtree_get_point_count(); ++i){
        query_pt = requry_pts.kdtree_get_pt(i, 0);
        
        const size_t nMatches = search_tree_1d_->radiusSearch(&query_pt, threshold_distance, ret_matches[i], params);
    }
}

void SearchTree::RadiusSearch(  const PointCloud2D<double>  &requry_pts,
                    const double threshold_distance,
                    std::vector<std::vector<std::pair<size_t,double> > > &ret_matches)const{
    if(search_tree_2d_==nullptr)
        return;
    
    ret_matches.clear();
    ret_matches.resize(requry_pts.kdtree_get_point_count());
    nanoflann::SearchParams params;
    double query_pt[2];
    for(int i=0; i<requry_pts.kdtree_get_point_count(); ++i){
        for(int j=0; j<2; ++j)
            query_pt[j] = requry_pts.kdtree_get_pt(i, j);
        
        const size_t nMatches = search_tree_2d_->radiusSearch(query_pt, threshold_distance, ret_matches[i], params);
    }                    
}

void SearchTree::RadiusSearch(  const PointCloud5D<double>  &requry_pts,
                    const double threshold_distance,
                    std::vector<std::vector<std::pair<size_t,double> > > &ret_matches)const{
    if(search_tree_5d_==nullptr)
        return;
    
    ret_matches.clear();
    ret_matches.resize(requry_pts.kdtree_get_point_count());
    nanoflann::SearchParams params;
    double query_pt[5];
    for(int i=0; i<requry_pts.kdtree_get_point_count(); ++i){
        for(int j=0; j<5; ++j)
            query_pt[j] = requry_pts.kdtree_get_pt(i, j);
        
        const size_t nMatches = search_tree_5d_->radiusSearch(query_pt, threshold_distance, ret_matches[i], params);
    }                    
}

void SearchTree::RadiusSearch(  const PointCloud6D<double>  &requry_pts,
                    const double threshold_distance,
                    std::vector<std::vector<std::pair<size_t,double> > > &ret_matches)const{
    if(search_tree_6d_==nullptr)
        return;
    
    ret_matches.clear();
    ret_matches.resize(requry_pts.kdtree_get_point_count());
    nanoflann::SearchParams params;
    double query_pt[6];
    for(int i=0; i<requry_pts.kdtree_get_point_count(); ++i){
        for(int j=0; j<6; ++j)
            query_pt[j] = requry_pts.kdtree_get_pt(i, j);
        
        const size_t nMatches = search_tree_6d_->radiusSearch(query_pt, threshold_distance, ret_matches[i], params);
    }                    
}

void SearchTree::RadiusSearch(  const PointCloud10D<double>  &requry_pts,
                    const double threshold_distance,
                    std::vector<std::vector<std::pair<size_t,double> > > &ret_matches)const{
    if(search_tree_10d_==nullptr)
        return;
    
    ret_matches.clear();
    ret_matches.resize(requry_pts.kdtree_get_point_count());
    nanoflann::SearchParams params;
    double query_pt[10];
    for(int i=0; i<requry_pts.kdtree_get_point_count(); ++i){
        for(int j=0; j<10; ++j)
            query_pt[j] = requry_pts.kdtree_get_pt(i, j);
        
        const size_t nMatches = search_tree_10d_->radiusSearch(query_pt, threshold_distance, ret_matches[i], params);
    }                    
}

}
