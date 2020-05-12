#ifndef RCLL_CROSS_POINT_TREE_H_
#define RCLL_CROSS_POINT_TREE_H_

#include <vector>
#include <iostream>
#include <memory>
#include <nanoflann.hpp>
#include <opencv2/core/core.hpp>

namespace rcll{

template <typename T>
struct PointCloud{
	struct Point {
		T  x,y;
	};

	std::vector<Point>  pts;
    
	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return pts.size(); }

	// Returns the dim'th component of the idx'th point in the class:
	// Since this is inlined and the "dim" argument is typically an immediate value, the
	//  "if/else's" are actually solved at compile time.
	inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
		if (dim == 0) return pts[idx].x;
		else return pts[idx].y;
	}

	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

};

typedef nanoflann::KDTreeSingleIndexAdaptor<
                nanoflann::L2_Simple_Adaptor<double, PointCloud<double> > ,
                PointCloud<double>,
                2 /* dim */ > RoadMapTree_;
typedef std::unique_ptr<RoadMapTree_> RoadMapTreeUniquePtr;
                
class RoadMapTree{
public:
    /* Build the search tree 
     */
    void BuildKDTree(const std::vector<cv::Point2d> &points );
    
    /* Find all points within a certain distance 
     */
    void RadiusSearch(  const std::vector<cv::Point2d> &requry_pts,
                        const double threshold_distance,
                        std::vector<std::vector<std::pair<size_t,double> > > &ret_matches);
      
private:
    // copy data to pts_
    void FillData(const std::vector<cv::Point2d> &points);
    
    RoadMapTreeUniquePtr tree_;
    PointCloud<double> pts_;    
};

    
}   // namespace rcl

#endif
