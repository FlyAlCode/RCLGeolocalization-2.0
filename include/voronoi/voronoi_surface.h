
#ifndef VORONOI_SURFACE_H_
#define VORONOI_SURFACE_H_

#define JC_VORONOI_IMPLEMENTATION
#include "jc_voronoi.h"


#include <vector>
#include <opencv2/core/core.hpp>

class VoronoiSurfaceGenerator{
public:
    /*
        voronoi_surface --- save the min distance of each point to its nearest point with type of CV_32F
        input_binary_img --- 1 for existing point, 0 for noexisting point
        return value --- max distance
    */
    float Generate(const std::vector<cv::Point2f> &input_pts, int width, int height, cv::Mat &voronoi_surface);

    /*
        padding_size --- 由于最近距离的计算收边缘处点的影响，使用padding_size，来截取中间部分
                            格式为[up,down, left, right]
    */
    float Generate(const cv::Mat &input_binary_img, const cv::Vec4f &padding_size, cv::Mat &voronoi_surface);


private:
    // http://fgiesen.wordpress.com/2013/02/08/triangle-rasterization-in-practice/
    inline int orient2d(const jcv_point* a, 
                        const jcv_point* b, 
                        const jcv_point* c)    {
        return ((int)b->x - (int)a->x)*((int)c->y - (int)a->y) - ((int)b->y - (int)a->y)*((int)c->x - (int)a->x);
    }

    inline int min2(int a, int b) {
        return (a < b) ? a : b;
    }

    inline int max2(int a, int b) {
        return (a > b) ? a : b;
    }

    inline int min3(int a, int b, int c) {
        return min2(a, min2(b, c));
    }
    inline int max3(int a, int b, int c) {
        return max2(a, max2(b, c));
    }

    /*
        Compute the min distances to v0 for all points inside or on the triangle.
        voronoi_surface --- It's memory is supposed to be allocated already.
    */
    float FillTriangeWithMinDistance(const jcv_point* v0, 
                              const jcv_point* v1, 
                              const jcv_point* v2, 
                              cv::Mat &voronoi_surface);


};



#endif