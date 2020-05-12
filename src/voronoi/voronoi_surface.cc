#include "voronoi_surface.h"

float VoronoiSurfaceGenerator::Generate(const std::vector<cv::Point2f> &input_pts, 
                                        int width, int height, 
                                        cv::Mat &voronoi_surface){
    // 1. Set center points
    jcv_point* points = 0; 
    int count = input_pts.size();

    points = (jcv_point*)malloc( sizeof(jcv_point) * (size_t)count);

    for(int i=0; i<count; ++i){
        points[i].x = input_pts[i].x;
        points[i].y = input_pts[i].y;
    }  
    
    // 2. generator voronoi diagram with plane sweep algrithom
    jcv_diagram diagram; 
    jcv_clipper* clipper = 0;
    jcv_rect rect;
    rect.max.x = width;
    rect.max.y = height;
    rect.min.x = 0;
    rect.min.y = 0;

    memset(&diagram, 0, sizeof(jcv_diagram));
    jcv_diagram_generate(count, (const jcv_point*)points, &rect, clipper, &diagram);

    // 3. generator voronoi surface
    float max_d = 0;
    voronoi_surface = cv::Mat::zeros(height, width, CV_32F);
    const jcv_site* sites = jcv_diagram_get_sites( &diagram );
    for( int i = 0; i < diagram.numsites; ++i )  {
        const jcv_site* site = &sites[i];

        srand((unsigned int)site->index); // for generating colors for the triangles

        const jcv_graphedge* e = site->edges;
        while( e )  {
            jcv_point s = site->p;
            jcv_point p0 = e->pos[0];
            jcv_point p1 = e->pos[1];
            float tmp = FillTriangeWithMinDistance( &s, &p0, &p1, voronoi_surface);
            if(tmp > max_d)
                max_d = tmp;
            e = e->next;
        }
    }
    return max_d;
}

float VoronoiSurfaceGenerator::Generate(const cv::Mat &input_binary_img,
                                    const cv::Vec4f &padding_size,
                                    cv::Mat &voronoi_surface){
    // 1. Generate point set
    std::vector<cv::Point2f> input_pts;
    for(int i=0; i<input_binary_img.rows; ++i){
        for(int j=0; j<input_binary_img.cols; ++j){
            if(input_binary_img.at<uchar>(i,j)==1)
                input_pts.push_back(cv::Point2f(j, i));
        }
    } 

    // 2. Generate voronoi surface
    cv::Mat tmp;
    float max_d = Generate(input_pts, input_binary_img.cols, input_binary_img.rows, tmp);
    cv::Rect roi(padding_size[2], padding_size[0], 
                tmp.cols-padding_size[2]-padding_size[3], 
                tmp.rows-padding_size[0]-padding_size[1]);
    if(roi.height<=0 || roi.width<=0)
        return 0;
    
    tmp(roi).copyTo(voronoi_surface);
    return max_d;    
}



/************** private ***************/
float VoronoiSurfaceGenerator::FillTriangeWithMinDistance(const jcv_point* v0, 
                              const jcv_point* v1, 
                              const jcv_point* v2, 
                              cv::Mat &voronoi_surface)  {
    int area = orient2d(v0, v1, v2);
    if( area == 0 )
        return -1;

    // Compute triangle bounding box
    int minX = min3((int)v0->x, (int)v1->x, (int)v2->x);
    int minY = min3((int)v0->y, (int)v1->y, (int)v2->y);
    int maxX = max3((int)v0->x, (int)v1->x, (int)v2->x);
    int maxY = max3((int)v0->y, (int)v1->y, (int)v2->y);

    // Clip against screen bounds
    minX = max2(minX, 0);
    minY = max2(minY, 0);
    maxX = min2(maxX, voronoi_surface.cols - 1);
    maxY = min2(maxY, voronoi_surface.rows - 1);

    // Rasterize
    float d_max = 0;
    jcv_point p;
    for (p.y = (jcv_real)minY; p.y <= maxY; p.y++) {
        for (p.x = (jcv_real)minX; p.x <= maxX; p.x++) {
            // Determine barycentric coordinates
            int w0 = orient2d(v1, v2, &p);
            int w1 = orient2d(v2, v0, &p);
            int w2 = orient2d(v0, v1, &p);

            // If p is on or inside all edges, render pixel.
            if (w0 >= 0 && w1 >= 0 && w2 >= 0)            {
                float tmp_d = std::sqrt((v0->x-p.x)*(v0->x-p.x) + (v0->y-p.y)*(v0->y-p.y));
                voronoi_surface.at<float>(p.y, p.x) = tmp_d;

                if(d_max<tmp_d)
                    d_max = tmp_d;
            }
        }
    }

    return d_max;
}
