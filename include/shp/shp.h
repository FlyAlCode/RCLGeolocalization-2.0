#ifndef SHP_SHP_H_
#define SHP_SHP_H_
#include <vector>
#include <opencv2/core/core.hpp>

class Shp{
public:
    bool Init(const std::string &shp_file_name);

    bool ConvertAreaToImage(const cv::Rect_<double> &area_geo_bound,
                            double scale,
                            cv::Mat &result_img,
                            cv::Rect2d &real_geo_bound,
                            int line_width = 5,
                            const cv::Scalar &color = cv::Scalar(255, 255, 255),
                            bool black_background = true) const;

    bool ShowOnMap(const cv::Mat &show_img,
                   const cv::Mat &T,
                   int padding_size,
                   const cv::Scalar &color,
                   cv::Mat &result_show_img,
                   bool black_background = true) const;

    cv::Mat GetTransformationFromImageToGeo() const;

    inline const cv::Rect2d &get_geo_bound() const { return geo_bound_; }

    cv::Point2d ImageToGeo(const cv::Point &img_pt) const;

    cv::Point GeoToImage(const cv::Point2d &geo_pt) const;

    inline const std::vector<std::vector<cv::Point2d>>& get_all_polylines() const { return geo_polylines_; } 

private:
    

    cv::Point2d GetTransformedPt(const cv::Point2d &src_pt, const cv::Mat &H) const;

    cv::Rect_<double> geo_bound_;           // min_geo_x, min_geo_y, width(x), height(y)
    cv::Rect pixel_bound_;
    std::vector<std::vector<cv::Point2d>> geo_polylines_;
    std::vector<std::vector<cv::Point>> pixel_polylines_;
};

#endif