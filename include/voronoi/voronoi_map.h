

#ifndef VORONOI_MAP_H_
#define VORONOI_MAP_H_

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core/core.hpp>


class VoronoiMap{
public:
    /* Load tiles from file under the guidance of the given info file. 
       The corresponding voronoi surface image file must be provide in the specific path.
    */
    int ReadFromFile(const std::string &info_file);

    /* Save the voronoi surface information into file.
     * @ img_file_pre --- 保存voronoi surface图片文件名的前缀，后续命名规则为_row_col.png
    */
    int WriteToFile(const std::string &info_file, const std::string &img_file_pre);

    // write data
    void Init(float scale, int tile_rows, int tile_cols, int tile_size);

    void SetGeoBound(float geo_min_x, float geo_min_y, float geo_max_x, float geo_max_y);

    /* Ps: row/col --- start from 0 
        @ tile_data --- 未进行归一化的距离
    */
    void SetTileData(const cv::Mat &tile_data, int row, int col, float max_distance);

    // access data
    float GetVoronoiDistance(float geo_x, float geo_y);

    inline cv::Vec4f get_geo_bound() { return cv::Vec4f(geo_x_min_, geo_y_min_, geo_x_max_, geo_y_max_); }
    inline float get_gsd() { return scale_; }
    inline cv::Vec3f get_tile_info() { return cv::Vec3f(tile_size_, tile_rows_, tile_cols_); }
    const std::shared_ptr<cv::Mat> &get_tile_data(int r, int c) const { return tile_data_[r*tile_cols_+c]; } 

private:

    /*************************** data *******************************/
    // 保存的为缩放后的图片对应的未最大值归一化的voronoi surface
    std::vector<std::shared_ptr<cv::Mat> > tile_data_;
    std::vector<float> tile_distance_scale_;
    // pixel/mete
    float scale_;   

    int tile_size_;        
    int tile_rows_;
    int tile_cols_;
    int vaild_tile_num_;

    int img_w_;
    int img_h_;

    float geo_x_min_;
    float geo_y_min_;
    float geo_x_max_;
    float geo_y_max_;

    const float FAR_AWAY = 10000.0;  

};




#endif