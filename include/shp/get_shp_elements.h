#ifndef GET_SHP_ELEMENTS_H_
#define GET_SHP_ELEMENTS_H_

#include <vector>
#include <string>
#include <../../opt/ros/kinetic/include/opencv-3.3.1-dev/opencv2/core/core.hpp>
#include <../../opt/ros/kinetic/include/opencv-3.3.1-dev/opencv2/core.hpp>
#include <../../opt/ros/kinetic/include/opencv-3.3.1-dev/opencv2/core/types.hpp>

typedef std::vector<cv::Point2d> pts;

/* Read all the records in a .shp file and save the bound and records.
 * We deal with different types of records as follow:
 * --shape type--                           --action--
 *   null shape                                 not save
 * point(Z,M)/polyLine(Z,M)/multiPoint(Z,M)     save different parts in individual pts
 * polygon(Z,M)                                 save every part in individal pts with the copy for first point at the end
 */
int GetShpElements(const std::string &shp_file_name, 
                   cv::Rect2d &bound, 
                   std::vector<pts> &elements);

void GetShpInfo(const std::string &shp_file_name, double &area_size, double &total_line_length);

#endif
