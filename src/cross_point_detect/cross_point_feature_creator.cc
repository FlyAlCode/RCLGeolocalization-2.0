#include "cross_point_feature_creator.h"
#include <opencv2/imgproc/imgproc.hpp>

// debug
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

namespace rcll{
    
bool CreateCrossPointFeatures(const cv::Mat &input_img,
                             const CrossPointFeatureParam &param,
                             std::vector<CrossPointPtr> &cross_points,
                             cv::Mat &thined_img,
                              bool img_thined ){
    cross_points.clear();
    // 1.thin image 
    cv::Mat tmp_img;
    if(input_img.channels()==3)
        cv::cvtColor(input_img, tmp_img, CV_RGB2GRAY);
    else
        input_img.copyTo(tmp_img);

    
    cv::threshold(tmp_img, tmp_img, param.threshold, 1, cv::THRESH_BINARY);
 
    
    if(!img_thined){
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Point(5,5));
        cv::erode(tmp_img, tmp_img,kernel );
        cv::dilate(tmp_img, tmp_img, kernel);
        ThinImage(tmp_img, tmp_img);
    }
    
    // 2.detector cross point
    cv::Mat neighbor_img;
    std::vector<cv::Rect> cross_point_areas;
    CrossPointDetector my_detector(param.min_cross_point_distance);
    my_detector.Run(tmp_img, neighbor_img, cross_point_areas);
    
    // 3.create all cross point features 
    CrossLineExtractor my_line_extractor(param.branch_length);
    my_line_extractor.SetNeighborImage(neighbor_img);
    std::vector<std::vector<cv::Point> > branches;
    for(int i=0; i<cross_point_areas.size(); i++){
        my_line_extractor.FindAllBranches(cross_point_areas[i], branches);
        if(branches.size()>2){
            CrossPointPtr tmp_cross_point(new CrossPoint);
            if(tmp_cross_point->FillData(cross_point_areas[i], branches, param.merge_angle_threshold))
                cross_points.push_back(tmp_cross_point);
        }    
    }
    
    tmp_img.copyTo(thined_img);
    return true;
}

void FillImageHole(const cv::Mat &input_img, cv::Mat &output_img){
    input_img.copyTo(output_img);
    
    for(int i=1; i<input_img.rows-1;i++){
        for(int j=1; j<input_img.cols-1; j++){
            if(input_img.at<char>(i,j)==0){
                uchar up = input_img.at<uchar>(i-1,j);
                uchar down = input_img.at<uchar>(i+1,j);  
                uchar left = input_img.at<uchar>(i,j-1);
                uchar right = input_img.at<uchar>(i,j+1);
                if(up==1 && down==1 && right==1 && left==1)
                    output_img.at<uchar>(i,j) = 1;
                
            }
                      
        }
    }
}

void ThinImage(const cv::Mat & src,
               cv::Mat &thined_img,
               const int maxIterations ){
    CV_Assert(src.type() == CV_8UC1);
    cv::Mat dst;
    int width = src.cols;
    int height = src.rows;
    src.copyTo(dst);
    int count = 0;  
    while (true)
    {
        count++;
        if (maxIterations != -1 && count > maxIterations)  
            break;
        std::vector<uchar *> mFlag; 
        for (int i = 0; i < height; ++i)
        {
            uchar * p = dst.ptr<uchar>(i);
            for (int j = 0; j < width; ++j)
            { 
                uchar p1 = p[j];
                if (p1 != 1) continue;
                uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
                uchar p8 = (j == 0) ? 0 : *(p + j - 1);
                uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
                uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
                uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
                uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
                uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
                uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
                if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6) {
                    int ap = 0;
                    if (p2 == 0 && p3 == 1) ++ap;
                    if (p3 == 0 && p4 == 1) ++ap;
                    if (p4 == 0 && p5 == 1) ++ap;
                    if (p5 == 0 && p6 == 1) ++ap;
                    if (p6 == 0 && p7 == 1) ++ap;
                    if (p7 == 0 && p8 == 1) ++ap;
                    if (p8 == 0 && p9 == 1) ++ap;
                    if (p9 == 0 && p2 == 1) ++ap;

                    if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0){
                        mFlag.push_back(p + j);
                    }
                }
            }
        }

        for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i) {
            **i = 0;
        }

        if (mFlag.empty()) {
            break;
        }
        else  {
            mFlag.clear();//œ«mFlagÇå¿Õ  
        }
 
        for (int i = 0; i < height; ++i) {
            uchar * p = dst.ptr<uchar>(i);
            for (int j = 0; j < width; ++j) { 
                uchar p1 = p[j];
                if (p1 != 1) continue;
                uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
                uchar p8 = (j == 0) ? 0 : *(p + j - 1);
                uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
                uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
                uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
                uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
                uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
                uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);

                if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
                {
                    int ap = 0;
                    if (p2 == 0 && p3 == 1) ++ap;
                    if (p3 == 0 && p4 == 1) ++ap;
                    if (p4 == 0 && p5 == 1) ++ap;
                    if (p5 == 0 && p6 == 1) ++ap;
                    if (p6 == 0 && p7 == 1) ++ap;
                    if (p7 == 0 && p8 == 1) ++ap;
                    if (p8 == 0 && p9 == 1) ++ap;
                    if (p9 == 0 && p2 == 1) ++ap;

                    if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0) {
                        mFlag.push_back(p + j);
                    }
                }
            }
        }

        for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
        {
            **i = 0;
        }

        if (mFlag.empty())  {
            break;
        }
        else  {
            mFlag.clear();
        }
    }
    dst.copyTo(thined_img);
}

void ShowCrossDetectorResult(const cv::Mat &img,
                             std::vector<CrossPointPtr> &cross_points){
    cv::Mat draw_img;
    img.copyTo(draw_img);
    // save
    draw_img = 255*draw_img;
    
    for(int i=0; i<cross_points.size(); i++){
        cross_points[i]->Draw(draw_img, cv::Scalar(255,255,0));
    }
    
    cv::namedWindow("result", cv::WINDOW_NORMAL);
    cv::imshow("result", draw_img);
    cv::waitKey();
}



}   // namespace rcll
