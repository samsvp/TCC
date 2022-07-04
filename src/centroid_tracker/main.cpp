#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>

#include "include/tracker.hpp"

cv::Mat translateImg(cv::Mat &img, int offsetx, int offsety){
    cv::Mat trans_mat = (cv::Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
    warpAffine(img,img,trans_mat,img.size());
    return img;
}

int main(int argc, char *argv[])
{

    Tracker tracker;
    // cv::Point2f p1(1,1);
    // cv::Point2f p2(4,1);
    // cv::Point2f p3(5,5);
    // cv::Point2f p4(2,2);

    // std::vector<cv::Point2f> old_points = {p1, p2, p3};
    // std::vector<cv::Point2f> new_points = {p2, p1, p4, p3};

    // for (int i=0; i<new_points.size(); i++)
    // {
    //     auto p = new_points[i];
    //     int index = tracker.find_min(i, new_points, old_points);
    //     std::cout << index << "\n";
    // }

    // Load the image
    cv::CommandLineParser parser( argc, argv, "{@input | cards.png | input image}" );
    cv::Mat src = cv::imread( parser.get<cv::String>( "@input" ) );
    if(src.empty())
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
        std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
        return -1;
    }
    

    while (1)
    {
        tracker.update(src);
        auto contours = tracker.get_countours();
        auto mc = tracker.get_centers();
        auto hierarchy = tracker.get_hierarchy();

        // draw contours
        cv::Mat drawing(src.size(), CV_8UC3, cv::Scalar(255,255,255));
        for(int i = 0; i<contours.size(); i++)
        {
            auto color = tracker.get_color(i);
            cv::drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
            cv::circle( drawing, mc[i], 4, color, -1, 8, 0 );
        }
        
        // show the resultant image
        cv::namedWindow("Contours", cv::WINDOW_NORMAL);
        cv::imshow("Contours", drawing);
        cv::waitKey(0);

        src = translateImg(src, 0, -10);
    }
    

    return 0;
}