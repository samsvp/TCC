#include <cmath>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


class Tracker
{
public:
    Tracker();
    ~Tracker();

    std::vector<cv::Point2f> calculate_centers(cv::Mat src, 
        std::vector<cv::Vec4i>& hierarchy,
        std::vector<std::vector<cv::Point>>& contours,
        bool show_bw=0) const;
    std::vector<float> calc_dists(
        cv::Point2f p, const std::vector<cv::Point2f>& points) const;
    int find_min (int i, 
        std::vector<cv::Point2f> new_points,
        std::vector<cv::Point2f> old_points) const;

    void update(cv::Mat img);

    cv::Scalar get_color(int i);
    
    const std::vector<cv::Point2f>& get_centers() const { return centers; }
    const std::vector<cv::Vec4i>& get_hierarchy() const { return hierarchy; }
    const std::vector<std::vector<cv::Point>>& get_countours() const { return contours; }

private:
    cv::Scalar new_color();
    std::vector<cv::Point2f> calculate_centers(cv::Mat src, bool show_bw=0);

    std::vector<int> ids;
    std::vector<cv::Scalar> colors;
    std::vector<cv::Point2f> centers;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point>> contours;    
};


Tracker::Tracker() { }
Tracker::~Tracker() { }


cv::Scalar Tracker::new_color()
{
    int b = cv::theRNG().uniform(0, 256);
    int g = cv::theRNG().uniform(0, 256);
    int r = cv::theRNG().uniform(0, 256);
    cv::Scalar color(b, g, r);
    return color;
}


cv::Scalar Tracker::get_color(int i)
{
    return this->colors[i];
}


std::vector<float> Tracker::calc_dists(
    cv::Point2f p, const std::vector<cv::Point2f>& points) const
{
    std::vector<float> distances(points.size());
    
    std::transform(points.begin(), points.end(), distances.begin(),
        [&p](cv::Point2f _p){ return abs(p.x - _p.x) + abs(p.y- _p.y); }
    );

    return distances;
}


std::vector<cv::Point2f> Tracker::calculate_centers(cv::Mat src, 
    std::vector<cv::Vec4i>& hierarchy,
    std::vector<std::vector<cv::Point>>& contours,
    bool show_bw) const
{
   
    // Create binary image from source image
    cv::Mat bw;
    cv::cvtColor(src, bw, cv::COLOR_BGR2GRAY);
    cv::threshold(bw, bw, 40, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    
    if (show_bw) cv::imshow("Binary Image", bw);
    
    
    // find contours
    cv::findContours(bw, contours, hierarchy, 
        cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0) 
    );

    // remove small regions of the image
    contours.erase(
        std::remove_if(contours.begin(), contours.end(),
            [](std::vector<cv::Point> p){ return cv::contourArea(p) < 100; }),
        contours.end()
    );

    // get the moments
    std::vector<cv::Moments> mu(contours.size());
    for(int i = 0; i<contours.size(); i++)
    { 
        mu[i] = cv::moments( contours[i], false ); 
    }

    // get the centroid of figures.
    std::vector<cv::Point2f> centers(contours.size());
    for(int i = 0; i<contours.size(); i++)
    {
        centers[i] = cv::Point2f(
            mu[i].m10/(mu[i].m00+0.0001), 
            mu[i].m01/(mu[i].m00+0.0001)
        ); 
    }

    return centers;
}


std::vector<cv::Point2f> Tracker::calculate_centers(cv::Mat src, bool show_bw)
{
    return calculate_centers(src, this->hierarchy, this->contours, show_bw);
}


int Tracker::find_min(int i, 
        std::vector<cv::Point2f> new_points,
        std::vector<cv::Point2f> old_points) const
{
    if (new_points.empty() || old_points.empty()) return -1;
    
    cv::Point2f p = new_points[i];

    // distance from this point to the old points
    std::vector<float> pdists = this->calc_dists(p, old_points);
    int pargmin = std::min_element(pdists.begin(), pdists.end()) - pdists.begin();
    // distance from target point to other points
    cv::Point2f target_p = old_points[pargmin];
    std::vector<float> odists = calc_dists(target_p, new_points);
    int oargmin = std::min_element(odists.begin(), odists.end()) - odists.begin();
    // found new point position
    if (oargmin == i)
    {
        return pargmin;
    }
    // did not found new point position; remove old points
    else
    {
        new_points.erase(new_points.begin() + oargmin);
        old_points.erase(old_points.begin() + pargmin);
        // update index
        int new_i = std::find(new_points.begin(), new_points.end(), p) - new_points.begin();
        return this->find_min(new_i, new_points, old_points);
    }
}



void Tracker::update(cv::Mat img)
{
    std::vector<cv::Point2f> new_centers = this->calculate_centers(img, 0);

    if (this->centers.empty())
    {
        for (int i = 0; i < new_centers.size(); i++)
        {
            this->colors.push_back(this->new_color());
            this->ids.push_back(i);
        }
        this->centers = new_centers;
        return;
    }

    // init new variables
    std::vector<cv::Point2f> old_centers(this->centers);
    std::vector<int> new_ids(new_centers.size());
    std::vector<cv::Scalar> new_colors(new_centers.size());

    // find correspondence between centers
    for (int i=0; i<new_centers.size(); i++)
    {
        cv::Point2f p = new_centers[i];
        int index = this->find_min(i, new_centers, old_centers);
        // assign new id
        if (index != -1)
        {
            new_ids[i] = this->ids[index];
            new_colors[i] = this->colors[index];
        }
        else
        {
            // assign new color and id
            int m_id = *std::max_element(this->ids.begin(), this->ids.end());
            int m_nid = *std::max_element(new_ids.begin(), new_ids.end());
            new_ids[i] = std::max(m_id, m_nid) + 1;
            new_colors[i] = this->new_color();
        }
    }

    this->colors = new_colors;
    this->ids = new_ids;
    this->centers = new_centers;
}
