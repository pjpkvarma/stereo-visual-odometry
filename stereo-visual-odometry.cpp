#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>

using namespace std;
using namespace cv;

const float fx = 718.856f;
const float fy = 718.856f;
const double cx = 607.1928;
const double cy = 185.2157;
const double baseline = 0.537;

const int new_width = 640;
const int new_height = 192;
const float scale_x = float(new_width) / 1241.0f;
const float scale_y = float(new_height) / 376.0f;

const float fx_new = fx * scale_x;
const float fy_new = fy * scale_y;
const float cx_new = cx * scale_x;
const float cy_new = cy * scale_y;

Point3f reprojection(float x, float y, float disparity) {
    float Z = fx_new * baseline / disparity;
    float X = (x - cx_new) * Z / fx_new;
    float Y = (y - cy_new) * Z / fy_new;
    return Point3f(X, Y, Z);
}

int main() {
    string base_path = "C:/Users/adams/Documents/c++/stereo-visual-odometry/data_odometry_gray/dataset/sequences/00/";
    int ind = 0;

    Ptr<StereoSGBM> stereo = StereoSGBM::create(0, 96, 9);
    Mat global_pose = Mat::eye(4, 4, CV_64F);
    Ptr<ORB> orb = ORB::create(1000);

    ofstream traj_file("trajectory.txt");
    Mat traj_vis = Mat::zeros(600, 600, CV_8UC3);

    while (true) {
        ostringstream ss;
        ss << setw(6) << setfill('0') << ind << ".png";
        string filename = ss.str();

        Mat left_img = imread(base_path + "image_0/" + filename, IMREAD_GRAYSCALE);
        Mat right_img = imread(base_path + "image_1/" + filename, IMREAD_GRAYSCALE);

        if (left_img.empty() || right_img.empty()) break;

        resize(left_img, left_img, Size(new_width, new_height));
        resize(right_img, right_img, Size(new_width, new_height));

        Mat disparity_raw;
        stereo->compute(left_img, right_img, disparity_raw);
        disparity_raw.convertTo(disparity_raw, CV_16S);

        Mat disparity_vis;
        normalize(disparity_raw, disparity_vis, 0, 255, NORM_MINMAX, CV_8U);
        imshow("Disparity", disparity_vis);

        ostringstream ss2;
        ss2 << setw(6) << setfill('0') << (ind + 1) << ".png";
        Mat next_left = imread(base_path + "image_0/" + ss2.str(), IMREAD_GRAYSCALE);
        if (next_left.empty()) break;

        resize(next_left, next_left, Size(new_width, new_height));

        vector<KeyPoint> kpL;
        Mat descL;
        orb->detectAndCompute(left_img, noArray(), kpL, descL);

        vector<Point2f> points;
        for (auto& kp : kpL) points.push_back(kp.pt);

        vector<Point2f> tracked;
        vector<uchar> status;
        vector<float> err;
        calcOpticalFlowPyrLK(left_img, next_left, points, tracked, status, err);

        vector<Point3f> pts3D;
        vector<Point2f> pts2D;
        for (int i = 0; i < points.size(); ++i) {
            if (!status[i]) continue;

            int x = int(points[i].x);
            int y = int(points[i].y);
            if (x < 0 || y < 0 || x >= disparity_raw.cols || y >= disparity_raw.rows) continue;

            float d = disparity_raw.at<short>(y, x) / 16.0f;
            if (d <= 0.0 || d > 96.0) continue;

            pts3D.push_back(reprojection(x, y, d));
            pts2D.push_back(tracked[i]);
        }

        if (pts3D.size() >= 6) {
            Mat rvec, tvec;
            Mat K = (Mat_<double>(3, 3) << fx_new, 0, cx_new, 0, fy_new, cy_new, 0, 0, 1);
            solvePnPRansac(pts3D, pts2D, K, noArray(), rvec, tvec);

            Mat R;
            Rodrigues(rvec, R);
            Mat T = Mat::eye(4, 4, CV_64F);
            R.copyTo(T(Range(0, 3), Range(0, 3)));
            tvec.copyTo(T(Range(0, 3), Range(3, 4)));

            global_pose = global_pose * T;

            double x = global_pose.at<double>(0, 3);
            double z = global_pose.at<double>(2, 3);
            int draw_x = int(x) + 300;
            int draw_y = int(z) + 100;
            circle(traj_vis, Point(draw_x, draw_y), 1, Scalar(0, 255, 0), 2);
            imshow("Trajectory", traj_vis);

            traj_file << x << " 0 " << z << "\n";
        }

        if (waitKey(1) == 27) break;
        ind++;
    }

    traj_file.close();
    return 0;
}
