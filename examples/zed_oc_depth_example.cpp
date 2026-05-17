///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2021, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

// ----> Includes
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#include "videocapture.hpp"

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

//#undef HAVE_OPENCV_VIZ // Uncomment if cannot use Viz3D for point cloud rendering

#ifdef HAVE_OPENCV_VIZ
#include <opencv2/viz.hpp>
#include <opencv2/viz/viz3d.hpp>
#endif

// Sample includes
#include "calibration.hpp"
#include "stopwatch.hpp"
#include "stereo.hpp"
#include "ocv_display.hpp"
// <---- Includes

#define USE_OCV_TAPI // Comment to use "normal" cv::Mat instead of CV::UMat
#define USE_HALF_SIZE_DISP // Comment to compute depth matching on full image frames

int main(int argc, char *argv[])
{
    // ----> Silence unused warning
    (void)argc;
    (void)argv;
    // <---- Silence unused warning

    sl_oc::VERBOSITY verbose = sl_oc::VERBOSITY::INFO;

    // ----> Set Video parameters
    sl_oc::video::VideoParams params;
#ifdef EMBEDDED_ARM
    params.res = sl_oc::video::RESOLUTION::VGA;
#else
    params.res = sl_oc::video::RESOLUTION::HD1080;
#endif
    params.fps = sl_oc::video::FPS::FPS_30;
    params.verbose = verbose;
    // <---- Set Video parameters

    // ----> Create Video Capture
    sl_oc::video::VideoCapture cap(params);
    if( !cap.initializeVideo(-1) )
    {
        std::cerr << "Cannot open camera video capture" << std::endl;
        std::cerr << "See verbosity level for more details." << std::endl;

        return EXIT_FAILURE;
    }
    int sn = cap.getSerialNumber();
    std::cout << "Connected to camera sn: " << sn << std::endl;

    cap.setAutoWhiteBalance(true);
    // cap.resetAutoWhiteBalance();
    // <---- Create Video Capture

    // ----> Retrieve calibration file from Stereolabs server
    std::string calibration_file;
    // ZED Calibration
    unsigned int serial_number = sn;
    // Download camera calibration file
    if( !sl_oc::tools::downloadCalibrationFile(serial_number, calibration_file) )
    {
        std::cerr << "Could not load calibration file from Stereolabs servers" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Calibration file found. Loading..." << std::endl;

    // ----> Frame size
    int w,h;
    cap.getFrameSize(w,h);
    // <---- Frame size

    // ----> Initialize calibration
    cv::Mat map_left_x, map_left_y;
    cv::Mat map_right_x, map_right_y;
    cv::Mat cameraMatrix_left, cameraMatrix_right;
    double baseline=0;
    sl_oc::tools::initCalibration(calibration_file, cv::Size(w/2,h), map_left_x, map_left_y, map_right_x, map_right_y,
                                  cameraMatrix_left, cameraMatrix_right, &baseline);

    double fx = cameraMatrix_left.at<double>(0,0);
    double fy = cameraMatrix_left.at<double>(1,1);
    double cx = cameraMatrix_left.at<double>(0,2);
    double cy = cameraMatrix_left.at<double>(1,2);


#ifdef USE_OCV_TAPI
    cv::UMat map_left_x_gpu = map_left_x.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat map_left_y_gpu = map_left_y.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat map_right_x_gpu = map_right_x.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat map_right_y_gpu = map_right_y.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
#endif
    // ----> Initialize calibration

    // ----> Declare OpenCV images
#ifdef USE_OCV_TAPI
    cv::UMat frameYUV;  // Full frame side-by-side in YUV 4:2:2 format
    cv::UMat frameBGR(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Full frame side-by-side in BGR format
    cv::UMat left_raw(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Left unrectified image
    cv::UMat right_raw(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Right unrectified image
    cv::UMat left_rect(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Left rectified image
    cv::UMat right_rect(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Right rectified image
    cv::UMat left_for_matcher(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Left image for the stereo matcher
    cv::UMat right_for_matcher(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Right image for the stereo matcher
    cv::UMat left_disp_half(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Half sized disparity map
    cv::UMat left_disp(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Full output disparity
    cv::UMat left_disp_float(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Final disparity map in float32
    cv::UMat left_disp_image(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Normalized and color remapped disparity map to be displayed
    cv::UMat left_depth_map(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Depth map in float32
#else
    cv::Mat frameBGR, left_raw, left_rect, right_raw, right_rect, frameYUV, left_for_matcher, right_for_matcher, left_disp_half,left_disp,left_disp_float, left_disp_vis;
#endif
    // <---- Declare OpenCV images

    // ----> Stereo matcher initialization
    sl_oc::tools::StereoSgbmPar stereoPar;

    //Note: you can use the tool 'zed_open_capture_depth_tune_stereo' to tune the parameters and save them to YAML
    if(!stereoPar.load())
    {
        stereoPar.save(); // Save default parameters.
    }

    cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(stereoPar.minDisparity,stereoPar.numDisparities,stereoPar.blockSize);
    left_matcher->setMinDisparity(stereoPar.minDisparity);
    left_matcher->setNumDisparities(stereoPar.numDisparities);
    left_matcher->setBlockSize(stereoPar.blockSize);
    left_matcher->setP1(stereoPar.P1);
    left_matcher->setP2(stereoPar.P2);
    left_matcher->setDisp12MaxDiff(stereoPar.disp12MaxDiff);
    left_matcher->setMode(stereoPar.mode);
    left_matcher->setPreFilterCap(stereoPar.preFilterCap);
    left_matcher->setUniquenessRatio(stereoPar.uniquenessRatio);
    left_matcher->setSpeckleWindowSize(stereoPar.speckleWindowSize);
    left_matcher->setSpeckleRange(stereoPar.speckleRange);

    stereoPar.print();
    // <---- Stereo matcher initialization


    // Cached PnP result — updated whenever all 4 markers are visible
    cv::Mat cached_R, cached_tvec;
    bool pnp_valid = false;

    // ArUco setup — markers 1,2,3,4 printed from DICT_4X4_50
    cv::Ptr<cv::aruco::Dictionary> aruco_dict =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params =
        cv::aruco::DetectorParameters::create();

    uint64_t last_ts=0; // Used to check new frame arrival

    // Infinite video grabbing loop
    while (1)
    {
        // Get a new frame from camera
        const sl_oc::video::Frame frame = cap.getLastFrame();

        // ----> If the frame is valid we can convert, rectify and display it
        if(frame.data!=nullptr && frame.timestamp!=last_ts)
        {
            last_ts = frame.timestamp;

            // ----> Conversion from YUV 4:2:2 to BGR for visualization
#ifdef USE_OCV_TAPI
            // Clone immediately: frame.data is a raw pointer into the grab thread's
            // internal buffer, which can be overwritten mid-frame causing black flashes.
            cv::Mat frameYUV_cpu = cv::Mat( frame.height, frame.width, CV_8UC2, frame.data ).clone();
            frameYUV = frameYUV_cpu.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_HOST_MEMORY);
#else
            frameYUV = cv::Mat( frame.height, frame.width, CV_8UC2, frame.data ).clone();
#endif
            cv::cvtColor(frameYUV,frameBGR,cv::COLOR_YUV2BGR_YUYV);
            // <---- Conversion from YUV 4:2:2 to BGR for visualization

            // ----> Extract left and right images from side-by-side
            left_raw = frameBGR(cv::Rect(0, 0, frameBGR.cols / 2, frameBGR.rows));
            right_raw = frameBGR(cv::Rect(frameBGR.cols / 2, 0, frameBGR.cols / 2, frameBGR.rows));
            // <---- Extract left and right images from side-by-side

            // ----> Apply rectification
            sl_oc::tools::StopWatch remap_clock;
#ifdef USE_OCV_TAPI
            cv::remap(left_raw, left_rect, map_left_x_gpu, map_left_y_gpu, cv::INTER_AREA );
            cv::remap(right_raw, right_rect, map_right_x_gpu, map_right_y_gpu, cv::INTER_AREA );
#else
            cv::remap(left_raw, left_rect, map_left_x, map_left_y, cv::INTER_AREA );
            cv::remap(right_raw, right_rect, map_right_x, map_right_y, cv::INTER_AREA );
#endif
            double remap_elapsed = remap_clock.toc();
            std::stringstream remapElabInfo;
            remapElabInfo << "Rectif. processing: " << remap_elapsed << " sec - Freq: " << 1./remap_elapsed;
            // <---- Apply rectification

            // ----> Stereo matching
            sl_oc::tools::StopWatch stereo_clock;
            double resize_fact = 1.0;
#ifdef USE_HALF_SIZE_DISP
            resize_fact = 0.5;
            // Resize the original images to improve performances
            cv::resize(left_rect,  left_for_matcher,  cv::Size(), resize_fact, resize_fact, cv::INTER_AREA);
            cv::resize(right_rect, right_for_matcher, cv::Size(), resize_fact, resize_fact, cv::INTER_AREA);
#else
            left_for_matcher = left_rect; // No data copy
            right_for_matcher = right_rect; // No data copy
#endif
            // Apply stereo matching
            left_matcher->compute(left_for_matcher, right_for_matcher,left_disp_half);

            left_disp_half.convertTo(left_disp_float,CV_32FC1);
            cv::multiply(left_disp_float,1./16.,left_disp_float); // Last 4 bits of SGBM disparity are decimal

#ifdef USE_HALF_SIZE_DISP
            cv::multiply(left_disp_float,2.,left_disp_float); // Last 4 bits of SGBM disparity are decimal
            cv::UMat tmp = left_disp_float; // Required for OpenCV 3.2
            cv::resize(tmp, left_disp_float, cv::Size(), 1./resize_fact, 1./resize_fact, cv::INTER_AREA);
#else
            left_disp = left_disp_float;
#endif


            double elapsed = stereo_clock.toc();
            std::stringstream stereoElabInfo;
            stereoElabInfo << "Stereo processing: " << elapsed << " sec - Freq: " << 1./elapsed;
            // <---- Stereo matching

            // ----> Extract Depth map
            // depth = (f * B) / disparity
            cv::add(left_disp_float,-static_cast<double>(stereoPar.minDisparity-1),left_disp_float); // Minimum disparity offset correction
            double num = static_cast<double>(fx*baseline);
            cv::divide(num,left_disp_float,left_depth_map);

            float central_depth = left_depth_map.getMat(cv::ACCESS_READ).at<float>(left_depth_map.rows/2, left_depth_map.cols/2 );
            // std::cout << "Depth of the central pixel: " << central_depth << " mm" << std::endl;
            // <---- Extract Depth map

            // ----> Green balloon detection + combined display
            {
                cv::Mat right_cpu = right_rect.getMat(cv::ACCESS_READ).clone();
                cv::Mat depth_mat = left_depth_map.getMat(cv::ACCESS_READ);


                // ----> Green balloon detection
                cv::Point2f balloon_img_pt(-1, -1);
                float balloon_depth_mm = -1;
                {
                    cv::Mat hsv, mask;
                    cv::cvtColor(right_cpu, hsv, cv::COLOR_BGR2HSV);
                    cv::inRange(hsv, cv::Scalar(35, 60, 40), cv::Scalar(85, 255, 255), mask);

                    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
                    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  kernel);
                    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

                    std::vector<std::vector<cv::Point>> contours;
                    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                    double max_area = 0;
                    int best = -1;
                    for (int i = 0; i < (int)contours.size(); i++)
                    {
                        double a = cv::contourArea(contours[i]);
                        if (a > max_area) { max_area = a; best = i; }
                    }

                    if (best >= 0 && max_area > 500)
                    {
                        cv::Rect bbox = cv::boundingRect(contours[best]);
                        int bx = std::min(bbox.x + bbox.width  / 2, depth_mat.cols - 1);
                        int by = std::min(bbox.y + bbox.height / 2, depth_mat.rows - 1);
                        balloon_img_pt    = cv::Point2f(bx, by);
                        balloon_depth_mm  = depth_mat.at<float>(by, bx);

                        cv::rectangle(right_cpu, bbox, cv::Scalar(0, 255, 0), 2);
                    }
                }
                // <---- Green balloon detection

                // ----> ArUco marker detection, rectangle boundary, PnP + balloon world position
                {
                    std::vector<std::vector<cv::Point2f>> corners, rejected;
                    std::vector<int> ids;
                    cv::aruco::detectMarkers(right_cpu, aruco_dict, corners, ids,
                                            aruco_params, rejected);
                    cv::aruco::drawDetectedMarkers(right_cpu, corners, ids);

                    // float-precision centers keyed by marker ID
                    std::map<int, cv::Point2f> centers_f;
                    for (size_t i = 0; i < ids.size(); i++)
                    {
                        cv::Point2f c(0, 0);
                        for (auto& pt : corners[i]) c += pt;
                        centers_f[ids[i]] = c * 0.25f;
                    }

                    const int order[4] = {1, 2, 3, 4}; // clockwise: TL, TR, BR, BL
                    bool all_found = true;
                    for (int id : order)
                        if (centers_f.find(id) == centers_f.end()) { all_found = false; break; }

                    if (all_found)
                    {
                        // draw rectangle overlay
                        std::vector<cv::Point> pts;
                        for (int id : order) pts.push_back(cv::Point(centers_f[id]));

                        cv::Mat overlay = right_cpu.clone();
                        cv::fillConvexPoly(overlay, pts, cv::Scalar(0, 200, 255));
                        cv::addWeighted(overlay, 0.15, right_cpu, 0.85, 0, right_cpu);

                        for (int i = 0; i < 4; i++)
                            cv::line(right_cpu, pts[i], pts[(i+1)%4],
                                     cv::Scalar(0, 200, 255), 2, cv::LINE_AA);

                        const char* corner_labels[4] = {"1(TL)", "2(TR)", "3(BR)", "4(BL)"};
                        for (int i = 0; i < 4; i++)
                            cv::putText(right_cpu, corner_labels[i],
                                        pts[i] + cv::Point(8, -8),
                                        cv::FONT_HERSHEY_SIMPLEX, 0.55,
                                        cv::Scalar(0, 200, 255), 2);

                        // PnP: world frame origin at marker 1, X along 1->4, Y along 1->2
                        // Units: metres
                        std::vector<cv::Point3f> obj_pts = {
                            {0.0f,   0.0f,   0.0f},  // marker 1 TL
                            {0.0f,   2.415f, 0.0f},  // marker 2 TR
                            {1.265f, 2.415f, 0.0f},  // marker 3 BR
                            {1.265f, 0.0f,   0.0f},  // marker 4 BL
                        };
                        std::vector<cv::Point2f> img_pts = {
                            centers_f[1], centers_f[2], centers_f[3], centers_f[4]
                        };

                        // 3x3 intrinsic matrix (same for both rectified cameras)
                        cv::Mat K = (cv::Mat_<double>(3,3)
                            << fx, 0, cx,
                               0, fy, cy,
                               0,  0,  1);

                        cv::Mat rvec, tvec;
                        cv::solvePnP(obj_pts, img_pts, K, cv::Mat(), rvec, tvec);
                        cv::Rodrigues(rvec, cached_R);
                        cached_tvec = tvec;
                        pnp_valid = true;
                    }
                }

                // Use cached extrinsics for camera + balloon positions
                if (pnp_valid)
                {
                    cv::Mat cam_world = -cached_R.t() * cached_tvec;
                    std::ostringstream cam_info;
                    cam_info << std::fixed << std::setprecision(2)
                             << "Cam: ("
                             << cam_world.at<double>(0) << ", "
                             << cam_world.at<double>(1) << ", "
                             << cam_world.at<double>(2) << ") m";
                    cv::putText(right_cpu, cam_info.str(), cv::Point(10, right_cpu.rows - 40),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 100), 2);

                    if (balloon_img_pt.x >= 0 &&
                        balloon_depth_mm > stereoPar.minDepth_mm &&
                        balloon_depth_mm < stereoPar.maxDepth_mm)
                    {
                        double Z_m = balloon_depth_mm / 1000.0;
                        cv::Mat P_cam = (cv::Mat_<double>(3,1)
                            << (balloon_img_pt.x - cx) * Z_m / fx,
                               (balloon_img_pt.y - cy) * Z_m / fy,
                               Z_m);

                        cv::Mat P_world = cached_R.t() * (P_cam - cached_tvec);

                        std::ostringstream pos;
                        pos << std::fixed << std::setprecision(2)
                            << "Balloon: ("
                            << P_world.at<double>(0) << ", "
                            << P_world.at<double>(1) << ", "
                            << P_world.at<double>(2) << ") m";
                        cv::putText(right_cpu, pos.str(),
                                    cv::Point(balloon_img_pt.x + 10, balloon_img_pt.y + 20),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                    }
                }
                // <---- ArUco marker detection, rectangle boundary, PnP + balloon world position

                // scale to fit screen
                const double scale_w = (double)1800 / right_cpu.cols;
                const double scale_h = (double)900  / right_cpu.rows;
                const double scale = std::min(1.0, std::min(scale_w, scale_h));
                cv::Mat right_small;
                cv::resize(right_cpu, right_small, cv::Size(), scale, scale);
                cv::imshow("Right", right_small);
            }
            // <---- Green balloon detection + combined display

        }

        // ----> Keyboard handling
        int key = cv::waitKey( 5 );
        if(key=='q' || key=='Q') // Quit
            break;
        // <---- Keyboard handling
    }

    return EXIT_SUCCESS;
}


