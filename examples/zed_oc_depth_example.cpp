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
#include <fstream>
#include <string>
#include <chrono>
#include <cmath>
#include <ctime>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

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

#include <librealsense2/rs.hpp>
// <---- Includes

// ---- Recording helpers -----------------------------------------------------

static void rec_mkdirp(const std::string& p)
{
    int r = system(("mkdir -p \"" + p + "\"").c_str());
    if (r != 0) std::cerr << "Warning: mkdir -p failed: " << p << "\n";
}

// Depth mm → colormap BGR for video encoding (0–8000 mm, TURBO colormap)
static cv::Mat depth_to_colormap(const cv::Mat& depth_mm)
{
    cv::Mat gray, color;
    depth_mm.convertTo(gray, CV_8U, 255.0 / 8000.0);
    cv::applyColorMap(gray, color, cv::COLORMAP_TURBO);
    return color;
}

// Frame queued from the main ZED loop to the recording thread
struct RecFrame {
    cv::Mat left_rect;   // full-res left rectified (BGR)
    cv::Mat right_rect;  // full-res right rectified (BGR)
    double  timestamp_s;
};

// RealSense recording thread: grab + write color and depth to MP4
static void rs_record_thread(
    rs2::pipeline& pipe, rs2::align& align_to_color, float depth_scale,
    const std::string& color_path, const std::string& depth_path,
    const std::string& csv_path,
    std::atomic<bool>& stop)
{
    std::ofstream csv(csv_path);
    csv << "frame,timestamp_s\n";
    const int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
    cv::VideoWriter color_vw, depth_vw;
    int idx = 0;
    while (!stop) {
        rs2::frameset frames;
        if (!pipe.try_wait_for_frames(&frames, 100)) continue;
        frames = align_to_color.process(frames);
        rs2::video_frame cf = frames.get_color_frame();
        rs2::depth_frame df = frames.get_depth_frame();
        if (!cf || !df) continue;
        int W = cf.get_width(), H = cf.get_height();
        if (!color_vw.isOpened()) {
            color_vw.open(color_path, fourcc, 30.0, cv::Size(W, H), true);
            depth_vw.open(depth_path, fourcc, 30.0, cv::Size(W, H), true);
        }
        cv::Mat color(H, W, CV_8UC3, const_cast<void*>(cf.get_data()));
        color_vw.write(color.clone());
        cv::Mat depth_raw(H, W, CV_16UC1, const_cast<void*>(df.get_data()));
        cv::Mat depth_f;
        depth_raw.convertTo(depth_f, CV_32F);
        cv::multiply(depth_f, depth_scale * 1000.f, depth_f);
        cv::Mat depth_mm;
        depth_f.convertTo(depth_mm, CV_16UC1);
        depth_vw.write(depth_to_colormap(depth_mm));
        csv << idx << "," << std::fixed << std::setprecision(6)
            << cf.get_timestamp() / 1000.0 << "\n";
        idx++;
    }
}

// ZED recording thread: receive rectified pairs, compute full SGBM, write to MP4
static void zed_record_thread(
    std::queue<RecFrame>& q, std::mutex& q_mtx,
    double fx, double baseline,
    const sl_oc::tools::StereoSgbmPar& par,
    const std::string& color_path, const std::string& depth_path,
    const std::string& csv_path,
    std::atomic<bool>& stop)
{
    cv::Ptr<cv::StereoSGBM> matcher =
        cv::StereoSGBM::create(par.minDisparity, par.numDisparities, par.blockSize);
    matcher->setMinDisparity(par.minDisparity);
    matcher->setNumDisparities(par.numDisparities);
    matcher->setBlockSize(par.blockSize);
    matcher->setP1(par.P1);  matcher->setP2(par.P2);
    matcher->setDisp12MaxDiff(par.disp12MaxDiff);
    matcher->setMode(par.mode);
    matcher->setPreFilterCap(par.preFilterCap);
    matcher->setUniquenessRatio(par.uniquenessRatio);
    matcher->setSpeckleWindowSize(par.speckleWindowSize);
    matcher->setSpeckleRange(par.speckleRange);

    std::ofstream csv(csv_path);
    csv << "frame,timestamp_s\n";
    const int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
    cv::VideoWriter color_vw, depth_vw;
    int idx = 0;

    while (!stop || !q.empty()) {
        RecFrame frm;
        {
            std::lock_guard<std::mutex> lk(q_mtx);
            if (q.empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }
            frm = std::move(q.front());
            q.pop();
        }
        if (!color_vw.isOpened()) {
            color_vw.open(color_path, fourcc, 5.0,
                          cv::Size(frm.left_rect.cols, frm.left_rect.rows), true);
            depth_vw.open(depth_path, fourcc, 5.0,
                          cv::Size(frm.left_rect.cols, frm.left_rect.rows), true);
        }
        // Half-size stereo for SGBM
        cv::Mat lh, rh, disp_raw, disp_f;
        cv::resize(frm.left_rect,  lh, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
        cv::resize(frm.right_rect, rh, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
        matcher->compute(lh, rh, disp_raw);
        disp_raw.convertTo(disp_f, CV_32F, 1.0 / 16.0);
        cv::multiply(disp_f, 2.0, disp_f);  // restore full-res scale
        // Disparity → depth mm
        cv::Mat depth_half(disp_f.rows, disp_f.cols, CV_16UC1, cv::Scalar(0));
        for (int r = 0; r < disp_f.rows; r++) {
            const float* dp = disp_f.ptr<float>(r);
            uint16_t*    zp = depth_half.ptr<uint16_t>(r);
            for (int c = 0; c < disp_f.cols; c++) {
                if (dp[c] > par.minDisparity) {
                    float dm = (float)(fx * baseline / dp[c]);
                    if (dm > par.minDepth_mm && dm < par.maxDepth_mm)
                        zp[c] = (uint16_t)dm;
                }
            }
        }
        cv::Mat depth_full;
        cv::resize(depth_half, depth_full,
                   cv::Size(frm.left_rect.cols, frm.left_rect.rows),
                   0, 0, cv::INTER_NEAREST);
        color_vw.write(frm.left_rect);
        depth_vw.write(depth_to_colormap(depth_full));
        csv << idx << "," << std::fixed << std::setprecision(6)
            << frm.timestamp_s << "\n";
        idx++;
    }
}
// ---- End recording helpers -------------------------------------------------

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


    // Cached PnP result
    cv::Mat cached_R, cached_tvec;
    bool pnp_valid = false;

    // ArUco setup — markers 1,2,3,4 printed from DICT_4X4_50
    cv::Ptr<cv::aruco::Dictionary> aruco_dict =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params =
        cv::aruco::DetectorParameters::create();

    // World points: origin at marker 1, X along 1->4, Y along 1->2, metres
    const std::vector<cv::Point3f> aruco_obj_pts = {
        {0.0f,   0.0f,   0.0f},
        {0.0f,   2.415f, 0.0f},
        {1.265f, 2.415f, 0.0f},
        {1.265f, 0.0f,   0.0f},
    };
    const cv::Mat K_mat = (cv::Mat_<double>(3,3)
        << fx, 0, cx,
           0, fy, cy,
           0,  0,  1);

    // Cached image-space marker centers (keyed by marker ID 1-4)
    std::map<int, cv::Point2f> cached_centers;
    bool detect_aruco = true;  // false once corners are loaded or found

    // Try to load saved corners from file
    const std::string corners_file = "aruco_corners.yml";
    {
        cv::FileStorage fs(corners_file, cv::FileStorage::READ);
        if (fs.isOpened())
        {
            bool valid = true;
            for (int id : {1, 2, 3, 4})
            {
                cv::Point2f pt;
                fs["marker_" + std::to_string(id)] >> pt;
                if (pt.x == 0 && pt.y == 0) { valid = false; break; }
                cached_centers[id] = pt;
            }
            fs.release();

            if (valid)
            {
                std::vector<cv::Point2f> img_pts = {
                    cached_centers[1], cached_centers[2],
                    cached_centers[3], cached_centers[4]
                };
                cv::Mat rvec, tvec;
                cv::solvePnP(aruco_obj_pts, img_pts, K_mat, cv::Mat(), rvec, tvec);
                cv::Rodrigues(rvec, cached_R);
                cached_tvec = tvec;
                pnp_valid = true;
                detect_aruco = false;
                std::cout << "Loaded ArUco corners from " << corners_file << std::endl;
            }
        }
    }

    // Connect to Redis on localhost:6379 (persistent connection, RESP protocol)
    int redis_fd = -1;
    {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd >= 0) {
            struct sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port   = htons(6379);
            addr.sin_addr.s_addr = inet_addr("127.0.0.1");
            if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                redis_fd = fd;
                std::cout << "Connected to Redis on localhost:6379" << std::endl;
            } else {
                ::close(fd);
                std::cerr << "Cannot connect to Redis on localhost:6379" << std::endl;
            }
        }
    }

    // Send a Redis SET command over the persistent connection
    auto redis_set = [&redis_fd](const std::string& key, const std::string& val) {
        if (redis_fd < 0) return;
        std::string cmd = "*3\r\n$3\r\nSET\r\n$" + std::to_string(key.size()) + "\r\n"
            + key + "\r\n$" + std::to_string(val.size()) + "\r\n" + val + "\r\n";
        if (::send(redis_fd, cmd.c_str(), cmd.size(), MSG_NOSIGNAL) < 0) {
            ::close(redis_fd);
            redis_fd = -1;
        } else {
            char buf[32];
            ::recv(redis_fd, buf, sizeof(buf), MSG_DONTWAIT); // drain "+OK\r\n"
        }
    };

    // Last valid velocity command — held for up to this many frames on detection dropout
    double last_vx = 0.0, last_vy = 0.0, last_vyaw = 0.0;
    int dropout_frames = 0;
    const int max_dropout_frames = 5;

    // Balloon Z tracking for fall detection
    double prev_balloon_z  = -1.0;
    double prev_balloon_ts = -1.0;

    uint64_t last_ts=0; // Used to check new frame arrival

    // ----> Recording setup
    auto rec_now = std::chrono::system_clock::now();
    std::time_t rec_t = std::chrono::system_clock::to_time_t(rec_now);
    char rec_ts[32];
    std::strftime(rec_ts, sizeof(rec_ts), "%Y%m%d_%H%M%S", std::localtime(&rec_t));
    std::string rec_out = std::string("recording_") + rec_ts;
    rec_mkdirp(rec_out);
    std::cout << "Recording to: " << rec_out << std::endl;

    std::atomic<bool> rec_stop{false};

    // ZED recording queue + thread
    std::queue<RecFrame> zed_q;
    std::mutex           zed_q_mtx;
    std::thread zed_rec_th(zed_record_thread,
        std::ref(zed_q), std::ref(zed_q_mtx),
        fx, baseline, std::cref(stereoPar),
        rec_out + "/zed_color.mp4", rec_out + "/zed_depth.mp4",
        rec_out + "/zed_timestamps.csv",
        std::ref(rec_stop));

    // RealSense recording thread (optional — skipped if no RS camera found)
    rs2::pipeline    rs_pipe;
    rs2::align       rs_align(RS2_STREAM_COLOR);
    float            rs_depth_scale = 0.001f;
    bool             rs_ok = false;
    std::thread      rs_rec_th;
    try {
        rs2::config rs_cfg;
        rs_cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 30);
        rs_cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16,  30);
        rs2::pipeline_profile rs_prof = rs_pipe.start(rs_cfg);
        for (auto s : rs_prof.get_device().query_sensors())
            if (auto ds = s.as<rs2::depth_sensor>())
                { rs_depth_scale = ds.get_depth_scale(); break; }
        rs_ok = true;
        std::cout << "RealSense started (depth scale " << rs_depth_scale << " m/unit)" << std::endl;
        rs_rec_th = std::thread(rs_record_thread,
            std::ref(rs_pipe), std::ref(rs_align), rs_depth_scale,
            rec_out + "/rs_color.mp4", rec_out + "/rs_depth.mp4",
            rec_out + "/rs_timestamps.csv",
            std::ref(rec_stop));
    } catch (const rs2::error& e) {
        std::cerr << "RealSense not available, skipping RS recording: " << e.what() << std::endl;
    }
    // <---- Recording setup

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

            // ----> Queue ZED frame for recording (non-blocking, drop if backed up)
            {
                cv::Mat lr = left_rect.getMat(cv::ACCESS_READ).clone();
                cv::Mat rr = right_rect.getMat(cv::ACCESS_READ).clone();
                std::lock_guard<std::mutex> lk(zed_q_mtx);
                if (zed_q.size() < 4)
                    zed_q.push({lr, rr, frame.timestamp / 1e9});
            }
            // <---- Queue ZED frame for recording

            // Resize rectified images for on-demand ROI stereo matching
            double resize_fact = 1.0;
#ifdef USE_HALF_SIZE_DISP
            resize_fact = 0.5;
            cv::resize(left_rect,  left_for_matcher,  cv::Size(), resize_fact, resize_fact, cv::INTER_AREA);
            cv::resize(right_rect, right_for_matcher, cv::Size(), resize_fact, resize_fact, cv::INTER_AREA);
#else
            left_for_matcher  = left_rect;
            right_for_matcher = right_rect;
#endif

            // ----> Green balloon detection + combined display
            {
                cv::Mat right_cpu = right_rect.getMat(cv::ACCESS_READ).clone();

                // Run SGBM on a small ROI around a point; returns depth in mm, -1 if invalid.
                // ROI extends numDisparities pixels left of the point so the full disparity
                // search range is contained within the same crop for both images.
                auto depth_at_pt = [&](int px, int py) -> float {
                    int phx = (int)std::round(px * resize_fact);
                    int phy = (int)std::round(py * resize_fact);
                    int nd  = stereoPar.numDisparities;
                    const int mar = 15;
                    int x0 = std::max(0, phx - nd - mar);
                    int y0 = std::max(0, phy - mar);
                    int x1 = std::min(left_for_matcher.cols, phx + mar + 1);
                    int y1 = std::min(left_for_matcher.rows, phy + mar + 1);
                    if (x1 <= x0 || y1 <= y0) return -1.f;
                    int rw = x1 - x0, rh = y1 - y0;
                    // SGBM requires contiguous memory and width > numDisparities
                    if (rw <= nd || rh < 3) return -1.f;
                    cv::Rect roi(x0, y0, rw, rh);
                    cv::UMat left_crop, right_crop, roi_disp, roi_disp_f;
                    left_for_matcher(roi).copyTo(left_crop);
                    right_for_matcher(roi).copyTo(right_crop);
                    left_matcher->compute(left_crop, right_crop, roi_disp);
                    roi_disp.convertTo(roi_disp_f, CV_32FC1);
                    cv::multiply(roi_disp_f, 1./16., roi_disp_f);
                    if (resize_fact < 1.0) cv::multiply(roi_disp_f, 1./resize_fact, roi_disp_f);
                    cv::Mat dm = roi_disp_f.getMat(cv::ACCESS_READ);
                    int lx = phx - x0, ly = phy - y0;
                    std::vector<float> samp;
                    for (int dy = -4; dy <= 4; dy++)
                        for (int dx = -4; dx <= 4; dx++) {
                            float d = dm.at<float>(std::max(0,std::min(ly+dy,dm.rows-1)),
                                                   std::max(0,std::min(lx+dx,dm.cols-1)));
                            if (d > stereoPar.minDisparity) samp.push_back(d);
                        }
                    if (samp.empty()) return -1.f;
                    auto mid = samp.begin() + samp.size()/2;
                    std::nth_element(samp.begin(), mid, samp.end());
                    float disp = *mid - (float)(stereoPar.minDisparity - 1);
                    if (disp <= 0) return -1.f;
                    float depth = (float)(fx * baseline / disp);
                    return (depth > stereoPar.minDepth_mm && depth < stereoPar.maxDepth_mm) ? depth : -1.f;
                };

                cv::Mat dog_P_world;      // empty = dog not localised this frame
                double  dog_yaw = 0.0;
                cv::Mat balloon_P_world;  // empty = balloon not localised this frame
                bool    balloon_in_bounds = false;

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
                        int bx = bbox.x + bbox.width  / 2;
                        int by = bbox.y + bbox.height / 2;
                        balloon_img_pt   = cv::Point2f(bx, by);
                        balloon_depth_mm = depth_at_pt(bx, by);

                        cv::rectangle(right_cpu, bbox, cv::Scalar(0, 255, 0), 2);
                    }
                }
                // <---- Green balloon detection

                // ----> ArUco marker detection, rectangle boundary, PnP + balloon world position
                {
                    // Always detect — needed for dynamic dog marker (ID 0)
                    std::vector<std::vector<cv::Point2f>> all_corners, rejected;
                    std::vector<int> all_ids;
                    cv::aruco::detectMarkers(right_cpu, aruco_dict, all_corners, all_ids,
                                            aruco_params, rejected);

                    // Partition into dog (ID 0) and ground markers (IDs 1-4)
                    int dog_idx = -1;
                    std::map<int, cv::Point2f> detected_ground;
                    std::vector<std::vector<cv::Point2f>> dog_corners_vec;
                    std::vector<int> dog_ids_vec;

                    for (size_t i = 0; i < all_ids.size(); i++)
                    {
                        cv::Point2f c(0, 0);
                        for (auto& pt : all_corners[i]) c += pt;
                        c *= 0.25f;
                        if (all_ids[i] == 0) {
                            dog_idx = (int)i;
                            dog_corners_vec.push_back(all_corners[i]);
                            dog_ids_vec.push_back(0);
                        } else {
                            detected_ground[all_ids[i]] = c;
                        }
                    }

                    // Ground markers 1-4: update cache if detect_aruco
                    if (detect_aruco)
                    {
                        bool all_found = true;
                        for (int id : {1, 2, 3, 4})
                            if (detected_ground.find(id) == detected_ground.end()) { all_found = false; break; }

                        if (all_found)
                        {
                            for (int id : {1, 2, 3, 4}) cached_centers[id] = detected_ground[id];

                            std::vector<cv::Point2f> img_pts = {
                                cached_centers[1], cached_centers[2],
                                cached_centers[3], cached_centers[4]
                            };
                            cv::Mat rvec, tvec;
                            cv::solvePnP(aruco_obj_pts, img_pts, K_mat, cv::Mat(), rvec, tvec);
                            cv::Rodrigues(rvec, cached_R);
                            cached_tvec = tvec;
                            pnp_valid = true;
                            detect_aruco = false;

                            cv::FileStorage fs(corners_file, cv::FileStorage::WRITE);
                            for (int id : {1, 2, 3, 4})
                                fs << "marker_" + std::to_string(id) << cached_centers[id];
                            fs.release();
                            std::cout << "Saved ArUco corners to " << corners_file << std::endl;
                        }
                    }

                    // Draw cached ground rectangle
                    if (!cached_centers.empty())
                    {
                        const int order[4] = {1, 2, 3, 4};
                        std::vector<cv::Point> pts;
                        for (int id : order) pts.push_back(cv::Point(cached_centers[id]));

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
                    }

                    // Dog marker (ID 0): draw outline + forward arrow + world position label
                    cv::aruco::drawDetectedMarkers(right_cpu, dog_corners_vec, dog_ids_vec);
                    if (dog_idx >= 0)
                    {
                        const auto& dc = all_corners[dog_idx];
                        // ArUco corners: [TL, TR, BR, BL]; tag "up" = top edge - bottom edge
                        cv::Point2f top_mid = (dc[0] + dc[1]) * 0.5f;
                        cv::Point2f bot_mid = (dc[3] + dc[2]) * 0.5f;
                        cv::Point2f center  = (top_mid + bot_mid) * 0.5f;
                        cv::Point2f fwd     = top_mid - bot_mid; // forward direction in image

                        cv::arrowedLine(right_cpu,
                                        cv::Point(center), cv::Point(center + fwd),
                                        cv::Scalar(0, 165, 255), 2, cv::LINE_AA, 0, 0.25);

                        if (pnp_valid)
                        {
                            float dog_depth_mm = depth_at_pt((int)center.x, (int)center.y);

                            if (dog_depth_mm > 0)
                            {
                                double Z_m = dog_depth_mm / 1000.0;
                                cv::Mat P_cam = (cv::Mat_<double>(3,1)
                                    << (center.x - cx) * Z_m / fx,
                                       (center.y - cy) * Z_m / fy,
                                       Z_m);
                                cv::Mat P_world = cached_R.t() * (P_cam - cached_tvec);

                                // Unproject forward-arrow tip at same depth → world-frame heading
                                cv::Point2f fwd_tip = center + fwd;
                                cv::Mat P_fwd_cam = (cv::Mat_<double>(3,1)
                                    << (fwd_tip.x - cx) * Z_m / fx,
                                       (fwd_tip.y - cy) * Z_m / fy,
                                       Z_m);
                                cv::Mat heading = cached_R.t() * (P_fwd_cam - cached_tvec) - P_world;
                                dog_yaw     = std::atan2(heading.at<double>(1), heading.at<double>(0));
                                dog_P_world = P_world.clone();

                                std::ostringstream dog_lbl;
                                dog_lbl << std::fixed << std::setprecision(2)
                                        << "Dog: ("
                                        << P_world.at<double>(0) << ", "
                                        << P_world.at<double>(1) << ", "
                                        << P_world.at<double>(2) << ") m";
                                cv::putText(right_cpu, dog_lbl.str(),
                                            cv::Point(center.x + 10, center.y - 10),
                                            cv::FONT_HERSHEY_SIMPLEX, 0.7,
                                            cv::Scalar(0, 165, 255), 2);
                            }
                        }
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

                    if (balloon_img_pt.x >= 0 && balloon_depth_mm > 0)
                    {
                        double Z_m = balloon_depth_mm / 1000.0;
                        cv::Mat P_cam = (cv::Mat_<double>(3,1)
                            << (balloon_img_pt.x - cx) * Z_m / fx,
                               (balloon_img_pt.y - cy) * Z_m / fy,
                               Z_m);

                        cv::Mat P_world = cached_R.t() * (P_cam - cached_tvec);
                        balloon_P_world = P_world.clone();

                        double bpx = P_world.at<double>(0);
                        double bpy = P_world.at<double>(1);
                        balloon_in_bounds = (bpx >= 0.0 && bpx <= 1.265 &&
                                             bpy >= 0.0 && bpy <= 2.415);

                        std::ostringstream pos;
                        pos << std::fixed << std::setprecision(2)
                            << "Balloon: ("
                            << bpx << ", " << bpy << ", "
                            << P_world.at<double>(2) << ") m";
                        if (balloon_in_bounds) pos << " [IN]";
                        cv::putText(right_cpu, pos.str(),
                                    cv::Point(balloon_img_pt.x + 10, balloon_img_pt.y + 20),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                                    balloon_in_bounds ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0), 2);
                    }
                }
                // <---- ArUco marker detection, rectangle boundary, PnP + balloon world position

                // ----> Velocity command: move dog under balloon when balloon is in rectangle bounds
                {
                    double ts = frame.timestamp / 1e9; // camera frame timestamp, nanoseconds → seconds

                    // Balloon vertical velocity and tilt trigger
                    double balloon_vz = 0.0;
                    bool tilt = false;
                    if (!balloon_P_world.empty()) {
                        double cur_z = balloon_P_world.at<double>(2);
                        if (prev_balloon_z >= 0.0 && prev_balloon_ts > 0.0) {
                            double dt = ts - prev_balloon_ts;
                            if (dt > 0.0) balloon_vz = (cur_z - prev_balloon_z) / dt;
                        }
                        // One-shot: fire only on the downward crossing of 1.2m,
                        // and only if the balloon XY is within 10cm of the dog's head
                        bool near_head = false;
                        if (!dog_P_world.empty()) {
                            double hx = dog_P_world.at<double>(0) + 0.22 * std::cos(dog_yaw);
                            double hy = dog_P_world.at<double>(1) + 0.22 * std::sin(dog_yaw);
                            double ddx = balloon_P_world.at<double>(0) - hx;
                            double ddy = balloon_P_world.at<double>(1) - hy;
                            near_head = std::sqrt(ddx*ddx + ddy*ddy) <= 0.10;
                        }
                        tilt = (prev_balloon_z >= 1.2 && cur_z < 1.2 && balloon_vz < 0.0 /* && near_head */);
                        prev_balloon_z  = cur_z;
                        prev_balloon_ts = ts;
                    }

                    double vx = 0.0, vy = 0.0, vyaw = 0.0;

                    if (balloon_in_bounds && !dog_P_world.empty() && !balloon_P_world.empty()
                        && balloon_P_world.at<double>(2) >= 0.5)
                    {
                        dropout_frames = 0;
                        const double gain     = 6.0;
                        const double max_v    = 1.0;
                        const double vy_gain  = 2.0;
                        const double max_vy   = 0.6;
                        const double yaw_gain = 1.2;
                        const double max_vyaw = 1.0;

                        // Target point: 22 cm in front of the ArUco tag (where the head is)
                        double head_x = dog_P_world.at<double>(0) + 0.22 * std::cos(dog_yaw);
                        double head_y = dog_P_world.at<double>(1) + 0.22 * std::sin(dog_yaw);
                        double ex = balloon_P_world.at<double>(0) - head_x;
                        double ey = balloon_P_world.at<double>(1) - head_y;

                        // Rotate world-frame error into dog body frame (x=forward, y=left)
                        double c = std::cos(dog_yaw), s = std::sin(dog_yaw);
                        double ex_b =  ex * c + ey * s;
                        double ey_b = -ex * s + ey * c;

                        double angle_err = std::atan2(ey_b, ex_b);
                        bool behind = std::abs(angle_err) > M_PI_2;

                        // Quadratic ramp: preserves sign, grows much faster with distance
                        auto sq = [](double v){ return std::copysign(v * v, v); };

                        if (!behind) {
                            // Balloon in front hemisphere: turn and drive forward
                            vyaw = std::max(-max_vyaw, std::min(max_vyaw, angle_err * yaw_gain));
                            vx   = std::max(0.0,      std::min(max_v,    sq(ex_b) * gain));
                            vy   = std::max(-max_vy,  std::min(max_vy,   sq(ey_b) * vy_gain));
                        } else {
                            // Balloon behind: back up toward it, light lateral, gentle yaw
                            vx   = std::max(-max_v,   std::min(0.0,      sq(ex_b) * gain));
                            vy   = std::max(-max_vy,  std::min(max_vy,   sq(ey_b) * vy_gain));
                            vyaw = std::max(-max_vyaw, std::min(max_vyaw, angle_err * 0.3));
                        }
                    }

                    // Safety: if dog center or head is >1m outside the marker boundary, stop
                    if (!dog_P_world.empty()) {
                        double dx = dog_P_world.at<double>(0);
                        double dy = dog_P_world.at<double>(1);
                        double hx = dx + 0.22 * std::cos(dog_yaw);
                        double hy = dy + 0.22 * std::sin(dog_yaw);
                        const double margin = 1.0;
                        auto outside = [&](double px, double py) {
                            return px < -margin || px > 1.265 + margin ||
                                   py < -margin || py > 2.415 + margin;
                        };
                        if (outside(dx, dy) || outside(hx, hy)) {
                            vx = 0.0; vy = 0.0; vyaw = 0.0;
                            last_vx = 0.0; last_vy = 0.0; last_vyaw = 0.0;
                            dropout_frames = max_dropout_frames + 1;
                        }
                    }

                    // On dropout, hold the last valid command for up to max_dropout_frames
                    if (vx == 0.0 && vy == 0.0 && vyaw == 0.0) {
                        dropout_frames++;
                        if (dropout_frames <= max_dropout_frames) {
                            vx = last_vx; vy = last_vy; vyaw = last_vyaw;
                        }
                    } else {
                        last_vx = vx; last_vy = vy; last_vyaw = vyaw;
                    }

                    std::ostringstream json;
                    json << std::fixed << std::setprecision(4)
                         << "{\"vx\":" << vx << ",\"vy\":" << vy
                         << ",\"vyaw\":" << vyaw << ",\"timestamp\":" << ts
                         << ",\"tilt\":" << (tilt ? "true" : "false") << "}";
                    redis_set("cmd_vel", json.str());
                }
                // <---- Velocity command

                // ----> Top-down view overlay (top-left corner, drawn directly on frame)
                // Axes: world Y → pixel X (horizontal), world X → pixel Y (vertical)
                // Layout: 1(TL) -- 2(TR)
                //         4(BL) -- 3(BR)
                {
                    const int   pad      = 18;
                    const int   label_h  = 26;
                    const float td_scale = 200.0f; // pixels per metre
                    const float world_W  = 1.265f; // X axis (vertical in panel)
                    const float world_H  = 2.415f; // Y axis (horizontal in panel)
                    int td_w = (int)(world_H * td_scale) + 2 * pad;
                    int td_h = label_h + (int)(world_W * td_scale) + 2 * pad;

                    // 20% opacity black background
                    if (td_w < right_cpu.cols && td_h < right_cpu.rows) {
                        cv::Mat roi = right_cpu(cv::Rect(0, 0, td_w, td_h));
                        cv::Mat black = cv::Mat::zeros(td_h, td_w, CV_8UC3);
                        cv::addWeighted(black, 0.4, roi, 0.6, 0, roi);
                    }

                    cv::putText(right_cpu, "Top Down View",
                                cv::Point(pad, label_h - 5),
                                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(200, 200, 200), 1);

                    // world Y → pixel x, world X → pixel y (offset below title)
                    auto w2td = [&](float wx, float wy) -> cv::Point {
                        return cv::Point((int)(pad + wy * td_scale),
                                         (int)(label_h + pad + wx * td_scale));
                    };

                    // Rectangle outline with corner labels
                    cv::Point mc[4] = {
                        w2td(0.0f,   0.0f),     // marker 1: top-left
                        w2td(0.0f,   world_H),  // marker 2: top-right
                        w2td(world_W, world_H), // marker 3: bottom-right
                        w2td(world_W, 0.0f),    // marker 4: bottom-left
                    };
                    const cv::Scalar rect_col(0, 200, 255);
                    for (int i = 0; i < 4; i++)
                        cv::line(right_cpu, mc[i], mc[(i+1)%4], rect_col, 2, cv::LINE_AA);
                    cv::putText(right_cpu, "1", mc[0] + cv::Point( 4,  14), cv::FONT_HERSHEY_SIMPLEX, 0.5, rect_col, 2);
                    cv::putText(right_cpu, "2", mc[1] + cv::Point(-14, 14), cv::FONT_HERSHEY_SIMPLEX, 0.5, rect_col, 2);
                    cv::putText(right_cpu, "3", mc[2] + cv::Point(-14, -5), cv::FONT_HERSHEY_SIMPLEX, 0.5, rect_col, 2);
                    cv::putText(right_cpu, "4", mc[3] + cv::Point( 4,  -5), cv::FONT_HERSHEY_SIMPLEX, 0.5, rect_col, 2);

                    // Balloon (green dot)
                    if (!balloon_P_world.empty()) {
                        cv::Point bp = w2td((float)balloon_P_world.at<double>(0),
                                            (float)balloon_P_world.at<double>(1));
                        cv::circle(right_cpu, bp, 8, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
                    }

                    // Dog (orange dot + heading arrow) + head point (white dot)
                    // World heading (cos yaw, sin yaw) in (X,Y); maps to pixel (sin yaw, cos yaw)
                    if (!dog_P_world.empty()) {
                        cv::Point dp = w2td((float)dog_P_world.at<double>(0),
                                            (float)dog_P_world.at<double>(1));
                        cv::circle(right_cpu, dp, 8, cv::Scalar(0, 165, 255), -1, cv::LINE_AA);
                        cv::Point tip = dp + cv::Point(
                            (int)(30.0f * std::sin(dog_yaw)),  // world Y component → pixel x
                            (int)(30.0f * std::cos(dog_yaw))); // world X component → pixel y
                        cv::arrowedLine(right_cpu, dp, tip, cv::Scalar(0, 165, 255), 2, cv::LINE_AA, 0, 0.3);

                        // Head point: 22 cm in front of tag
                        float hx = (float)(dog_P_world.at<double>(0) + 0.22 * std::cos(dog_yaw));
                        float hy = (float)(dog_P_world.at<double>(1) + 0.22 * std::sin(dog_yaw));
                        cv::circle(right_cpu, w2td(hx, hy), 5, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);
                    }
                }
                // <---- Top-down view overlay

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

    if (redis_fd >= 0) ::close(redis_fd);

    // ----> Recording teardown
    rec_stop = true;
    zed_rec_th.join();
    if (rs_ok) { rs_pipe.stop(); rs_rec_th.join(); }
    std::cout << "Recording saved to: " << rec_out << std::endl;
    // <---- Recording teardown

    return EXIT_SUCCESS;
}


