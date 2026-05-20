// record_rgbd.cpp — simultaneous RGB+depth recording from ZED 2 and RealSense
//
// ZED:       left-rectified color (JPEG) + SGBM depth at half res (16-bit PNG, mm)
// RealSense: color (JPEG) + native aligned depth (16-bit PNG, mm)
//
// Output:
//   recording_YYYYMMDD_HHMMSS/
//     zed/color/000000.jpg    zed/depth/000000.png
//     rs/color/000000.jpg     rs/depth/000000.png
//     zed_timestamps.csv      rs_timestamps.csv   (frame, timestamp_s)
//
// NOTE: ZED SGBM at half-res takes ~100-200 ms/frame → expect ~5-10 fps for ZED.
//       RealSense runs independently at up to 30 fps.
//
// Build (from examples/):
//   g++ -std=c++14 -O2 -pthread -DVIDEO_MOD_AVAILABLE record_rgbd.cpp -o /tmp/record_rgbd \
//       -I/home/iliad/zed-open-capture/include -I./include \
//       -L/home/iliad/zed-open-capture/build -lzed_open_capture \
//       -Wl,-rpath,/home/iliad/zed-open-capture/build \
//       $(pkg-config --cflags --libs realsense2) \
//       $(pkg-config --cflags --libs opencv4)

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <cmath>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <cstdlib>

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

#include "videocapture.hpp"
#include "calibration.hpp"
#include "stereo.hpp"

// ---- utilities -------------------------------------------------------------

static void mkdirp(const std::string& p)
{
    int r = system(("mkdir -p \"" + p + "\"").c_str());
    if (r != 0) std::cerr << "Warning: mkdir -p failed for: " << p << "\n";
}

static std::string frame_path(const std::string& dir, int idx, const char* ext)
{
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%06d", idx);
    return dir + "/" + buf + "." + ext;
}

// ---- shared preview buffer (BGR, for display only) -------------------------

struct PreviewBuf {
    std::mutex mtx;
    cv::Mat    img;
};

// ---- ZED record thread -----------------------------------------------------

void zed_record(
    sl_oc::video::VideoCapture& cap,
    const cv::Mat& map_lx, const cv::Mat& map_ly,
    const cv::Mat& map_rx, const cv::Mat& map_ry,
    double fx, double baseline,
    const sl_oc::tools::StereoSgbmPar& par,
    const std::string& color_dir, const std::string& depth_dir,
    const std::string& csv_path,
    std::atomic<bool>& stop,
    PreviewBuf& preview,
    std::atomic<int>& count)
{
    // SGBM matcher
    cv::Ptr<cv::StereoSGBM> matcher =
        cv::StereoSGBM::create(par.minDisparity, par.numDisparities, par.blockSize);
    matcher->setMinDisparity(par.minDisparity);
    matcher->setNumDisparities(par.numDisparities);
    matcher->setBlockSize(par.blockSize);
    matcher->setP1(par.P1);
    matcher->setP2(par.P2);
    matcher->setDisp12MaxDiff(par.disp12MaxDiff);
    matcher->setMode(par.mode);
    matcher->setPreFilterCap(par.preFilterCap);
    matcher->setUniquenessRatio(par.uniquenessRatio);
    matcher->setSpeckleWindowSize(par.speckleWindowSize);
    matcher->setSpeckleRange(par.speckleRange);

    std::ofstream csv(csv_path);
    csv << "frame,timestamp_s\n";

    const std::vector<int> jpg_params = {cv::IMWRITE_JPEG_QUALITY, 90};
    const std::vector<int> png_params = {cv::IMWRITE_PNG_COMPRESSION, 1};

    uint64_t last_ts = 0;
    int idx = 0;

    while (!stop) {
        auto frame = cap.getLastFrame();
        if (frame.data == nullptr || frame.timestamp == last_ts) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        last_ts = frame.timestamp;

        // YUV → BGR → split left/right
        cv::Mat yuv(frame.height, frame.width, CV_8UC2, frame.data);
        cv::Mat bgr;
        cv::cvtColor(yuv.clone(), bgr, cv::COLOR_YUV2BGR_YUYV);
        int hw = bgr.cols / 2;
        cv::Mat left_raw  = bgr(cv::Rect(0,  0, hw, bgr.rows));
        cv::Mat right_raw = bgr(cv::Rect(hw, 0, hw, bgr.rows));

        // Rectify
        cv::Mat left_rect, right_rect;
        cv::remap(left_raw,  left_rect,  map_lx, map_ly, cv::INTER_AREA);
        cv::remap(right_raw, right_rect, map_rx, map_ry, cv::INTER_AREA);

        // Half-size stereo pair for SGBM
        cv::Mat lh, rh;
        cv::resize(left_rect,  lh, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
        cv::resize(right_rect, rh, cv::Size(), 0.5, 0.5, cv::INTER_AREA);

        // SGBM → float disparity scaled to full-res equivalent
        cv::Mat disp_raw, disp_f;
        matcher->compute(lh, rh, disp_raw);
        disp_raw.convertTo(disp_f, CV_32F, 1.0 / 16.0);
        cv::multiply(disp_f, 2.0, disp_f);  // half→full res: disparity × (1/0.5)

        // Convert disparity → depth in mm → 16-bit, 0 = invalid
        cv::Mat depth_half(disp_f.rows, disp_f.cols, CV_16UC1, cv::Scalar(0));
        for (int r = 0; r < disp_f.rows; r++) {
            const float*  dp = disp_f.ptr<float>(r);
            uint16_t*     zp = depth_half.ptr<uint16_t>(r);
            for (int c = 0; c < disp_f.cols; c++) {
                if (dp[c] > par.minDisparity) {
                    float dm = (float)(fx * baseline / dp[c]);
                    if (dm > par.minDepth_mm && dm < par.maxDepth_mm)
                        zp[c] = (uint16_t)dm;
                }
            }
        }

        // Upsample depth to match color resolution (nearest-neighbor keeps mm values intact)
        cv::Mat depth_full;
        cv::resize(depth_half, depth_full,
                   cv::Size(left_rect.cols, left_rect.rows), 0, 0, cv::INTER_NEAREST);

        // Save
        double ts = frame.timestamp / 1e9;
        cv::imwrite(frame_path(color_dir, idx, "jpg"), left_rect, jpg_params);
        cv::imwrite(frame_path(depth_dir, idx, "png"), depth_full, png_params);
        csv << idx << "," << std::fixed << std::setprecision(6) << ts << "\n";

        // Preview: small BGR color image
        {
            cv::Mat small;
            cv::resize(left_rect, small, cv::Size(640, 360));
            std::lock_guard<std::mutex> lk(preview.mtx);
            preview.img = std::move(small);
        }

        count++;
        idx++;
    }
}

// ---- RealSense record thread -----------------------------------------------

void rs_record(
    rs2::pipeline& pipe,
    rs2::align& align_to_color,
    float depth_scale,
    const std::string& color_dir, const std::string& depth_dir,
    const std::string& csv_path,
    std::atomic<bool>& stop,
    PreviewBuf& preview,
    std::atomic<int>& count)
{
    std::ofstream csv(csv_path);
    csv << "frame,timestamp_s\n";

    const std::vector<int> jpg_params = {cv::IMWRITE_JPEG_QUALITY, 90};
    const std::vector<int> png_params = {cv::IMWRITE_PNG_COMPRESSION, 1};

    int idx = 0;

    while (!stop) {
        rs2::frameset frames;
        if (!pipe.try_wait_for_frames(&frames, 100)) continue;
        frames = align_to_color.process(frames);

        rs2::video_frame color_frame = frames.get_color_frame();
        rs2::depth_frame depth_frame = frames.get_depth_frame();
        if (!color_frame || !depth_frame) continue;

        int W = color_frame.get_width(), H = color_frame.get_height();
        double ts = color_frame.get_timestamp() / 1000.0;

        // Color — clone before imwrite (RS buffer is reused across frames)
        cv::Mat color_bgr(H, W, CV_8UC3,
                          const_cast<void*>(color_frame.get_data()));
        cv::Mat color_save = color_bgr.clone();

        // Depth: Z16 units × depth_scale → metres × 1000 → mm → uint16
        cv::Mat depth_raw(H, W, CV_16UC1,
                          const_cast<void*>(depth_frame.get_data()));
        cv::Mat depth_mm_f;
        depth_raw.convertTo(depth_mm_f, CV_32F);
        cv::multiply(depth_mm_f, depth_scale * 1000.f, depth_mm_f);
        cv::Mat depth_mm;
        depth_mm_f.convertTo(depth_mm, CV_16UC1);  // saturate_cast handles overflow

        // Save
        cv::imwrite(frame_path(color_dir, idx, "jpg"), color_save, jpg_params);
        cv::imwrite(frame_path(depth_dir, idx, "png"), depth_mm, png_params);
        csv << idx << "," << std::fixed << std::setprecision(6) << ts << "\n";

        // Preview
        {
            cv::Mat small;
            cv::resize(color_save, small, cv::Size(640, 360));
            std::lock_guard<std::mutex> lk(preview.mtx);
            preview.img = std::move(small);
        }

        count++;
        idx++;
    }
}

// ---- main ------------------------------------------------------------------

int main(int argc, char* argv[])
{
    (void)argc; (void)argv;

    // ---- Timestamped output directory --------------------------------------
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char ts_buf[32];
    std::strftime(ts_buf, sizeof(ts_buf), "%Y%m%d_%H%M%S", std::localtime(&t));
    std::string out = std::string("recording_") + ts_buf;

    for (auto& d : {out + "/zed/color", out + "/zed/depth",
                    out + "/rs/color",  out + "/rs/depth"})
        mkdirp(d);

    std::cout << "Recording to: " << out << "\n";

    // ---- ZED setup ---------------------------------------------------------
    sl_oc::video::VideoParams params;
    params.res     = sl_oc::video::RESOLUTION::HD1080;
    params.fps     = sl_oc::video::FPS::FPS_30;
    params.verbose = sl_oc::VERBOSITY::INFO;

    sl_oc::video::VideoCapture cap(params);
    if (!cap.initializeVideo(-1)) {
        std::cerr << "Failed to open ZED camera\n";
        return 1;
    }
    std::cout << "ZED SN: " << cap.getSerialNumber() << "\n";
    cap.setAutoWhiteBalance(true);

    std::string cal_file;
    if (!sl_oc::tools::downloadCalibrationFile(cap.getSerialNumber(), cal_file)) {
        std::cerr << "Failed to download ZED calibration\n";
        return 1;
    }

    int zed_w, zed_h;
    cap.getFrameSize(zed_w, zed_h);

    cv::Mat map_lx, map_ly, map_rx, map_ry, K_left, K_right;
    double baseline = 0;
    sl_oc::tools::initCalibration(cal_file, cv::Size(zed_w / 2, zed_h),
                                  map_lx, map_ly, map_rx, map_ry,
                                  K_left, K_right, &baseline);
    double fx = K_left.at<double>(0, 0);

    sl_oc::tools::StereoSgbmPar sgbm_par;
    if (!sgbm_par.load()) sgbm_par.save();
    sgbm_par.print();

    // ---- RealSense setup ---------------------------------------------------
    rs2::pipeline pipe;
    rs2::config   cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16,  30);
    rs2::pipeline_profile profile = pipe.start(cfg);
    rs2::align align_to_color(RS2_STREAM_COLOR);

    float depth_scale = 0.001f;
    for (auto s : profile.get_device().query_sensors())
        if (auto ds = s.as<rs2::depth_sensor>())
            { depth_scale = ds.get_depth_scale(); break; }
    std::cout << "RS depth scale: " << depth_scale << " m/unit\n";

    // ---- Launch record threads ---------------------------------------------
    std::atomic<bool> stop{false};
    std::atomic<int>  zed_count{0}, rs_count{0};
    PreviewBuf zed_preview, rs_preview;

    std::thread zed_th(zed_record,
        std::ref(cap),
        std::cref(map_lx), std::cref(map_ly),
        std::cref(map_rx), std::cref(map_ry),
        fx, baseline, std::cref(sgbm_par),
        out + "/zed/color", out + "/zed/depth",
        out + "/zed_timestamps.csv",
        std::ref(stop), std::ref(zed_preview), std::ref(zed_count));

    std::thread rs_th(rs_record,
        std::ref(pipe), std::ref(align_to_color), depth_scale,
        out + "/rs/color", out + "/rs/depth",
        out + "/rs_timestamps.csv",
        std::ref(stop), std::ref(rs_preview), std::ref(rs_count));

    // ---- Display loop: side-by-side preview, 'q' to stop -------------------
    std::cout << "Recording — press 'q' or Esc to stop.\n";
    auto t_start = std::chrono::steady_clock::now();

    while (true) {
        cv::Mat left_img, right_img;
        { std::lock_guard<std::mutex> lk(zed_preview.mtx);  left_img  = zed_preview.img.clone(); }
        { std::lock_guard<std::mutex> lk(rs_preview.mtx);   right_img = rs_preview.img.clone();  }

        if (!left_img.empty() && !right_img.empty()) {
            // Match heights
            if (right_img.rows != left_img.rows)
                cv::resize(right_img, right_img,
                           cv::Size(right_img.cols * left_img.rows / right_img.rows,
                                    left_img.rows));
            cv::Mat combined;
            cv::hconcat(left_img, right_img, combined);

            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_start).count();

            char status[128];
            std::snprintf(status, sizeof(status),
                "ZED %d frm | RS %d frm | %.0f s  [q=quit]",
                zed_count.load(), rs_count.load(), elapsed);
            cv::putText(combined, status, cv::Point(10, 28),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

            cv::putText(combined, "ZED",          cv::Point(10,           combined.rows - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200,200,200), 1);
            cv::putText(combined, "RealSense",    cv::Point(left_img.cols + 10, combined.rows - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200,200,200), 1);

            cv::imshow("Recording", combined);
        }

        int key = cv::waitKey(30);
        if (key == 'q' || key == 'Q' || key == 27) break;
    }

    std::cout << "Stopping...\n";
    stop = true;
    zed_th.join();
    rs_th.join();

    std::cout << "\nRecording complete.\n"
              << "  ZED:       " << zed_count << " frames\n"
              << "  RealSense: " << rs_count  << " frames\n"
              << "  Output:    " << out << "\n";
    return 0;
}
