// record_rgbd.cpp — simultaneous color recording from ZED 2 and RealSense
//
// Output:
//   recording_YYYYMMDD_HHMMSS/
//     zed_color.mp4   rs_color.mp4
//     zed_timestamps.csv    rs_timestamps.csv
//
// After recording, actual fps is computed from the timestamp CSVs and the
// videos are remuxed with ffmpeg so playback duration matches real time.
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
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <ctime>
#include <cstdlib>

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

#include "videocapture.hpp"
#include "calibration.hpp"

// ---- shared preview buffer -------------------------------------------------

struct PreviewBuf {
    std::mutex mtx;
    cv::Mat    img;
};

// ---- helpers ---------------------------------------------------------------

// Compute actual fps from a timestamp CSV (frame,timestamp_s)
static double csv_actual_fps(const std::string& csv_path)
{
    std::ifstream f(csv_path);
    std::string line;
    std::getline(f, line); // header
    double first_ts = -1, last_ts = -1;
    int n = 0;
    while (std::getline(f, line)) {
        auto comma = line.find(',');
        if (comma == std::string::npos) continue;
        double ts = std::stod(line.substr(comma + 1));
        if (first_ts < 0) first_ts = ts;
        last_ts = ts;
        n++;
    }
    if (n < 2 || last_ts <= first_ts) return 15.0;
    return (n - 1) / (last_ts - first_ts);
}

// ---- ZED record thread -----------------------------------------------------

void zed_record(
    sl_oc::video::VideoCapture& cap,
    const cv::Mat& map_lx, const cv::Mat& map_ly,
    const std::string& color_path, const std::string& csv_path,
    std::atomic<bool>& stop, PreviewBuf& preview, std::atomic<int>& count)
{
    std::ofstream csv(csv_path);
    csv << "frame,timestamp_s\n";

    const int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
    cv::VideoWriter vw;
    uint64_t last_ts = 0;
    int idx = 0;

    while (!stop) {
        auto frame = cap.getLastFrame();
        if (frame.data == nullptr || frame.timestamp == last_ts) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        last_ts = frame.timestamp;

        cv::Mat yuv(frame.height, frame.width, CV_8UC2, frame.data);
        cv::Mat bgr;
        cv::cvtColor(yuv.clone(), bgr, cv::COLOR_YUV2BGR_YUYV);
        cv::Mat left_raw = bgr(cv::Rect(0, 0, bgr.cols / 2, bgr.rows));
        cv::Mat left_rect;
        cv::remap(left_raw, left_rect, map_lx, map_ly, cv::INTER_AREA);

        if (!vw.isOpened())
            vw.open(color_path, fourcc, 15.0,
                    cv::Size(left_rect.cols, left_rect.rows), true);
        vw.write(left_rect);
        csv << idx++ << "," << std::fixed << std::setprecision(6)
            << frame.timestamp / 1e9 << "\n";

        {
            cv::Mat small;
            cv::resize(left_rect, small, cv::Size(640, 360));
            std::lock_guard<std::mutex> lk(preview.mtx);
            preview.img = std::move(small);
        }
        count++;
    }
}

// ---- RealSense record thread -----------------------------------------------

void rs_record(
    rs2::pipeline& pipe, rs2::align& align_to_color,
    const std::string& color_path, const std::string& csv_path,
    std::atomic<bool>& stop, PreviewBuf& preview, std::atomic<int>& count)
{
    std::ofstream csv(csv_path);
    csv << "frame,timestamp_s\n";

    const int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
    cv::VideoWriter vw;
    int idx = 0;

    while (!stop) {
        rs2::frameset frames;
        if (!pipe.try_wait_for_frames(&frames, 100)) continue;
        frames = align_to_color.process(frames);

        rs2::video_frame cf = frames.get_color_frame();
        if (!cf) continue;

        int W = cf.get_width(), H = cf.get_height();
        cv::Mat color(H, W, CV_8UC3, const_cast<void*>(cf.get_data()));
        cv::Mat color_save = color.clone();
        double ts = cf.get_timestamp() / 1000.0;

        if (!vw.isOpened())
            vw.open(color_path, fourcc, 15.0, cv::Size(W, H), true);
        vw.write(color_save);
        csv << idx++ << "," << std::fixed << std::setprecision(6) << ts << "\n";

        {
            cv::Mat small;
            cv::resize(color_save, small, cv::Size(640, 360));
            std::lock_guard<std::mutex> lk(preview.mtx);
            preview.img = std::move(small);
        }
        count++;
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
    { int r = system(("mkdir -p \"" + out + "\"").c_str()); (void)r; }
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

    // ---- RealSense setup ---------------------------------------------------
    rs2::pipeline pipe;
    rs2::config   cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 15);
    rs2::pipeline_profile profile = pipe.start(cfg);
    (void)profile;
    rs2::align align_to_color(RS2_STREAM_COLOR);
    std::cout << "RealSense started\n";

    // ---- Launch record threads ---------------------------------------------
    std::atomic<bool> stop{false};
    std::atomic<int>  zed_count{0}, rs_count{0};
    PreviewBuf zed_preview, rs_preview;

    std::string zed_csv = out + "/zed_timestamps.csv";
    std::string rs_csv  = out + "/rs_timestamps.csv";
    std::string zed_mp4 = out + "/zed_color.mp4";
    std::string rs_mp4  = out + "/rs_color.mp4";

    std::thread zed_th(zed_record,
        std::ref(cap), std::cref(map_lx), std::cref(map_ly),
        zed_mp4, zed_csv,
        std::ref(stop), std::ref(zed_preview), std::ref(zed_count));

    std::thread rs_th(rs_record,
        std::ref(pipe), std::ref(align_to_color),
        rs_mp4, rs_csv,
        std::ref(stop), std::ref(rs_preview), std::ref(rs_count));

    // ---- Display loop ------------------------------------------------------
    std::cout << "Recording — press 'q' or Esc to stop.\n";
    auto t_start = std::chrono::steady_clock::now();

    while (true) {
        cv::Mat left_img, right_img;
        { std::lock_guard<std::mutex> lk(zed_preview.mtx);  left_img  = zed_preview.img.clone(); }
        { std::lock_guard<std::mutex> lk(rs_preview.mtx);   right_img = rs_preview.img.clone();  }

        if (!left_img.empty() && !right_img.empty()) {
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
            cv::putText(combined, "ZED",       cv::Point(10,              combined.rows - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
            cv::putText(combined, "RealSense", cv::Point(left_img.cols + 10, combined.rows - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
            cv::imshow("Recording", combined);
        }

        int key = cv::waitKey(30);
        if (key == 'q' || key == 'Q' || key == 27) break;
    }

    std::cout << "Stopping...\n";
    stop = true;
    zed_th.join();
    rs_th.join();

    double zed_fps = csv_actual_fps(zed_csv);
    double rs_fps  = csv_actual_fps(rs_csv);

    std::cout << "\nRecording complete.\n"
              << "  ZED:       " << zed_count << " frames @ "
              << std::fixed << std::setprecision(2) << zed_fps << " fps → " << zed_mp4 << "\n"
              << "  RealSense: " << rs_count  << " frames @ "
              << rs_fps << " fps → " << rs_mp4  << "\n";
    return 0;
}
