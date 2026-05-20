// Standalone RealSense test: streams color + depth, shows both in OpenCV windows.
// Build: g++ -std=c++14 -O2 realsense_test.cpp -o realsense_test \
//            $(pkg-config --cflags --libs realsense2) \
//            $(pkg-config --cflags --libs opencv4)

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>

int main()
{
    // ---- enumerate devices ------------------------------------------------
    rs2::context ctx;
    auto devices = ctx.query_devices();
    if (devices.size() == 0) {
        std::cerr << "No RealSense device found.\n";
        return 1;
    }
    std::cout << "Found " << devices.size() << " RealSense device(s):\n";
    for (auto d : devices)
        std::cout << "  " << d.get_info(RS2_CAMERA_INFO_NAME)
                  << "  SN=" << d.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << "\n";

    // ---- configure pipeline -----------------------------------------------
    rs2::pipeline pipe;
    rs2::config   cfg;

    // 640×480 @ 30 fps — works on all D4xx devices
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16,  30);

    rs2::pipeline_profile profile = pipe.start(cfg);

    // ---- depth scale (device units → metres) ------------------------------
    float depth_scale = 0.001f;
    for (auto& s : profile.get_device().query_sensors()) {
        if (auto ds = s.as<rs2::depth_sensor>()) {
            depth_scale = ds.get_depth_scale();
            break;
        }
    }
    std::cout << "Depth scale: " << depth_scale << " m/unit\n";

    // ---- align depth to colour --------------------------------------------
    rs2::align align_to_color(RS2_STREAM_COLOR);

    // ---- coloriser for depth visualisation --------------------------------
    rs2::colorizer colorizer;
    colorizer.set_option(RS2_OPTION_COLOR_SCHEME, 2); // white-to-black

    std::cout << "Streaming — press 'q' or Esc to quit.\n";

    while (true) {
        rs2::frameset frames = pipe.wait_for_frames(5000);  // 5 s timeout
        frames = align_to_color.process(frames);

        rs2::video_frame color_frame = frames.get_color_frame();
        rs2::depth_frame depth_frame = frames.get_depth_frame();

        if (!color_frame || !depth_frame) continue;

        int W = color_frame.get_width();
        int H = color_frame.get_height();

        // Wrap in cv::Mat — no copy, data owned by frame
        cv::Mat color_mat(H, W, CV_8UC3,
                          const_cast<void*>(color_frame.get_data()));

        // Colourised depth for display
        rs2::frame depth_colored = colorizer.colorize(depth_frame);
        cv::Mat depth_mat(H, W, CV_8UC3,
                          const_cast<void*>(depth_colored.get_data()));

        // Print depth at centre pixel
        float cx = depth_frame.get_distance(W / 2, H / 2);
        std::cout << "\r  Centre depth: " << std::fixed << std::setprecision(3)
                  << cx << " m    " << std::flush;

        // Overlay centre crosshair + depth text on colour image
        cv::Mat display = color_mat.clone();
        cv::drawMarker(display, {W/2, H/2}, {0,255,0},
                       cv::MARKER_CROSS, 20, 2, cv::LINE_AA);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.3f m", cx);
        cv::putText(display, buf, {W/2 + 12, H/2 - 6},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0}, 2, cv::LINE_AA);

        cv::imshow("RealSense Color", display);
        cv::imshow("RealSense Depth", depth_mat);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }

    pipe.stop();
    std::cout << "\nDone.\n";
    return 0;
}
