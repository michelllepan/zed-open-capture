// realsense_pipeline.cpp — full balloon/dog pipeline using Intel RealSense
// Drop-in replacement for zed_oc_depth_example: same ArUco PnP world coords,
// green balloon detection, velocity controller, top-down overlay, Redis output.
//
// Build (from examples/):
//   g++ -std=c++14 -O2 realsense_pipeline.cpp -o /tmp/realsense_pipeline \
//       $(pkg-config --cflags --libs realsense2) \
//       $(pkg-config --cflags --libs opencv4) \
//       -lopencv_aruco

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cmath>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

int main(int argc, char* argv[])
{
    (void)argc; (void)argv;

    // ---- RealSense pipeline -----------------------------------------------
    rs2::pipeline pipe;
    rs2::config   cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16,  30);
    rs2::pipeline_profile profile = pipe.start(cfg);

    rs2::align align_to_color(RS2_STREAM_COLOR);

    // ---- Intrinsics --------------------------------------------------------
    auto color_profile = profile.get_stream(RS2_STREAM_COLOR)
                                .as<rs2::video_stream_profile>();
    rs2_intrinsics intr = color_profile.get_intrinsics();
    double fx = intr.fx, fy = intr.fy;
    double cx = intr.ppx, cy = intr.ppy;

    cv::Mat dist_coeffs;
    if (intr.model == RS2_DISTORTION_BROWN_CONRADY ||
        intr.model == RS2_DISTORTION_MODIFIED_BROWN_CONRADY) {
        dist_coeffs = (cv::Mat_<double>(1,5)
            << intr.coeffs[0], intr.coeffs[1],
               intr.coeffs[2], intr.coeffs[3], intr.coeffs[4]);
    }

    const cv::Mat K_mat = (cv::Mat_<double>(3,3)
        << fx, 0, cx,
           0, fy, cy,
           0,  0,  1);

    std::cout << "Color " << intr.width << "x" << intr.height
              << "  fx=" << fx << " fy=" << fy
              << " cx=" << cx << " cy=" << cy << "\n";

    // ---- depth_at_pt: median depth in mm over 9x9 window ------------------
    // Updated to point at the current frame's depth each iteration
    rs2::depth_frame* cur_depth = nullptr;

    auto depth_at_pt = [&](int px, int py) -> float {
        if (!cur_depth) return -1.f;
        int W = cur_depth->get_width(), H = cur_depth->get_height();
        const int rad = 4;
        std::vector<float> samp;
        for (int dy = -rad; dy <= rad; dy++)
            for (int dx = -rad; dx <= rad; dx++) {
                int u = std::max(0, std::min(W-1, px+dx));
                int v = std::max(0, std::min(H-1, py+dy));
                float d = cur_depth->get_distance(u, v) * 1000.f; // m → mm
                if (d > 100.f && d < 10000.f) samp.push_back(d);
            }
        if (samp.empty()) return -1.f;
        auto mid = samp.begin() + samp.size() / 2;
        std::nth_element(samp.begin(), mid, samp.end());
        return *mid;
    };

    // ---- ArUco setup -------------------------------------------------------
    cv::Ptr<cv::aruco::Dictionary> aruco_dict =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params =
        cv::aruco::DetectorParameters::create();

    // World points: origin at marker 1, X along 1→4 (1.265m), Y along 1→2 (2.415m)
    const std::vector<cv::Point3f> aruco_obj_pts = {
        {0.0f,   0.0f,   0.0f},
        {0.0f,   2.415f, 0.0f},
        {1.265f, 2.415f, 0.0f},
        {1.265f, 0.0f,   0.0f},
    };

    cv::Mat cached_R, cached_tvec;
    bool pnp_valid = false;
    std::map<int, cv::Point2f> cached_centers;

    // ---- Redis setup -------------------------------------------------------
    int redis_fd = -1;
    {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd >= 0) {
            struct sockaddr_in addr{};
            addr.sin_family      = AF_INET;
            addr.sin_port        = htons(6379);
            addr.sin_addr.s_addr = inet_addr("127.0.0.1");
            if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                redis_fd = fd;
                std::cout << "Connected to Redis on localhost:6379\n";
            } else {
                ::close(fd);
                std::cerr << "Cannot connect to Redis on localhost:6379\n";
            }
        }
    }

    auto redis_set = [&redis_fd](const std::string& key, const std::string& val) {
        if (redis_fd < 0) return;
        std::string cmd = "*3\r\n$3\r\nSET\r\n$" + std::to_string(key.size()) + "\r\n"
            + key + "\r\n$" + std::to_string(val.size()) + "\r\n" + val + "\r\n";
        if (::send(redis_fd, cmd.c_str(), cmd.size(), MSG_NOSIGNAL) < 0) {
            ::close(redis_fd);
            redis_fd = -1;
        } else {
            char buf[32];
            ::recv(redis_fd, buf, sizeof(buf), MSG_DONTWAIT);
        }
    };

    // ---- Velocity controller state ----------------------------------------
    double last_vx = 0.0, last_vy = 0.0, last_vyaw = 0.0;
    int    dropout_frames = 0;
    const  int max_dropout_frames = 5;

    double prev_balloon_z  = -1.0;
    double prev_balloon_ts = -1.0;

    std::cout << "Streaming — press 'q' or Esc to quit.\n";

    while (true)
    {
        rs2::frameset frames = pipe.wait_for_frames(5000);
        frames = align_to_color.process(frames);

        rs2::video_frame color_frame = frames.get_color_frame();
        rs2::depth_frame depth_frame = frames.get_depth_frame();
        if (!color_frame || !depth_frame) continue;

        cur_depth = &depth_frame;  // expose to depth_at_pt lambda

        int W = color_frame.get_width(), H = color_frame.get_height();

        // frame_bgr is a zero-copy view into the RS frame buffer; clone before writing
        cv::Mat frame_bgr(H, W, CV_8UC3,
                          const_cast<void*>(color_frame.get_data()));

        // Undistort into img (also serves as a writable copy)
        cv::Mat img;
        if (dist_coeffs.empty())
            img = frame_bgr.clone();
        else
            cv::undistort(frame_bgr, img, K_mat, dist_coeffs);

        // RealSense timestamps are milliseconds since device epoch → seconds
        double ts = color_frame.get_timestamp() / 1000.0;

        // ---- Per-frame state -----------------------------------------------
        cv::Mat dog_P_world;
        double  dog_yaw = 0.0;
        cv::Mat balloon_P_world;
        bool    balloon_in_bounds = false;

        // ---- Green balloon detection ---------------------------------------
        cv::Point2f balloon_img_pt(-1, -1);
        float balloon_depth_mm = -1;
        {
            cv::Mat hsv, mask;
            cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
            cv::inRange(hsv, cv::Scalar(35, 60, 40), cv::Scalar(85, 255, 255), mask);

            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  kernel);
            cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            double max_area = 0; int best = -1;
            for (int i = 0; i < (int)contours.size(); i++) {
                double a = cv::contourArea(contours[i]);
                if (a > max_area) { max_area = a; best = i; }
            }
            if (best >= 0 && max_area > 200) {
                cv::Rect bbox = cv::boundingRect(contours[best]);
                int bx = bbox.x + bbox.width  / 2;
                int by = bbox.y + bbox.height / 2;
                balloon_img_pt   = cv::Point2f(bx, by);
                balloon_depth_mm = depth_at_pt(bx, by);
                cv::rectangle(img, bbox, cv::Scalar(0, 255, 0), 2);
            }
        }

        // ---- ArUco detection + PnP ----------------------------------------
        {
            std::vector<std::vector<cv::Point2f>> all_corners, rejected;
            std::vector<int> all_ids;
            cv::aruco::detectMarkers(img, aruco_dict, all_corners, all_ids,
                                     aruco_params, rejected);

            int dog_idx = -1;
            std::map<int, cv::Point2f> detected_ground;
            std::vector<std::vector<cv::Point2f>> dog_corners_vec;
            std::vector<int> dog_ids_vec;

            for (size_t i = 0; i < all_ids.size(); i++) {
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

            {
                bool all_found = true;
                for (int id : {1,2,3,4})
                    if (detected_ground.find(id) == detected_ground.end())
                        { all_found = false; break; }
                if (all_found) {
                    for (int id : {1,2,3,4}) cached_centers[id] = detected_ground[id];
                    std::vector<cv::Point2f> img_pts = {
                        cached_centers[1], cached_centers[2],
                        cached_centers[3], cached_centers[4]
                    };
                    cv::Mat rvec, tvec;
                    cv::solvePnP(aruco_obj_pts, img_pts, K_mat, dist_coeffs, rvec, tvec);
                    cv::Rodrigues(rvec, cached_R);
                    cached_tvec = tvec;
                    pnp_valid   = true;
                }
            }

            // Draw cached ground rectangle
            if (!cached_centers.empty()) {
                std::vector<cv::Point> pts;
                for (int id : {1,2,3,4}) pts.push_back(cv::Point(cached_centers[id]));
                cv::Mat overlay = img.clone();
                cv::fillConvexPoly(overlay, pts, cv::Scalar(0, 200, 255));
                cv::addWeighted(overlay, 0.15, img, 0.85, 0, img);
                for (int i = 0; i < 4; i++)
                    cv::line(img, pts[i], pts[(i+1)%4], cv::Scalar(0,200,255), 2, cv::LINE_AA);
                const char* corner_labels[4] = {"1(TL)", "2(TR)", "3(BR)", "4(BL)"};
                for (int i = 0; i < 4; i++)
                    cv::putText(img, corner_labels[i], pts[i] + cv::Point(8,-8),
                                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0,200,255), 2);
            }

            // Dog marker (ID 0)
            cv::aruco::drawDetectedMarkers(img, dog_corners_vec, dog_ids_vec);
            if (dog_idx >= 0) {
                const auto& dc = all_corners[dog_idx];
                cv::Point2f top_mid = (dc[0] + dc[1]) * 0.5f;
                cv::Point2f bot_mid = (dc[3] + dc[2]) * 0.5f;
                cv::Point2f center  = (top_mid + bot_mid) * 0.5f;
                cv::Point2f fwd     = top_mid - bot_mid;

                cv::arrowedLine(img, cv::Point(center), cv::Point(center + fwd),
                                cv::Scalar(0,165,255), 2, cv::LINE_AA, 0, 0.25);

                if (pnp_valid) {
                    float dog_depth_mm = depth_at_pt((int)center.x, (int)center.y);
                    if (dog_depth_mm > 0) {
                        double Z_m = dog_depth_mm / 1000.0;
                        cv::Mat P_cam = (cv::Mat_<double>(3,1)
                            << (center.x - cx) * Z_m / fx,
                               (center.y - cy) * Z_m / fy,
                               Z_m);
                        cv::Mat P_world = cached_R.t() * (P_cam - cached_tvec);

                        cv::Point2f fwd_tip = center + fwd;
                        cv::Mat P_fwd_cam = (cv::Mat_<double>(3,1)
                            << (fwd_tip.x - cx) * Z_m / fx,
                               (fwd_tip.y - cy) * Z_m / fy,
                               Z_m);
                        cv::Mat heading = cached_R.t() * (P_fwd_cam - cached_tvec) - P_world;
                        dog_yaw     = std::atan2(heading.at<double>(1), heading.at<double>(0));
                        dog_P_world = P_world.clone();

                        std::ostringstream lbl;
                        lbl << std::fixed << std::setprecision(2)
                            << "Dog: (" << P_world.at<double>(0) << ", "
                            << P_world.at<double>(1) << ", "
                            << P_world.at<double>(2) << ") m";
                        cv::putText(img, lbl.str(),
                                    cv::Point(center.x+10, center.y-10),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,165,255), 2);
                    }
                }
            }
        }

        // ---- Camera + balloon world position --------------------------------
        if (pnp_valid) {
            cv::Mat cam_world = -cached_R.t() * cached_tvec;
            std::ostringstream ci;
            ci << std::fixed << std::setprecision(2)
               << "Cam: (" << cam_world.at<double>(0) << ", "
               << cam_world.at<double>(1) << ", "
               << cam_world.at<double>(2) << ") m";
            cv::putText(img, ci.str(), cv::Point(10, img.rows - 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,100), 2);

            if (balloon_img_pt.x >= 0 && balloon_depth_mm > 0) {
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
                    << "Balloon: (" << bpx << ", " << bpy << ", "
                    << P_world.at<double>(2) << ") m";
                if (balloon_in_bounds) pos << " [IN]";
                cv::putText(img, pos.str(),
                            cv::Point(balloon_img_pt.x+10, balloon_img_pt.y+20),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7,
                            balloon_in_bounds ? cv::Scalar(0,255,255) : cv::Scalar(0,255,0), 2);
            }
        }

        // ---- Velocity controller -------------------------------------------
        {
            double balloon_vz = 0.0;
            bool tilt = false;
            if (!balloon_P_world.empty()) {
                double cur_z = balloon_P_world.at<double>(2);
                if (prev_balloon_z >= 0.0 && prev_balloon_ts > 0.0) {
                    double dt = ts - prev_balloon_ts;
                    if (dt > 0.0) balloon_vz = (cur_z - prev_balloon_z) / dt;
                }
                bool near_head = false;
                if (!dog_P_world.empty()) {
                    double hx = dog_P_world.at<double>(0) + 0.22 * std::cos(dog_yaw);
                    double hy = dog_P_world.at<double>(1) + 0.22 * std::sin(dog_yaw);
                    double ddx = balloon_P_world.at<double>(0) - hx;
                    double ddy = balloon_P_world.at<double>(1) - hy;
                    near_head = std::sqrt(ddx*ddx + ddy*ddy) <= 0.10;
                }
                tilt = (prev_balloon_z >= 1.5 && cur_z < 1.5 && balloon_vz < 0.0 /* && near_head */);
                prev_balloon_z  = cur_z;
                prev_balloon_ts = ts;
            }

            double vx = 0.0, vy = 0.0, vyaw = 0.0;

            if (balloon_in_bounds && !dog_P_world.empty() && !balloon_P_world.empty()
                && balloon_P_world.at<double>(2) >= 0.2)
            {
                dropout_frames = 0;
                const double gain     = 6.0;
                const double max_v    = 1.0;
                const double vy_gain  = 2.0;
                const double max_vy   = 0.6;
                const double yaw_gain = 1.2;
                const double max_vyaw = 1.0;

                double head_x = dog_P_world.at<double>(0) + 0.22 * std::cos(dog_yaw);
                double head_y = dog_P_world.at<double>(1) + 0.22 * std::sin(dog_yaw);
                double ex = balloon_P_world.at<double>(0) - head_x;
                double ey = balloon_P_world.at<double>(1) - head_y;

                double c = std::cos(dog_yaw), s = std::sin(dog_yaw);
                double ex_b =  ex * c + ey * s;
                double ey_b = -ex * s + ey * c;

                double angle_err = std::atan2(ey_b, ex_b);
                bool behind = std::abs(angle_err) > M_PI_2;

                auto sq = [](double v){ return std::copysign(v*v, v); };

                if (!behind) {
                    vyaw = std::max(-max_vyaw, std::min(max_vyaw, angle_err * yaw_gain));
                    vx   = std::max(0.0,       std::min(max_v,    sq(ex_b) * gain));
                    vy   = std::max(-max_vy,   std::min(max_vy,   sq(ey_b) * vy_gain));
                } else {
                    vx   = std::max(-max_v,    std::min(0.0,      sq(ex_b) * gain));
                    vy   = std::max(-max_vy,   std::min(max_vy,   sq(ey_b) * vy_gain));
                    vyaw = std::max(-max_vyaw,  std::min(max_vyaw, angle_err * 0.3));
                }
            }

            // Safety: stop if dog or head is >1m outside the marker boundary
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

        // ---- Top-down overlay ---------------------------------------------
        // Axes: world Y → pixel X (horizontal), world X → pixel Y (vertical)
        // Layout: 1(TL)--2(TR) / 4(BL)--3(BR)
        {
            const int   pad      = 18;
            const int   label_h  = 26;
            const float td_scale = 200.0f;
            const float world_W  = 1.265f;
            const float world_H  = 2.415f;
            int td_w = (int)(world_H * td_scale) + 2 * pad;
            int td_h = label_h + (int)(world_W * td_scale) + 2 * pad;

            if (td_w < img.cols && td_h < img.rows) {
                cv::Mat roi = img(cv::Rect(0, 0, td_w, td_h));
                cv::Mat black = cv::Mat::zeros(td_h, td_w, CV_8UC3);
                cv::addWeighted(black, 0.4, roi, 0.6, 0, roi);
            }

            cv::putText(img, "Top Down View", cv::Point(pad, label_h - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(200,200,200), 1);

            auto w2td = [&](float wx, float wy) -> cv::Point {
                return cv::Point((int)(pad + wy * td_scale),
                                 (int)(label_h + pad + wx * td_scale));
            };

            cv::Point mc[4] = {
                w2td(0.0f,    0.0f),
                w2td(0.0f,    world_H),
                w2td(world_W, world_H),
                w2td(world_W, 0.0f),
            };
            const cv::Scalar rect_col(0, 200, 255);
            for (int i = 0; i < 4; i++)
                cv::line(img, mc[i], mc[(i+1)%4], rect_col, 2, cv::LINE_AA);
            cv::putText(img, "1", mc[0]+cv::Point( 4, 14), cv::FONT_HERSHEY_SIMPLEX, 0.5, rect_col, 2);
            cv::putText(img, "2", mc[1]+cv::Point(-14,14), cv::FONT_HERSHEY_SIMPLEX, 0.5, rect_col, 2);
            cv::putText(img, "3", mc[2]+cv::Point(-14,-5), cv::FONT_HERSHEY_SIMPLEX, 0.5, rect_col, 2);
            cv::putText(img, "4", mc[3]+cv::Point( 4, -5), cv::FONT_HERSHEY_SIMPLEX, 0.5, rect_col, 2);

            if (!balloon_P_world.empty()) {
                cv::Point bp = w2td((float)balloon_P_world.at<double>(0),
                                    (float)balloon_P_world.at<double>(1));
                cv::circle(img, bp, 8, cv::Scalar(0,255,0), -1, cv::LINE_AA);
            }

            if (!dog_P_world.empty()) {
                cv::Point dp = w2td((float)dog_P_world.at<double>(0),
                                    (float)dog_P_world.at<double>(1));
                cv::circle(img, dp, 8, cv::Scalar(0,165,255), -1, cv::LINE_AA);
                cv::Point tip = dp + cv::Point(
                    (int)(30.0f * std::sin(dog_yaw)),
                    (int)(30.0f * std::cos(dog_yaw)));
                cv::arrowedLine(img, dp, tip, cv::Scalar(0,165,255), 2, cv::LINE_AA, 0, 0.3);

                float hx = (float)(dog_P_world.at<double>(0) + 0.22 * std::cos(dog_yaw));
                float hy = (float)(dog_P_world.at<double>(1) + 0.22 * std::sin(dog_yaw));
                cv::circle(img, w2td(hx, hy), 5, cv::Scalar(255,255,255), -1, cv::LINE_AA);
            }
        }

        // ---- Display -------------------------------------------------------
        const double scale_w = (double)1800 / img.cols;
        const double scale_h = (double)900  / img.rows;
        const double scale   = std::min(1.0, std::min(scale_w, scale_h));
        cv::Mat display;
        cv::resize(img, display, cv::Size(), scale, scale);
        cv::imshow("RealSense Pipeline", display);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) break;
    }

    pipe.stop();
    if (redis_fd >= 0) ::close(redis_fd);
    return 0;
}
