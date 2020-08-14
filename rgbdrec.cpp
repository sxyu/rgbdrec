/** Depth camera RGB-D data recording program, supports KinectV2/Azure Kinect
 *  (c) Alex Yu 2019-20, Apache License 2.0
 *  Partly adapted from OpenARK https://github.com/augcog/OpenARK */
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "Version.h"

#ifdef OPENARK_AZURE_KINECT_ENABLED
#include "AzureKinectCamera.h"
#endif
#ifdef OPENARK_FREENECT2_ENABLED
#include "Freenect2Camera.h"
#endif

#include "Util.h"

#include "opencv2/imgcodecs.hpp"

using namespace ark;

int main(int argc, char** argv) {
    namespace po = boost::program_options;

    // **** BEGIN ARG PARSING ****
    std::string outPath;
    int max_rows;
    bool forceK4a = false, forceFreenect2 = false, forceRS2 = false,
         unflip = false, manual_expo = false, useKDE = false;
    float integr_time, analog_gain, max_fps;

    po::options_description desc("Option arguments");
    po::options_description descPositional(
        "RGB-D Data Recording Tool (c) Alex Yu 2019-20\n\n> OUTPUT FORMAT\n"
        "> rgb/*.jpg "
        "color images\n"
        "> depth_exr/*.exr depth images aligned to color, read with\n"
        "    cv2.imread(PATH_TO_EXR_FILE,  cv2.IMREAD_ANYCOLOR | "
        "cv2.IMREAD_ANYDEPTH)"
        "\n> intrin.txt Camera intrinsics, 4 rows <name> <value>\n> "
        "timestamp.txt "
        "Timestamps in "
        "nanoseconds (1e-9)\n\nPositional arguments");
    po::options_description descCombined("");
    desc.add_options()("help", "produce help message")(
        "maxrows", po::value<int>(&max_rows)->default_value(500),
        "Maximum height of camera output visualizer window")(
        "fps,f", po::value<float>(&max_fps)->default_value(10000.0),
        "Capture FPS cap")
#ifdef OPENARK_AZURE_KINECT_ENABLED
        ("k4a", po::bool_switch(&forceK4a),
         "if set, forces Kinect Azure (k4a) depth camera")
#endif
#ifdef OPENARK_FREENECT2_ENABLED
            ("freenect2", po::bool_switch(&forceFreenect2),
             "if set, forces Freenect2 depth camera")(
                "use-kde", po::bool_switch(&useKDE),
                "if set, Freenect2 depth camera uses KDE algorithm (Lawin et "
                "al. ECCV16)")
#endif
#if defined(OPENARK2_RSSDK2_ENABLED)
                ("rs2", po::bool_switch(&forceRS2),
                 "if set, forces librealsense2 depth cameras")
#endif
                    ("unflip,u", po::bool_switch(&unflip),
                     "if set, un-mirrors camera images along x-axis")(
                        "manual,m", po::bool_switch(&manual_expo),
                        "if set, uses manual exposure if available; please set "
                        "--integr_time/-e and --analog_gain/-g")(
                        "integr-time,e",
                        po::value<float>(&integr_time)->default_value(10.f),
                        "manual expo integration time (0.0, 66.0]")(
                        "analog-gain,g",
                        po::value<float>(&analog_gain)->default_value(2.f),
                        "manual expo analog gain [1.0, 4.0]");

    descPositional.add_options()("output-path",
                                 po::value<std::string>(&outPath)->required(),
                                 "Output Path");
    descCombined.add(descPositional);
    descCombined.add(desc);
    po::variables_map vm;

    po::positional_options_description posopt;
    posopt.add("output-path", 1);
    try {
        po::store(po::command_line_parser(argc, argv)
                      .options(descCombined)
                      .positional(posopt)
                      .run(),
                  vm);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << descPositional << "\n" << desc << "\n";
        return 1;
    }

    if (vm.count("help")) {
        std::cout << descPositional << "\n" << desc << "\n";
        return 0;
    }

    try {
        po::notify(vm);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << descPositional << "\n" << desc << "\n";
        return 1;
    }
    // **** END ARG PARSING ****

    if ((int)forceK4a + (int)forceFreenect2 > 1) {
        std::cerr << "Only one camera preference may be provided";
        return 1;
    }

    printf(
        "CONTROLS:\nQ or ESC to stop recording,\nSPACE to start/pause"
        "(warning: if pausing in the middle, may mess up timestamps)\n\n");

    // seed the rng
    srand(time(NULL));

    const int min_time_between_frames = int(1e9 / max_fps);

    using boost::filesystem::path;
    const path directory_path(outPath);
    const path image_path = directory_path / "rgb";
    const path depth_path = directory_path / "depth_exr";

    path timestamp_path = directory_path / "timestamp.txt";
    path intrin_path = directory_path / "intrin.txt";
    cv::Mat lastXYZMap;
    // initialize the camera
    DepthCamera::Ptr camera;

    if (forceK4a) {
#ifdef OPENARK_AZURE_KINECT_ENABLED
        camera = std::make_shared<AzureKinectCamera>();
#endif
    } else if (forceFreenect2) {
#ifdef OPENARK_FREENECT2_ENABLED
        camera = std::make_shared<Freenect2Camera>("", useKDE);
#endif
    } else {
        camera = std::make_shared<OPENARK_PREFERRED_CAMERA>();
    }
    camera->config.unflip = unflip;
    camera->config.auto_exposure = !manual_expo;
    camera->config.integr_time = integr_time;
    camera->config.analog_gain = analog_gain;

    std::cerr << "Starting data recording, saving to: "
              << directory_path.string() << "\n";
    auto capture_start_time = std::chrono::high_resolution_clock::now();

    // turn on the camera
    camera->beginCapture();

    // If failed to opened camera
    if (!camera->isCapturing()) {
        std::cerr << "Failed to open camera, quitting...\n";
        return 1;
    }
    // Read in camera input and save it to the buffer
    std::vector<uint64_t> timestamps;

    // Pausing feature
    bool pause = true;
    std::cerr << "Note: paused, press space to begin recording.\n";
    std::ofstream timestamp_ofs;

    int currFrame = 0;  // current frame number (since launch/last pause)
    while (true) {
        // get latest image from the camera
        cv::Mat xyzMap = camera->getXYZMap();
        cv::Mat rgbMap = camera->getRGBMap();

        if (!xyzMap.empty() && !rgbMap.empty()) {
            if (pause) {
                const cv::Scalar RECT_COLOR = cv::Scalar(0, 160, 255);
                const std::string NO_SIGNAL_STR = "PAUSED";
                const cv::Point STR_POS(rgbMap.cols / 2 - 50,
                                        rgbMap.rows / 2 + 7);
                const int RECT_WID = 120, RECT_HI = 40;
                cv::Rect rect(rgbMap.cols / 2 - RECT_WID / 2,
                              rgbMap.rows / 2 - RECT_HI / 2, RECT_WID, RECT_HI);

                // show 'paused' and do not record
                cv::rectangle(rgbMap, rect, RECT_COLOR, -1);
                cv::putText(rgbMap, NO_SIGNAL_STR, STR_POS, 0, 0.8,
                            cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
                // cv::rectangle(xyzMap, rect, RECT_COLOR / 255.0, -1);
                // cv::putText(xyzMap, NO_SIGNAL_STR, STR_POS, 0, 0.8,
                // cv::Scalar(1.0f, 1.0f, 1.0f), 1, cv::LINE_AA);
            } else {
                if (currFrame == 0) {
                    if (!boost::filesystem::exists(directory_path)) {
                        boost::filesystem::create_directories(directory_path);
                    }
                    if (!boost::filesystem::exists(image_path)) {
                        boost::filesystem::create_directories(image_path);
                    }
                    if (!boost::filesystem::exists(depth_path)) {
                        boost::filesystem::create_directories(depth_path);
                    }
                    timestamp_ofs.open(timestamp_path.string());
                }
                ++currFrame;
                // store images
                auto ts = camera->getTimestamp();
                if (timestamps.size() &&
                    ts - timestamps.back() < min_time_between_frames)
                    continue;

                int img_index = timestamps.size();
                std::stringstream ss_img_id;
                ss_img_id << std::setw(4) << std::setfill('0')
                          << std::to_string(img_index);
                const std::string depth_img_path =
                    (depth_path / (ss_img_id.str() + ".exr")).string();
                const std::string rgb_img_path =
                    (image_path / (ss_img_id.str() + ".jpg")).string();
                cv::Mat depth;
                cv::extractChannel(xyzMap, depth, 2);

                cv::imwrite(depth_img_path, depth);
                cv::imwrite(rgb_img_path, rgbMap);
                timestamp_ofs << ts << "\n";  // write timestamp

                timestamps.push_back(ts);
            }
            // visualize
            cv::Mat visual, rgbMapFloat;
            rgbMap.convertTo(rgbMapFloat, CV_32FC3, 1. / 255.);
            cv::hconcat(xyzMap, rgbMapFloat, visual);
            if (visual.rows > max_rows) {
                cv::resize(
                    visual, visual,
                    cv::Size(max_rows * visual.cols / visual.rows, max_rows));
            }
            cv::imshow(camera->getModelName() + " XYZ/RGB Maps", visual);
        }

        int c = cv::waitKey(1);

        // make case insensitive (convert to upper)
        if (c >= 'a' && c <= 'z') c &= 0xdf;

        // 27 is ESC
        if (c == 'Q' || c == 27) {
            break;
        } else if (c == ' ') {
            pause = !pause;
        }
    }
    camera->endCapture();
    cv::destroyWindow(camera->getModelName() + " XYZ/RGB Maps");
    std::cout << "Quitting" << std::endl;

    if (currFrame) {
        CameraIntrin intrin;
        // Fit intrinsics from an XYZ map
        // intrin._setVec4d(util::getCameraIntrinFromXYZ(lastXYZMap));
        // Get intrinsics from camera
        intrin = camera->getIntrinsics();

        // Write intrinsics
        intrin.writeFile(intrin_path.string());
        std::cout << "Wrote intrinsics" << std::endl;
    } else {
        std::cout << "Note: no frames recorded" << std::endl;
    }
}
