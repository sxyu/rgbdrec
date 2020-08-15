/** ORB_SLAM2 RGB-D camera pose estimation wrapper;
 *  (c) Alex Yu 2019-20, Apache License 2.0 */
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <System.h>
#include "Calibration.h"
#include "Util.h"

namespace {
using boost::filesystem::directory_iterator;
using boost::filesystem::path;
using std::string;
}  // namespace

int main(int argc, char** argv) {
    namespace po = boost::program_options;
    string dataPath, orbVocPath, intrinName;
    double bf, camFPS, thDepth, orbScaleFactor;
    int orbNFeatures, orbNLevels, orbIniThFAST, orbMinThFAST;
    bool rerun, useViewer;

    // **** BEGIN ARG PARSING ****
    po::options_description desc("Option arguments");
    po::options_description descPositional(
        "ORB_SLAM2 wrapper for RGB-D camera pose estimation\n(c) Alex Yu "
        "2020\n"
        "\nPositional arguments");
    po::options_description descCombined("");
    desc.add_options()("help", "produce help message")(
        "vocab,V",
        po::value<string>(&orbVocPath)
            ->default_value(ark::util::resolveRootPath("data/ORBvoc.bin")),
        "ORB vocabulary path for ORB_SLAM2")(
        "force-rerun", po::bool_switch(&rerun),
        "force rerun ORB_SLAM2 even if outputs already exist")(
        "intrin", po::value<string>(&intrinName)->default_value("intrin.txt"),
        "intrinsics file to use inside data directory")(
        "bf", po::value<double>(&bf)->default_value(40.0),
        "'IR baseline times fx' need by ORB_SLAM. Probably "
        "ignore if "
        "not applicable")("fps,f",
                          po::value<double>(&camFPS)->default_value(30.0),
                          "camera capture FPS")(
        "thdepth", po::value<double>(&thDepth)->default_value(1e9),
        "ThDepth parameter in ORB_SLAM2 'close/far threshold'. No idea what it "
        "does")("orb-nfeatures,F",
                po::value<int>(&orbNFeatures)->default_value(18000),
                "number of ORB features per image")(
        "orb-scalefactor",
        po::value<double>(&orbScaleFactor)->default_value(1.2),
        "scale factor between levels in scale pyramid of ORB extraction")(
        "orb-nlevels", po::value<int>(&orbNLevels)->default_value(20),
        "number of levels in scale pyramid of ORB extraction")(
        "orb-inithfast", po::value<int>(&orbIniThFAST)->default_value(20),
        "ORB extractor FAST threshold")(
        "orb-minthfast", po::value<int>(&orbMinThFAST)->default_value(7),
        "ORB extractor FAST threshold when no corners detected (should be < "
        "orbIniThFAST)")("use-viewer,v", po::bool_switch(&useViewer),
                         "show ORB_SLAM2 viewer");
    descPositional.add_options()(
        "data-path", po::value<string>(&dataPath)->required(), "Data path");

    descCombined.add(descPositional);
    descCombined.add(desc);
    po::variables_map vm;

    po::positional_options_description posopt;
    posopt.add("data-path", 1);
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

    using boost::filesystem::exists;
    path rootPath = path(dataPath);
    string imagesPath = (rootPath / "rgb").string();
    string depthsPath = (rootPath / "depth_exr").string();
    string timestampsFile = (rootPath / "timestamp.txt").string();
    string intrinFile = (rootPath / intrinName).string();
    // ORB_SLAM2 outputs
    string keyFramesFile = (rootPath / "keyframes.txt").string();
    string trajectoryFile = (rootPath / "trajectory.txt").string();
    string trajectoryKITTIFile = (rootPath / "trajectory_kitti.txt").string();
    // Our outputs
    string transformsFile = (rootPath / "poses.txt").string();
    string intrinOutFile = (rootPath / "intrin_refine.txt").string();

    if (!exists(rootPath) || !exists(imagesPath) || !exists(depthsPath) ||
        !exists(timestampsFile) || !exists(intrinFile)) {
        std::cerr << "Error: Invalid data directory: " << rootPath
                  << ". Please make sure it contains rgb/ subdirectory.\n";
        return 1;
    }

    int imWidth, imHeight;
    // List images and get timestamps
    std::vector<std::string> images = ark::util::listDir(imagesPath);
    std::vector<std::string> depths = ark::util::listDir(depthsPath, ".exr");

    // Timestamps in seconds
    std::vector<double> timestamps;
    timestamps.reserve(images.size());
    {
        std::ifstream timestampsIfs(timestampsFile);
        uint64_t timestampNano, timestampNanoInit;
        while (timestampsIfs) {
            timestampsIfs >> timestampNano;
            if (timestamps.empty()) timestampNanoInit = timestampNano;
            timestamps.push_back((double)(timestampNano - timestampNanoInit) /
                                 1e9);
        }
    }

    if (images.size() != depths.size()) {
        std::cerr << "Error: Found " << images.size() << " color images "
                  << depths.size() << " depth images, and "
                  << " timestamps. These must match.\n";
        return 1;
    }
    {
        if (images.empty()) {
            std::cerr << "No images in " << imagesPath << ", quitting...\n";
            return 1;
        }
        // Load a sample image to grab size
        cv::Mat colorIm = cv::imread(images[0]);
        imWidth = colorIm.cols;
        imHeight = colorIm.rows;
    }

    ark::CameraIntrin intrin;
    intrin.readFile(intrinFile);

    if (!rerun && exists(keyFramesFile) && exists(trajectoryFile) &&
        exists(trajectoryKITTIFile)) {
        std::cout << ">>> Note: not runnning ORB_SLAM2 because "
                     "keyframes.txt/trajectory.txt/trajectory_kitti.txt exist\n"
                     "    add --force-rerun"
                  << std::endl;
    } else {
        std::cout << ">>> ORB_SLAM2 start" << std::endl;
        // **** BEGIN YML GENERATION ****
        string tempYmlPath = (boost::filesystem::temp_directory_path() /
                              boost::filesystem::unique_path())
                                 .string() +
                             "rgbdpose-orbslam2.yml";
        {
            std::cout << ">>> Generating temporary ORB_SLAM2 config: "
                      << tempYmlPath;
            std::ofstream ofs(tempYmlPath);
            ofs << std::setprecision(12);
            // Camera calibration
            ofs << "%YAML:1.0\n# Camera\n";
            ofs << "Camera.fx: " << intrin.fx << "\n";
            ofs << "Camera.fy: " << intrin.fy << "\n";
            ofs << "Camera.cx: " << intrin.cx << "\n";
            ofs << "Camera.cy: " << intrin.cy << "\n";

            ofs << "Camera.k1: " << intrin.k[0] << "\n";
            ofs << "Camera.k2: " << intrin.k[1] << "\n";
            ofs << "Camera.p1: " << intrin.p[0] << "\n";
            ofs << "Camera.p2: " << intrin.p[1] << "\n";
            ofs << "Camera.k3: " << intrin.k[2] << "\n";

            ofs << "Camera.width: " << imWidth << "\n";
            ofs << "Camera.height: " << imHeight << "\n";

            // More camera params
            ofs << "Camera.fps: " << camFPS << "\n";

            // Weird 'baseline times fx' thing that I don't understand
            ofs << "Camera.bf: " << bf << "\n";

            // Use BGR
            ofs << "Camera.RGB: 0\n";

            // ThDepth: some threshold, don't know what it does
            ofs << "Camera.ThDepth: " << thDepth << "\n";

            // No scaling
            ofs << "Camera.DepthMapFactor: 1.0\n";

            ofs << "\n# ORB\n";
            ofs << "ORBextractor.nFeatures: " << orbNFeatures << "\n";
            ofs << "ORBextractor.scaleFactor: " << orbScaleFactor << "\n";
            ofs << "ORBextractor.nLevels: " << orbNLevels << "\n";
            ofs << "ORBextractor.iniThFAST: " << orbIniThFAST << "\n";
            ofs << "ORBextractor.minThFAST: " << orbMinThFAST << "\n";

            if (useViewer) {
                ofs << "\n# Viewer\n";
                ofs << "Viewer.KeyFrameSize: 0.05\n";
                ofs << "Viewer.KeyFrameLineWidth: 1\n";
                ofs << "Viewer.GraphLineWidth: 0.9\n";
                ofs << "Viewer.PointSize: 2\n";
                ofs << "Viewer.CameraSize: 0.08\n";
                ofs << "Viewer.CameraLineWidth: 3\n";
                ofs << "Viewer.ViewpointX: 0\n";
                ofs << "Viewer.ViewpointY: -0.7\n";
                ofs << "Viewer.ViewpointZ: -1.8\n";
                ofs << "Viewer.ViewpointF: 500\n";
            }
        }
        // **** END YML GENERATION ****

        // Create ORB SLAM system
        ORB_SLAM2::System SLAM(orbVocPath, tempYmlPath, ORB_SLAM2::System::RGBD,
                               useViewer);

        for (size_t i = 0; i < images.size(); ++i) {
            cv::Mat depth = cv::imread(
                depths[i], cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
            cv::Mat color = cv::imread(images[i]);
            std::cout << ">>> " << i << "/" << images.size() << " @ "
                      << timestamps[i] << "s:\n    " << images[i] << " "
                      << depths[i] << std::endl;

            SLAM.TrackRGBD(color, depth, timestamps[i]);
        }

        std::cout << ">>> Shutting down ORB_SLAM2" << std::endl;
        SLAM.Shutdown();
        SLAM.SaveKeyFrameTrajectoryTUM(keyFramesFile);
        SLAM.SaveTrajectoryTUM(trajectoryFile);
        SLAM.SaveTrajectoryKITTI(trajectoryKITTIFile);

        if (exists(tempYmlPath)) {
            boost::filesystem::remove(tempYmlPath);
        }
    }

    {
        std::cout << ">>> Parsing ORB_SLAM2 trajectory" << std::endl;
        std::ifstream trajIfs(trajectoryKITTIFile);
        std::ofstream transformsFs(transformsFile);
        double buf[12];
        for (size_t i = 0; i < images.size(); ++i) {
            for (int i = 0; i < 12; ++i) trajIfs >> buf[i];
            // x -y -z -> x y z
            transformsFs << buf[0] << " " << -buf[1] << " " << -buf[2] << " "
                         << buf[3] << " ";
            transformsFs << -buf[4] << " " << buf[5] << " " << buf[6] << " "
                         << -buf[7] << " ";
            transformsFs << -buf[8] << " " << buf[9] << " " << buf[10] << " "
                         << -buf[11] << "\n";
        }
    }
    // Copy to intrin_refine.txt to be compatible with rgbdpose-colmap
    intrin.writeFile(intrinOutFile);

    std::cout << ">>> Finished\n";
}
