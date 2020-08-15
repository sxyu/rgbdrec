/** COLMAP RGB-D camera pose estimation wrapper;
 *  expects 'colmap' executable (or --colmap value) to be in PATH
 *  (c) Alex Yu 2019-20, Apache License 2.0 */
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <algorithm>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/process.hpp>

#include <opencv2/imgcodecs.hpp>

#include <Eigen/Geometry>

#include "Calibration.h"
#include "Util.h"

namespace {
using boost::filesystem::path;
using std::string;

struct CameraModel {
    std::string name;
    int numParams;
} camModels[] = {{"SIMPLE_PINHOLE", 3},
                 {"PINHOLE", 4},
                 {"SIMPLE_RADIAL", 4},
                 {"RADIAL", 5},
                 {"OPENCV", 8},
                 {"OPENCV_FISHEYE", 8},
                 {"FULL_OPENCV", 12},
                 {"FOV", 5},
                 {"SIMPLE_RADIAL_FISHEYE", 4},
                 {"RADIAL_FISHEYE", 5},
                 {"THIN_PRISM_FISHEYE", 12}};

int readCamerasBin(const std::string& camerasFile, ark::CameraIntrin& intrin) {
    std::ifstream camerasFs(camerasFile,
                            std::ifstream::in | std::ifstream::binary);
    size_t numCams;
    using ark::util::readBin;
    readBin(camerasFs, numCams);
    if (numCams != 1) {
        std::cerr << "Error: Expected 1 camera to be estimated by COLMAP, got "
                  << numCams << "\n";
        return 1;
    }
    int camId, modelId;
    size_t width, height;
    readBin(camerasFs, camId);
    readBin(camerasFs, modelId);
    readBin(camerasFs, width);
    readBin(camerasFs, height);
    const auto& camModel = camModels[modelId];
    int numCamParams = camModel.numParams;

    std::vector<double> params(numCamParams);
    camerasFs.read(reinterpret_cast<char*>(params.data()),
                   params.size() * sizeof(double));

    std::cout << ">>> Camera model: " << camModel.name << "\n";
    if (camModel.name == "SIMPLE_PINHOLE") {
        intrin.fx = intrin.fy = params[0];
        intrin.cx = params[1];
        intrin.cy = params[2];
    } else if (camModel.name == "PINHOLE") {
        intrin.fx = params[0];
        intrin.fy = params[1];
        intrin.cx = params[2];
        intrin.cy = params[3];
    } else if (camModel.name == "SIMPLE_RADIAL") {
        intrin.fx = intrin.fy = params[0];
        intrin.cx = params[1];
        intrin.cy = params[2];
        intrin.k[0] = params[3];
    } else if (camModel.name == "RADIAL") {
        intrin.fx = intrin.fy = params[0];
        intrin.cx = params[1];
        intrin.cy = params[2];
        intrin.k[0] = params[3];
        intrin.k[1] = params[4];
    } else {
        std::cerr << "Error: Unsupported camera model " << camModel.name
                  << " from COLMAP output\n";
        return 1;
    }

    return 0;
}

int readPoints3dBin(const std::string& points3DFile, size_t& numPoints,
                    Eigen::Matrix<double, 3, Eigen::Dynamic>& P,
                    std::map<size_t, size_t>& pointIdMap) {
    std::ifstream points3dFs(points3DFile,
                             std::ifstream::in | std::ifstream::binary);
    using ark::util::readBin;
    readBin(points3dFs, numPoints);
    P.resize(3, numPoints);
    std::cout << ">>> " << numPoints << " total 3D matched points\n";
    for (size_t i = 0; i < numPoints; ++i) {
        size_t pointId, trackLen;
        readBin(points3dFs, pointId);
        pointIdMap[pointId] = i;
        points3dFs.read(reinterpret_cast<char*>(P.col(i).data()),
                        sizeof(double) * 3);

        char rgb[3];
        points3dFs.read(rgb, 3);
        double error;
        readBin(points3dFs, error);

        // // Ignore color and error
        // points3dFs.seekg(7, points3dFs.cur);

        readBin(points3dFs, trackLen);
        // Ignore track
        points3dFs.seekg(8 * trackLen, points3dFs.cur);
    }
    return 0;
}

int readImageBin(const std::string& imagesFile, const std::string& depthPath,
                 const ark::CameraIntrin& intrin, size_t expectedNumImages,
                 Eigen::Matrix<double, 3, Eigen::Dynamic>& P,
                 std::map<size_t, size_t>& pointIdMap, size_t& numImages,
                 Eigen::Matrix<double, 12, Eigen::Dynamic>& T,
                 std::vector<std::pair<std::string, size_t>>& sor) {
    std::ifstream imagesFs(imagesFile,
                           std::ifstream::in | std::ifstream::binary);
    using ark::util::readBin;
    readBin(imagesFs, numImages);
    if (numImages != expectedNumImages) {
        std::cout << "Error: expected " << expectedNumImages << " but only "
                  << numImages << " images found in COLMAP output\n";
        return 1;
    }
    std::cout << ">>> " << numImages << " total images to process\n";
    T.resize(12, numImages);
    sor.resize(numImages);

    std::vector<double> scales;
    for (size_t i = 0; i < numImages; ++i) {
        if (!imagesFs) {
            std::cerr << "Error: Unexpected EOF\n";
            return 1;
        }
        int imageId, cameraId;
        Eigen::Quaterniond q;
        Eigen::Vector3d t;
        Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> trans(
            T.col(i).data());

        readBin(imagesFs, imageId);
        readBin(imagesFs, q.w());
        imagesFs.read(reinterpret_cast<char*>(q.vec().data()),
                      sizeof(double) * 3);
        imagesFs.read(reinterpret_cast<char*>(t.data()), sizeof(double) * 3);
        readBin(imagesFs, cameraId);

        q.normalize();
        trans.leftCols<3>().noalias() = q.toRotationMatrix();
        trans.rightCols<1>().noalias() = t;

        string imgName;
        char c;
        while (true) {
            imagesFs.read(&c, 1);
            if (!c) break;
            imgName.push_back(c);
        }

        sor[i] = {imgName, i};

        size_t numFeatPoints;
        readBin(imagesFs, numFeatPoints);
        std::cout << imgName << ": " << numFeatPoints
                  << " feats, q=" << q.coeffs().transpose()
                  << ", t=" << t.transpose() << "\n";

        std::string depthImagePath =
            (path(depthPath) / imgName).replace_extension(".exr").string();
        cv::Mat depth = cv::imread(depthImagePath,
                                   cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

        for (size_t j = 0; j < numFeatPoints; ++j) {
            double x, y;
            int64_t points3dId;
            readBin(imagesFs, x);
            readBin(imagesFs, y);
            readBin(imagesFs, points3dId);
            if (points3dId >= 0) {
                int64_t points3dIdx = pointIdMap[points3dId];
                int yf = (int)std::floor(y), xf = (int)std::floor(x);
                float yi = y - yf, xi = x - xf;
                float ztl = depth.at<float>(yf, xf);
                float ztr = depth.at<float>(yf, xf + 1);
                float zbl = depth.at<float>(yf + 1, xf);
                float zbr = depth.at<float>(yf + 1, xf + 1);
                if (ztl > 0.0 && ztr > 0.0 && zbl > 0.0 && zbr > 0.0) {
                    // Bilinear
                    float zl = ztl * yi + zbl * (1. - yi);
                    float zr = ztr * yi + zbr * (1. - yi);
                    float z = zl * xi + zr * (1. - xi);
                    double zFromColmap =
                        (trans * P.col(points3dIdx).homogeneous()).z();
                    scales.push_back(z / zFromColmap);
                }
            }
        }

        // x -y -z -> x y z
        trans.bottomRightCorner<2, 1>() *= -1;
        trans.block<1, 2>(0, 1) *= -1;
        trans.bottomLeftCorner<2, 1>() *= -1;
    }
    Eigen::Map<Eigen::VectorXd> scalesVec(scales.data(), scales.size());
    double scaleMean = scalesVec.mean();
    {
        scalesVec.array() -= scaleMean;
        double scaleStd =
            std::sqrt(scalesVec.squaredNorm() / (double)scales.size());
        std::cout << ">>> Estimated depth scaling (real [meters]/COLMAP) "
                  << scaleMean << ", std=" << scaleStd << "\n";
        if (scaleStd / scaleMean > 0.1) {
            std::cout << ">>> WARNING: scale error is high, poses may be too "
                         "inaccurate\n";
        }
    }
    for (size_t i = 0; i < numImages; ++i) {
        Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> trans(
            T.col(i).data());
        trans.rightCols<1>() *= scaleMean;
        // Make it cam -> world
        ark::util::invHomogeneous<double, Eigen::RowMajor>(trans);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    namespace po = boost::program_options;

    // **** BEGIN ARG PARSING ****
    string dataPath, logName, colmapExe, matchType, camModel, intrinName;
    int numThreads;
    bool rerun, noRefineIntrin;
    po::options_description desc("Option arguments");
    po::options_description descPositional(
        "COLMAP wrapper for RGB-D camera pose estimation, with automatic "
        "scale "
        "correction\n(c) Alex Yu 2020\n"
        "\nPositional arguments");
    po::options_description descCombined("");

    desc.add_options()("help", "produce help message")(
        "log-name",
        po::value<string>(&logName)->default_value("colmap_output.txt"),
        "log file name")("force-rerun", po::bool_switch(&rerun),
                         "force rerun COLMAP even if outputs already exist")(
        "no-refine-intrin", po::bool_switch(&noRefineIntrin),
        "do not refine intrinsics")(
        "cam-model", po::value<string>(&camModel)->default_value("PINHOLE"),
        "camera model, one of SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL")(
        "colmap", po::value<string>(&colmapExe)->default_value("colmap"),
        "COLMAP executable")(
        "threads,j",
        po::value<int>(&numThreads)
            ->default_value(std::thread::hardware_concurrency()),
        "number of CPU threads to use for COLMAP mapper")(
        "match-type,m",
        po::value<string>(&matchType)->default_value("exhaustive"),
        "COLMAP matching type, either 'exhaustive' or 'sequential'")(
        "intrin", po::value<string>(&intrinName)->default_value("intrin.txt"),
        "intrinsics file to use inside data directory");
    descPositional.add_options()("data-path",
                                 po::value<string>(&dataPath)->required(),
                                 "Data path, as constructed by rgbdrec");

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

    if (matchType != "exhaustive" && matchType != "sequential") {
        std::cerr << "Error: matcher type " << matchType
                  << " is not valid, options are: exhaustive, sequential\n";
        return 1;
    }
    // **** END ARG PARSING ****

    namespace bp = boost::process;
    using boost::filesystem::exists;

    path rootPath = path(dataPath);
    string databasePath = (rootPath / "database.db").string();
    string imagesPath = (rootPath / "rgb").string();
    string depthPath = (rootPath / "depth_exr").string();
    string logPath = (rootPath / logName).string();
    string intrinFile = (rootPath / intrinName).string();
    // COLMAP outputs
    path sparsePath = rootPath / "sparse";
    string camerasFile = (sparsePath / "0" / "cameras.bin").string();
    string imagesFile = (sparsePath / "0" / "images.bin").string();
    string points3dFile = (sparsePath / "0" / "points3D.bin").string();
    // Our outputs
    string transformsFile = (rootPath / "poses.txt").string();
    string intrinOutFile = (rootPath / "intrin_refine.txt").string();

    if (!exists(rootPath) || !exists(imagesPath) || !exists(depthPath)) {
        std::cerr << "Error: Invalid data directory: " << rootPath
                  << ". Please make sure it contains rgb/ subdirectory.\n";
        return 1;
    }

    size_t numRGBFiles = std::count_if(
        boost::filesystem::directory_iterator(imagesPath),
        boost::filesystem::directory_iterator(),
        static_cast<bool (*)(const path&)>(boost::filesystem::is_regular_file));

    ark::CameraIntrin intrin;
    intrin.readFile(intrinFile);
    std::ostringstream intrinTxt;
    intrinTxt << std::setprecision(12);
    if (camModel == "SIMPLE_PINHOLE") {
        intrinTxt << intrin.fx << "," << intrin.cx << "," << intrin.cy;
    } else if (camModel == "PINHOLE") {
        intrinTxt << intrin.fx << "," << intrin.fy << "," << intrin.cx << ","
                  << intrin.cy;
    } else if (camModel == "SIMPLE_RADIAL") {
        intrinTxt << intrin.fx << "," << intrin.cx << "," << intrin.cy << ","
                  << intrin.k[0];
    } else if (camModel == "RADIAL") {
        intrinTxt << intrin.fx << "," << intrin.cx << "," << intrin.cy << ","
                  << intrin.k[0] << "," << intrin.k[1];
    } else {
        std::cerr << "Error: Unsupported camera model " << camModel << "\n";
        return 1;
    }
    if (!rerun && exists(databasePath) && exists(camerasFile) &&
        exists(imagesFile) && exists(points3dFile)) {
        std::cout << ">>> Note: not runnning COLMAP because "
                     "sparse/0/[camera|images|points3D].bin exist\n"
                     "    add --force-rerun"
                  << std::endl;
    } else {
        if (!exists(sparsePath)) {
            boost::filesystem::create_directories(sparsePath);
        }
        if (rerun) {
            if (exists(databasePath)) {
                boost::filesystem::remove(databasePath);
            }
        }

        // **** BEGIN COLMAP COMMANDS ****
        string featureExtractCmd =
            colmapExe + " feature_extractor --database_path " + databasePath +
            " --image_path " + imagesPath + " --ImageReader.single_camera 1 " +
            " --ImageReader.camera_model " + camModel +
            " --ImageReader.camera_params " + intrinTxt.str();

        string featureMatcherCmd = colmapExe + " " + matchType +
                                   "_matcher --database_path " + databasePath;

        string mapperCmd =
            colmapExe + " mapper --database_path " + databasePath +
            " --image_path " + imagesPath + " --output_path " +
            sparsePath.string() + " --Mapper.num_threads " +
            std::to_string(numThreads) + " --Mapper.init_min_tri_angle 4 " +
            (noRefineIntrin ? "--Mapper.ba_refine_focal_length 0 "
                              "--Mapper.ba_refine_principal_point 0 "
                              "--Mapper.ba_refine_extra_params 0"
                            : "");
        // **** END COLMAP COMMANDS ****

        // **** BEGIN COLMAP EXECUTION ****
        std::cout << ">>> COLMAP start" << std::endl;

        std::cout << ">>> Running COLMAP feature extractor: "
                  << featureExtractCmd << std::endl;
        if (bp::system(featureExtractCmd, bp::std_out > logPath)) {
            std::cerr << "COLMAP feature extractor FAILED, check " << logPath
                      << "\n";
            std::exit(1);
        }
        std::cout << ">>> Running COLMAP feature matcher (" << matchType
                  << "): " << featureMatcherCmd << std::endl;
        if (bp::system(featureMatcherCmd, bp::std_out > logPath)) {
            std::cerr << "COLMAP feature matcher FAILED, check " << logPath
                      << "\n";
            std::exit(1);
        }

        std::cout << ">>> Running COLMAP mapper (" << std::to_string(numThreads)
                  << " threads) [warning: CPU only, very slow]: " << mapperCmd
                  << std::endl;
        if (bp::system(mapperCmd, bp::std_out > logPath)) {
            std::cerr << "COLMAP mapper FAILED, check " << logPath << "\n";
            std::exit(1);
        }

        std::cout << ">>> COLMAP done" << std::endl;
        // **** END COLMAP EXECUTION ****
    }

    if (!exists(imagesFile) || !exists(points3dFile)) {
        std::cerr << "Error: sparse/0/images.bin not available, exiting\n";
        return 1;
    }

    std::cout << ">>> Parsing COLMAP output" << std::endl;

    size_t numImages, numPoints;
    // Transforms
    Eigen::Matrix<double, 12, Eigen::Dynamic> T;
    // Matched points in 3D
    Eigen::Matrix<double, 3, Eigen::Dynamic> P;
    // Files sorted by name (file name, file idx in T)
    std::vector<std::pair<std::string, size_t>> sor;
    // Map matched point id to index in P
    std::map<size_t, size_t> pointIdMap;

    ark::CameraIntrin origIntrin = intrin;
    int err;
    err = readCamerasBin(camerasFile, intrin);
    if (err) return err;
    if (intrin != origIntrin) {
        std::cout << ">>> Note: intrinsics have been refined by COLMAP\n";
    } else {
        std::cout << ">>> Note: intrinsics are from sensor\n";
    }
    intrin.writeFile(intrinOutFile);

    err = readPoints3dBin(points3dFile, numPoints, P, pointIdMap);
    if (err) return err;

    err = readImageBin(imagesFile, depthPath, intrin, numRGBFiles, P,
                       pointIdMap, numImages, T, sor);
    if (err) return err;

    std::cout << ">>> Writing poses.bin\n";
    {
        std::ofstream transformsFs(transformsFile);
        transformsFs << std::setprecision(12);
        std::sort(sor.begin(), sor.end());
        Eigen::IOFormat format(Eigen::StreamPrecision, Eigen::DontAlignCols,
                               " ", "\n");
        transformsFs << T.transpose().format(format);
    }

    std::cout << ">>> Finished\n";
}
