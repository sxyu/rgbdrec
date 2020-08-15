/** RGB-D point cloud registration visualizer.
 *  Use this after e.g. rgbdpose-colmap: rgbdpose-viewer <directory>
 *  (c) Alex Yu 2019-20, Apache License 2.0 */
#include <string>
#include <vector>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Dense>
#include "meshview/meshview.hpp"

#include "Calibration.h"
#include "Util.h"

namespace {
using boost::filesystem::path;
using std::string;

const float DEPTH_EPS = 0.001;
meshview::PointCloud toPointCloud(const cv::Mat& xyz_map, bool flip_z = false,
                                  bool flip_y = false, int step = 1) {
    size_t n_pts = 0;
    for (int i = 0; i < xyz_map.rows; i += step) {
        const auto* const ptr = xyz_map.ptr<cv::Vec3f>(i);
        for (int j = 0; j < xyz_map.cols; j += step)
            if (ptr[j][2] > DEPTH_EPS) ++n_pts;
    }
    meshview::PointCloud out(n_pts);
    size_t idx = 0;
    float z_fact = flip_z ? -1.f : 1.f;
    float y_fact = flip_y ? -1.f : 1.f;
    for (int i = 0; i < xyz_map.rows; i += step) {
        const auto* const ptr = xyz_map.ptr<cv::Vec3f>(i);
        for (int j = 0; j < xyz_map.cols; j += step) {
            if (ptr[j][2] > DEPTH_EPS) {
                out.verts_pos().row(idx++) << ptr[j][0], ptr[j][1] * y_fact,
                    ptr[j][2] * z_fact;
            }
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    namespace po = boost::program_options;
    string dataPath;
    int interval;
    bool trIdentity, noRefinedIntrin;

    // **** BEGIN ARG PARSING ****
    po::options_description desc("Option arguments");
    po::options_description descPositional(
        "RGB-D point cloud registration visualization for rgbdreg-colmap/etc. "
        "(c) Alex Yu 2020\n\nPositional "
        "arguments");
    po::options_description descCombined("");

    desc.add_options()("help", "produce help message")(
        "interval,n", po::value<int>(&interval)->default_value(1),
        "load interval: loads every x images")(
        "no-refined-intrin,N", po::bool_switch(&noRefinedIntrin),
        "use intrin.txt instead of intrin_refine.txt even if latter is "
        "available")(
        "no-poses,I", po::bool_switch(&trIdentity),
        "ignore poses.bin and set all point cloud transforms to identity");
    descPositional.add_options()("data-path",
                                 po::value<string>(&dataPath)->required(),
                                 "data path, as constructed by rgbdrec");

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
    string intrinFile = (rootPath / "intrin.txt").string();
    {
        string intrinRefineFile = (rootPath / "intrin_refine.txt").string();
        if (exists(intrinRefineFile) && !noRefinedIntrin) {
            std::cout << ">>> Using refined intrinsics\n";
            intrinFile = intrinRefineFile;
        }
    }
    string transformsFile = (rootPath / "poses.txt").string();
    if (!exists(rootPath) || !exists(imagesPath) || !exists(depthsPath)) {
        std::cerr << "Error: Invalid data directory: " << rootPath
                  << ". Please make sure it contains rgb/ subdirectory.\n";
        return 1;
    }

    std::vector<std::string> images = ark::util::listDir(imagesPath);
    std::vector<std::string> depths = ark::util::listDir(depthsPath, ".exr");
    if (images.size() != depths.size()) {
        std::cerr << "Error: Found " << images.size() << " color images but "
                  << depths.size() << " depth images. These must match.\n";
        return 1;
    }

    size_t numImages = images.size();
    Eigen::Matrix<double, 12, Eigen::Dynamic> T(12, numImages);
    {
        if (trIdentity) {
            transformsFile.clear();
        } else if (!exists(transformsFile)) {
            std::cout << "Warning: " << transformsFile
                      << " does not exist, did you use rgbdpose-... to get the "
                         "poses first? All point clouds will be centered\n";
            transformsFile.clear();
        } else {
            std::ifstream ifs(transformsFile);
            for (int i = 0; i < numImages * 12; ++i) {
                ifs >> T.data()[i];
            }
        }
    }

    ark::CameraIntrin intrin;
    intrin.readFile(intrinFile);

    meshview::Viewer viewer;

    for (size_t i = 0; i < numImages; i += interval) {
        cv::Mat depth =
            cv::imread(depths[i], cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
        cv::Mat xyz_map(depth.size(), CV_32FC3);
        // Depth to xyz, by pinhole projection (could be optimized, but
        // whatever)
        float* inPtr;
        cv::Vec3f* outPtr;
        for (int r = 0; r < depth.rows; ++r) {
            inPtr = depth.ptr<float>(r);
            outPtr = xyz_map.ptr<cv::Vec3f>(r);
            for (int c = 0; c < depth.cols; ++c) {
                const float z = inPtr[c];
                outPtr[c] = cv::Vec3f((c - intrin.cx) * z / intrin.fx,
                                      (r - intrin.cy) * z / intrin.fy, z);
            }
        }
        meshview::PointCloud dataCloud = toPointCloud(xyz_map, true, true, 1);

        cv::Mat color = cv::imread(images[i]);
        if (color.type() == CV_8U) {
            cv::cvtColor(color, color, cv::COLOR_GRAY2BGR);
        } else if (color.type() == CV_8UC4) {
            cv::cvtColor(color, color, cv::COLOR_BGRA2BGR);
        } else if (color.type() != CV_8UC3) {
            std::cout << "WARNING: specified color image " << images[i]
                      << " has unsupported color format\n";
            continue;
        }
        if (color.rows != depth.rows || color.cols != depth.cols) {
            std::cout << "WARNING: specified color image " << images[i]
                      << " has different size than the corresponding "
                         "depth map\n";
            continue;
        }
        cv::Vec3f* xyzPtr;
        cv::Vec3b* colorPtr;
        size_t idx = 0;
        for (int r = 0; r < depth.rows; ++r) {
            xyzPtr = xyz_map.ptr<cv::Vec3f>(r);
            colorPtr = color.ptr<cv::Vec3b>(r);
            for (int c = 0; c < depth.cols; ++c) {
                const float z = xyzPtr[c][2];
                if (z > 0.001 && idx < dataCloud.data.rows()) {
                    auto row = dataCloud.verts_rgb().row(idx);
                    row.x() = colorPtr[c][2] / 255.f;
                    row.y() = colorPtr[c][1] / 255.f;
                    row.z() = colorPtr[c][0] / 255.f;
                    ++idx;
                }
            }
        }

        Eigen::Matrix4f trans4f;

        if (transformsFile.empty()) {
            trans4f.setIdentity();
        } else {
            Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> trans(
                T.col(i).data());
            trans4f.topRows<3>().noalias() = trans.cast<float>();
            // trans4f.topRightCorner<3, 1>().setZero();
            trans4f.bottomLeftCorner<1, 3>().setZero();
            trans4f(3, 3) = 1.f;
            std::cout << ">>> LOAD " << depths[i] << ", T=\n"
                      << trans4f << "\n";
        }
        viewer.add_point_cloud(std::move(dataCloud)).set_transform(trans4f);
    }

    viewer.show();
}
