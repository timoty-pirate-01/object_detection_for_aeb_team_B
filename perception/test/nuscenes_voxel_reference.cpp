#include "perception/pointcloud_preprocessor.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: nuscenes_voxel_reference INPUT.bin OUTPUT_PREFIX\n";
        return 2;
    }

    const std::string input_path = argv[1];
    const std::string output_prefix = argv[2];
    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open input: " << input_path << "\n";
        return 1;
    }

    input.seekg(0, std::ios::end);
    const std::streamoff byte_count = input.tellg();
    input.seekg(0, std::ios::beg);
    if (byte_count < 0 || byte_count % (5 * static_cast<std::streamoff>(sizeof(float))) != 0) {
        std::cerr << "Input is not a float32[x5] array\n";
        return 1;
    }

    const std::size_t point_count =
        static_cast<std::size_t>(byte_count) / (5 * sizeof(float));
    std::vector<float> raw(point_count * 5);
    if (point_count > 0 &&
        !input.read(
            reinterpret_cast<char *>(raw.data()),
            static_cast<std::streamsize>(raw.size() * sizeof(float))))
    {
        std::cerr << "Failed to read input\n";
        return 1;
    }

    std::vector<perception::Point> points;
    points.reserve(point_count);
    for (std::size_t i = 0; i < point_count; ++i) {
        const std::size_t offset = i * 5;
        points.push_back({
            raw[offset],
            raw[offset + 1],
            raw[offset + 2],
            raw[offset + 3],
            raw[offset + 4]});
    }

    perception::PointCloudPreprocessor preprocessor(
        perception::centerPointPillarNuscenesConfig());
    const auto output = preprocessor.process(points);

    std::ofstream voxels(output_prefix + "_voxels.bin", std::ios::binary);
    std::ofstream num_points(output_prefix + "_num_points.bin", std::ios::binary);
    std::ofstream coors(output_prefix + "_coors.bin", std::ios::binary);
    if (!voxels || !num_points || !coors) {
        std::cerr << "Cannot open output files\n";
        return 1;
    }

    if (!output.voxels.empty()) {
        voxels.write(
            reinterpret_cast<const char *>(output.voxels.data()),
            static_cast<std::streamsize>(output.voxels.size() * sizeof(float)));
    }
    if (!output.num_points.empty()) {
        num_points.write(
            reinterpret_cast<const char *>(output.num_points.data()),
            static_cast<std::streamsize>(
                output.num_points.size() * sizeof(std::int32_t)));
    }
    if (!output.coors.empty()) {
        coors.write(
            reinterpret_cast<const char *>(output.coors.data()),
            static_cast<std::streamsize>(output.coors.size() * sizeof(std::int32_t)));
    }

    std::cout << "input_points=" << point_count << "\n";
    std::cout << "voxels=" << output.num_points.size() << "\n";
    std::cout << "voxels_floats=" << output.voxels.size() << "\n";
    std::cout << "num_points_floats=" << output.num_points.size() << "\n";
    std::cout << "coors_ints=" << output.coors.size() << "\n";
    return 0;
}
