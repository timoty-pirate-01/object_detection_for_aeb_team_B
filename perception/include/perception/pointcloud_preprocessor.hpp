#ifndef POINTCLOUD_PREPROCESSOR_HPP_
#define POINTCLOUD_PREPROCESSOR_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace perception
{

struct Point
{
    float x{};
    float y{};
    float z{};
    float intensity{};
    float feature_4{};
};

struct VoxelizationConfig
{
    std::array<float, 3> min_range{};
    std::array<float, 3> max_range{};
    std::array<float, 3> voxel_size{};
    std::size_t max_voxels{};
};

struct VoxelizedPointCloud
{
    static constexpr std::size_t kPointsPerVoxel = 20;
    static constexpr std::size_t kFeaturesPerPoint = 5;

    // Row-major tensors: voxels [N, 20, 5], num_points [N], coors [N, 4].
    std::vector<float> voxels;
    std::vector<std::int32_t> num_points;
    std::vector<std::int32_t> coors;
};

class PointCloudPreprocessor
{
public:
    explicit PointCloudPreprocessor(VoxelizationConfig config);

    VoxelizedPointCloud process(
        const std::vector<Point> & points,
        std::int32_t batch_index = 0) const;

private:
    VoxelizationConfig config_;
};

}  // namespace perception

#endif  // POINTCLOUD_PREPROCESSOR_HPP_
