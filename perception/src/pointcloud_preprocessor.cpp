#include "perception/pointcloud_preprocessor.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace perception
{

namespace
{

struct VoxelIndex
{
    std::int32_t x;
    std::int32_t y;
    std::int32_t z;

    bool operator==(const VoxelIndex & other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelIndexHash
{
    std::size_t operator()(const VoxelIndex & index) const
    {
        std::size_t seed = std::hash<std::int32_t>{}(index.x);
        seed ^= std::hash<std::int32_t>{}(index.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<std::int32_t>{}(index.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

bool isFinite(const Point & point)
{
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) &&
           std::isfinite(point.intensity) && std::isfinite(point.dt);
}

}  // namespace

VoxelizationConfig centerPointPillarNuscenesConfig()
{
    return {
        {-51.2F, -51.2F, -5.0F},
        {51.2F, 51.2F, 3.0F},
        {0.2F, 0.2F, 8.0F},
        40000};
}

PointCloudPreprocessor::PointCloudPreprocessor(VoxelizationConfig config)
: config_(std::move(config))
{
    if (config_.max_voxels == 0 ||
        config_.max_voxels > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
    {
        throw std::invalid_argument("max_voxels must be in the range [1, INT32_MAX]");
    }

    for (std::size_t axis = 0; axis < 3; ++axis) {
        const float min_value = config_.min_range[axis];
        const float max_value = config_.max_range[axis];
        const float voxel_size = config_.voxel_size[axis];

        if (!std::isfinite(min_value) || !std::isfinite(max_value) ||
            !std::isfinite(voxel_size) || min_value >= max_value || voxel_size <= 0.0F)
        {
            throw std::invalid_argument("voxel range and size must be finite and valid");
        }

        const double grid_size =
            (static_cast<double>(max_value) - static_cast<double>(min_value)) /
            static_cast<double>(voxel_size);
        const double rounded_grid_size = std::round(grid_size);
        if (rounded_grid_size < 1.0 ||
            rounded_grid_size > static_cast<double>(std::numeric_limits<std::int32_t>::max()) ||
            std::abs(grid_size - rounded_grid_size) > 1e-4)
        {
            throw std::invalid_argument("range extent must be an integer multiple of voxel size");
        }
    }
}

VoxelizedPointCloud PointCloudPreprocessor::process(
    const std::vector<Point> & points,
    std::int32_t batch_index) const
{
    if (batch_index < 0) {
        throw std::invalid_argument("batch_index must be non-negative");
    }

    VoxelizedPointCloud output;
    std::unordered_map<VoxelIndex, std::size_t, VoxelIndexHash> voxel_lookup;

    for (const Point & point : points) {
        if (!isFinite(point)) {
            continue;
        }

        const std::array<float, 3> position{point.x, point.y, point.z};
        VoxelIndex index{};
        bool in_range = true;
        std::array<std::int32_t *, 3> coordinates{&index.x, &index.y, &index.z};

        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (position[axis] < config_.min_range[axis] ||
                position[axis] >= config_.max_range[axis])
            {
                in_range = false;
                break;
            }

            const double voxel_coordinate =
                (static_cast<double>(position[axis]) -
                 static_cast<double>(config_.min_range[axis])) /
                static_cast<double>(config_.voxel_size[axis]);
            *coordinates[axis] = static_cast<std::int32_t>(std::floor(voxel_coordinate));
        }

        if (!in_range) {
            continue;
        }

        auto voxel = voxel_lookup.find(index);
        std::size_t voxel_id;
        if (voxel == voxel_lookup.end()) {
            if (output.num_points.size() >= config_.max_voxels) {
                continue;
            }

            voxel_id = output.num_points.size();
            voxel_lookup.emplace(index, voxel_id);
            output.num_points.push_back(0);
            output.coors.insert(
                output.coors.end(),
                {batch_index, index.z, index.y, index.x});
            output.voxels.resize(
                output.voxels.size() +
                VoxelizedPointCloud::kPointsPerVoxel *
                VoxelizedPointCloud::kFeaturesPerPoint,
                0.0F);
        } else {
            voxel_id = voxel->second;
        }

        const std::int32_t point_id = output.num_points[voxel_id];
        if (point_id >= static_cast<std::int32_t>(VoxelizedPointCloud::kPointsPerVoxel)) {
            continue;
        }

        const std::array<float, VoxelizedPointCloud::kFeaturesPerPoint> features{
            point.x, point.y, point.z, point.intensity, point.dt};
        const std::size_t feature_offset =
            (voxel_id * VoxelizedPointCloud::kPointsPerVoxel +
             static_cast<std::size_t>(point_id)) *
            VoxelizedPointCloud::kFeaturesPerPoint;
        for (std::size_t feature = 0; feature < features.size(); ++feature) {
            output.voxels[feature_offset + feature] = features[feature];
        }
        ++output.num_points[voxel_id];
    }

    return output;
}

}  // namespace perception
