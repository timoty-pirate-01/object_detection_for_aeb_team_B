#include "perception/pointcloud_preprocessor.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

perception::VoxelizationConfig makeConfig(std::size_t max_voxels = 8)
{
    auto config = perception::centerPointPillarNuscenesConfig();
    config.max_voxels = max_voxels;
    return config;
}

void testVoxelLayoutAndFiltering()
{
    perception::PointCloudPreprocessor preprocessor(makeConfig());
    const std::vector<perception::Point> points{
        {0.2F, 0.3F, 0.4F, 0.5F, 0.0F},
        {0.7F, 0.3F, 0.4F, 0.6F, -0.1F},
        {1.2F, 0.3F, 0.4F, 0.7F, 0.0F},
        {51.2F, 0.3F, 0.4F, 0.8F, 0.0F},
        {std::numeric_limits<float>::quiet_NaN(), 0.3F, 0.4F, 0.9F, 0.0F}};

    const auto output = preprocessor.process(points);
    assert(output.num_points.size() == 3);
    assert(output.voxels.size() == 3 * 20 * 5);
    assert(output.coors.size() == 3 * 4);
    assert(output.num_points[0] == 1);
    assert(output.num_points[1] == 1);
    assert(output.num_points[2] == 1);
    assert((output.coors == std::vector<std::int32_t>{
        0, 0, 257, 257,
        0, 0, 257, 259,
        0, 0, 257, 262}));
    assert(output.voxels[0] == 0.2F);
    assert(output.voxels[3] == 0.5F);
    assert(output.voxels[4] == 0.0F);
    assert(output.voxels[100] == 0.7F);
    assert(output.voxels[200] == 1.2F);
}

void testCenterPointConfiguration()
{
    const auto config = perception::centerPointPillarNuscenesConfig();
    assert((config.min_range == std::array<float, 3>{-51.2F, -51.2F, -5.0F}));
    assert((config.max_range == std::array<float, 3>{51.2F, 51.2F, 3.0F}));
    assert((config.voxel_size == std::array<float, 3>{0.2F, 0.2F, 8.0F}));
    assert(config.max_voxels == 40000);
}

void testPointAndVoxelLimits()
{
    perception::PointCloudPreprocessor preprocessor(makeConfig(1));
    std::vector<perception::Point> points;
    for (int i = 0; i < 21; ++i) {
        points.push_back({0.1F, 0.1F, 0.1F, static_cast<float>(i), 0.0F});
    }
    points.push_back({-51.0F, 0.1F, 0.1F, 99.0F, 0.0F});

    const auto output = preprocessor.process(points, 3);
    assert(output.num_points.size() == 1);
    assert(output.num_points[0] == 20);
    assert((output.coors == std::vector<std::int32_t>{3, 0, 256, 256}));
    assert(output.voxels[3] == 0.0F);
    assert(output.voxels[19 * 5 + 3] == 19.0F);
}

void testInvalidConfiguration()
{
    auto config = makeConfig();
    config.voxel_size[1] = 0.0F;
    bool threw = false;
    try {
        perception::PointCloudPreprocessor preprocessor(config);
        (void)preprocessor;
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);
}

void testEmptyInput()
{
    perception::PointCloudPreprocessor preprocessor(makeConfig());
    const auto output = preprocessor.process({});
    assert(output.voxels.empty());
    assert(output.num_points.empty());
    assert(output.coors.empty());
}

void testCenterPointBoundsAndGrid()
{
    perception::PointCloudPreprocessor preprocessor(
        perception::centerPointPillarNuscenesConfig());
    const std::vector<perception::Point> points{
        {-51.2F, -51.2F, -5.0F, 1.0F, 0.0F},
        {-51.0F, -51.0F, 2.99F, 2.0F, 0.0F},
        {51.2F, 51.2F, 3.0F, 3.0F, 0.0F}};

    const auto output = preprocessor.process(points);
    assert(output.num_points.size() == 2);
    assert(output.coors[2] == 0);
    assert(output.coors[3] == 0);
    assert(output.coors[6] == 1);
    assert(output.coors[7] == 1);
}

}  // namespace

int main()
{
    testVoxelLayoutAndFiltering();
    testCenterPointConfiguration();
    testPointAndVoxelLimits();
    testInvalidConfiguration();
    testEmptyInput();
    testCenterPointBoundsAndGrid();
    return 0;
}
