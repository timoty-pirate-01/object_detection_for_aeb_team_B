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
    return {
        {0.0F, 0.0F, 0.0F},
        {2.0F, 2.0F, 2.0F},
        {1.0F, 1.0F, 1.0F},
        max_voxels};
}

void testVoxelLayoutAndFiltering()
{
    perception::PointCloudPreprocessor preprocessor(makeConfig());
    const std::vector<perception::Point> points{
        {0.2F, 0.3F, 0.4F, 0.5F, 0.0F},
        {0.7F, 0.3F, 0.4F, 0.6F, -0.1F},
        {1.2F, 0.3F, 0.4F, 0.7F, 0.0F},
        {2.0F, 0.3F, 0.4F, 0.8F, 0.0F},
        {std::numeric_limits<float>::quiet_NaN(), 0.3F, 0.4F, 0.9F, 0.0F}};

    const auto output = preprocessor.process(points);
    assert(output.num_points.size() == 2);
    assert(output.voxels.size() == 2 * 20 * 5);
    assert(output.coors.size() == 2 * 4);
    assert(output.num_points[0] == 2);
    assert(output.num_points[1] == 1);
    assert((output.coors == std::vector<std::int32_t>{0, 0, 0, 0, 0, 0, 0, 1}));
    assert(output.voxels[0] == 0.2F);
    assert(output.voxels[3] == 0.5F);
    assert(output.voxels[4] == 0.0F);
    assert(output.voxels[5] == 0.7F);
    assert(output.voxels[100] == 1.2F);
}

void testPointAndVoxelLimits()
{
    perception::PointCloudPreprocessor preprocessor(makeConfig(1));
    std::vector<perception::Point> points;
    for (int i = 0; i < 21; ++i) {
        points.push_back({0.1F, 0.1F, 0.1F, static_cast<float>(i), 0.0F});
    }
    points.push_back({1.1F, 0.1F, 0.1F, 99.0F, 0.0F});

    const auto output = preprocessor.process(points, 3);
    assert(output.num_points.size() == 1);
    assert(output.num_points[0] == 20);
    assert((output.coors == std::vector<std::int32_t>{3, 0, 0, 0}));
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

}  // namespace

int main()
{
    testVoxelLayoutAndFiltering();
    testPointAndVoxelLimits();
    testInvalidConfiguration();
    testEmptyInput();
    return 0;
}
