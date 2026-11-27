#include "perception/live_multisweep.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/clock.hpp>
#include <tf2_ros/buffer.h>

namespace
{
void setField(
    sensor_msgs::msg::PointCloud2 & cloud,
    std::size_t point_index,
    std::uint32_t offset,
    const void * value,
    std::size_t size)
{
    std::memcpy(
        cloud.data.data() + point_index * cloud.point_step + offset,
        value,
        size);
}

sensor_msgs::msg::PointCloud2::SharedPtr makeCloud(
    std::int32_t stamp_seconds,
    float x)
{
    auto cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
    cloud->header.frame_id = "lidar";
    cloud->header.stamp.sec = stamp_seconds;
    cloud->height = 1;
    cloud->width = 1;
    cloud->point_step = 16;
    cloud->row_step = 16;
    cloud->fields.resize(4);
    const std::string names[] = {"x", "y", "z", "intensity"};
    for (std::size_t i = 0; i < cloud->fields.size(); ++i) {
        cloud->fields[i].name = names[i];
        cloud->fields[i].offset = static_cast<std::uint32_t>(i * sizeof(float));
        cloud->fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
        cloud->fields[i].count = 1;
    }
    cloud->data.resize(cloud->row_step);
    const float y = 0.0F;
    const float z = 0.0F;
    const float intensity = 12.0F;
    setField(*cloud, 0, 0, &x, sizeof(x));
    setField(*cloud, 0, 4, &y, sizeof(y));
    setField(*cloud, 0, 8, &z, sizeof(z));
    setField(*cloud, 0, 12, &intensity, sizeof(intensity));
    return cloud;
}

geometry_msgs::msg::TransformStamped makeOdomToLidar(
    std::int32_t stamp_seconds,
    double x)
{
    geometry_msgs::msg::TransformStamped transform;
    transform.header.frame_id = "odom";
    transform.child_frame_id = "lidar";
    transform.header.stamp.sec = stamp_seconds;
    transform.transform.translation.x = x;
    transform.transform.rotation.w = 1.0;
    return transform;
}

void testDecodeAndFeatureDefaults()
{
    auto cloud = makeCloud(3, 5.0F);
    const auto points = perception::decodePointCloud(*cloud, "intensity", 0.25F);
    assert(points.size() == 1);
    assert(points[0].x == 5.0F);
    assert(points[0].intensity == 12.0F);
    assert(points[0].dt == 0.25F);

    const auto without_intensity = perception::decodePointCloud(*cloud, "reflectivity", 0.0F);
    assert(without_intensity[0].intensity == 0.0F);

    bool rejected_missing_dt = false;
    try {
        (void)perception::decodePointCloud(*cloud, "intensity", 0.0F, "dt");
    } catch (const std::runtime_error &) {
        rejected_missing_dt = true;
    }
    assert(rejected_missing_dt);
}

void testBufferReadinessAndTimestampOrder()
{
    perception::LiveMultiSweepBuffer buffer(1);
    buffer.push(makeCloud(1, 0.0F));
    assert(buffer.size() == 1);
    assert(buffer.capacity() == 2);
    assert(!buffer.ready());
    buffer.push(makeCloud(2, 0.0F));
    assert(buffer.ready());

    bool rejected_out_of_order = false;
    try {
        buffer.push(makeCloud(2, 0.0F));
    } catch (const std::invalid_argument &) {
        rejected_out_of_order = true;
    }
    assert(rejected_out_of_order);

    buffer.push(makeCloud(1, 0.0F));
    assert(buffer.size() == 1);
    assert(!buffer.ready());
}

void testTransformAndSweepTime()
{
    auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
    tf2_ros::Buffer tf_buffer(clock);
    tf_buffer.setUsingDedicatedThread(true);
    assert(tf_buffer.setTransform(makeOdomToLidar(1, 0.0), "test", false));
    assert(tf_buffer.setTransform(makeOdomToLidar(2, 1.0), "test", false));

    perception::LiveMultiSweepBuffer buffer(1);
    buffer.push(makeCloud(1, 2.0F));
    buffer.push(makeCloud(2, 2.0F));
    const auto points = buffer.assemble(tf_buffer, "odom", "intensity");

    assert(points.size() == 2);
    assert(points[0].dt == 0.0F);
    assert(points[1].dt == 1.0F);
    assert(std::abs(points[0].x - 2.0F) < 1e-5F);
    assert(std::abs(points[1].x - 1.0F) < 1e-5F);
}
}  // namespace

int main()
{
    testDecodeAndFeatureDefaults();
    testBufferReadinessAndTimestampOrder();
    testTransformAndSweepTime();
    return 0;
}
