#ifndef PERCEPTION_LIVE_MULTISWEEP_HPP_
#define PERCEPTION_LIVE_MULTISWEEP_HPP_

#include <cstddef>
#include <deque>
#include <string>
#include <vector>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>

#include "perception/pointcloud_preprocessor.hpp"

namespace perception
{

std::vector<Point> decodePointCloud(
    const sensor_msgs::msg::PointCloud2 & cloud,
    const std::string & intensity_field,
    float dt,
    const std::string & dt_field = "");

class LiveMultiSweepBuffer
{
public:
    explicit LiveMultiSweepBuffer(std::size_t previous_sweeps);

    void push(sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud);
    bool ready() const;
    std::size_t size() const;
    std::size_t capacity() const;

    std::vector<Point> assemble(
        const tf2_ros::Buffer & tf_buffer,
        const std::string & fixed_frame,
        const std::string & intensity_field) const;

private:
    std::size_t capacity_;
    std::deque<sensor_msgs::msg::PointCloud2::ConstSharedPtr> sweeps_;
};

}  // namespace perception

#endif  // PERCEPTION_LIVE_MULTISWEEP_HPP_
