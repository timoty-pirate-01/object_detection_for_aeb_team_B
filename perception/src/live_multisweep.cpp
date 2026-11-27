#include "perception/live_multisweep.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include <rclcpp/time.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace perception
{

namespace
{
std::size_t fieldSize(std::uint8_t datatype)
{
    switch (datatype) {
        case sensor_msgs::msg::PointField::INT8:
        case sensor_msgs::msg::PointField::UINT8:
            return 1;
        case sensor_msgs::msg::PointField::INT16:
        case sensor_msgs::msg::PointField::UINT16:
            return 2;
        case sensor_msgs::msg::PointField::INT32:
        case sensor_msgs::msg::PointField::UINT32:
        case sensor_msgs::msg::PointField::FLOAT32:
            return 4;
        case sensor_msgs::msg::PointField::FLOAT64:
            return 8;
        default:
            throw std::runtime_error("Nem támogatott PointCloud2 mezőtípus");
    }
}

template<typename T>
T readValue(
    const std::uint8_t * data,
    bool is_bigendian)
{
    std::array<std::uint8_t, sizeof(T)> bytes{};
    std::copy(data, data + sizeof(T), bytes.begin());
    const std::uint16_t endian_probe = 1;
    const bool host_is_bigendian =
        reinterpret_cast<const std::uint8_t *>(&endian_probe)[0] == 0;
    if (host_is_bigendian != is_bigendian) {
        std::reverse(bytes.begin(), bytes.end());
    }
    T value{};
    std::memcpy(&value, bytes.data(), sizeof(T));
    return value;
}

float readField(
    const sensor_msgs::msg::PointCloud2 & cloud,
    const sensor_msgs::msg::PointField & field,
    std::size_t point_offset)
{
    if (field.count == 0 || field.offset + fieldSize(field.datatype) > cloud.point_step) {
        throw std::runtime_error("Érvénytelen PointCloud2 mezőleírás");
    }
    const auto * data = cloud.data.data() + point_offset + field.offset;
    switch (field.datatype) {
        case sensor_msgs::msg::PointField::INT8:
            return static_cast<float>(readValue<std::int8_t>(data, cloud.is_bigendian));
        case sensor_msgs::msg::PointField::UINT8:
            return static_cast<float>(readValue<std::uint8_t>(data, cloud.is_bigendian));
        case sensor_msgs::msg::PointField::INT16:
            return static_cast<float>(readValue<std::int16_t>(data, cloud.is_bigendian));
        case sensor_msgs::msg::PointField::UINT16:
            return static_cast<float>(readValue<std::uint16_t>(data, cloud.is_bigendian));
        case sensor_msgs::msg::PointField::INT32:
            return static_cast<float>(readValue<std::int32_t>(data, cloud.is_bigendian));
        case sensor_msgs::msg::PointField::UINT32:
            return static_cast<float>(readValue<std::uint32_t>(data, cloud.is_bigendian));
        case sensor_msgs::msg::PointField::FLOAT32:
            return readValue<float>(data, cloud.is_bigendian);
        case sensor_msgs::msg::PointField::FLOAT64:
            return static_cast<float>(readValue<double>(data, cloud.is_bigendian));
        default:
            throw std::runtime_error("Nem támogatott PointCloud2 mezőtípus");
    }
}

std::int64_t timestampNanoseconds(const builtin_interfaces::msg::Time & stamp)
{
    return static_cast<std::int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
}

}  // namespace

std::vector<Point> decodePointCloud(
    const sensor_msgs::msg::PointCloud2 & cloud,
    const std::string & intensity_field,
    float dt,
    const std::string & dt_field)
{
    if (cloud.point_step == 0 ||
        cloud.row_step < static_cast<std::uint64_t>(cloud.width) * cloud.point_step ||
        cloud.data.size() < static_cast<std::uint64_t>(cloud.row_step) * cloud.height)
    {
        throw std::runtime_error("A PointCloud2 adatmérete vagy lépésköze érvénytelen");
    }

    std::unordered_map<std::string, sensor_msgs::msg::PointField> fields;
    for (const auto & field : cloud.fields) {
        fields.emplace(field.name, field);
    }

    const auto x_field = fields.find("x");
    const auto y_field = fields.find("y");
    const auto z_field = fields.find("z");
    if (x_field == fields.end() || y_field == fields.end() || z_field == fields.end()) {
        throw std::runtime_error("A PointCloud2-ből hiányzik az x, y vagy z mező");
    }
    const auto intensity = fields.find(intensity_field);
    const auto point_dt = fields.find(dt_field);
    if (!dt_field.empty() && point_dt == fields.end()) {
        throw std::runtime_error("A megadott PointCloud2 időeltérésmező nem található");
    }

    std::vector<Point> points;
    points.reserve(static_cast<std::size_t>(cloud.width) * cloud.height);
    for (std::uint32_t row = 0; row < cloud.height; ++row) {
        for (std::uint32_t column = 0; column < cloud.width; ++column) {
            const std::size_t point_offset =
                static_cast<std::size_t>(row) * cloud.row_step +
                static_cast<std::size_t>(column) * cloud.point_step;
            points.push_back({
                readField(cloud, x_field->second, point_offset),
                readField(cloud, y_field->second, point_offset),
                readField(cloud, z_field->second, point_offset),
                intensity == fields.end() ? 0.0F :
                    readField(cloud, intensity->second, point_offset),
                point_dt == fields.end() ? dt :
                    readField(cloud, point_dt->second, point_offset)});
        }
    }
    return points;
}

LiveMultiSweepBuffer::LiveMultiSweepBuffer(std::size_t previous_sweeps)
: capacity_(previous_sweeps + 1)
{
    if (capacity_ == 0) {
        throw std::invalid_argument("Legalább egy sweep szükséges");
    }
}

void LiveMultiSweepBuffer::push(sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud)
{
    if (!cloud) {
        throw std::invalid_argument("Null PointCloud2 üzenet nem pufferelhető");
    }
    if (!sweeps_.empty()) {
        const auto new_timestamp = timestampNanoseconds(cloud->header.stamp);
        const auto last_timestamp = timestampNanoseconds(sweeps_.back()->header.stamp);
        if (new_timestamp == last_timestamp) {
            throw std::invalid_argument("Ismétlődő PointCloud2 időbélyeg");
        }
        if (new_timestamp < last_timestamp) {
            sweeps_.clear();
        }
    }

    sweeps_.push_back(std::move(cloud));
    while (sweeps_.size() > capacity_) {
        sweeps_.pop_front();
    }
}

bool LiveMultiSweepBuffer::ready() const
{
    return sweeps_.size() == capacity_;
}

std::size_t LiveMultiSweepBuffer::size() const
{
    return sweeps_.size();
}

std::size_t LiveMultiSweepBuffer::capacity() const
{
    return capacity_;
}

std::vector<Point> LiveMultiSweepBuffer::assemble(
    const tf2_ros::Buffer & tf_buffer,
    const std::string & fixed_frame,
    const std::string & intensity_field) const
{
    if (!ready()) {
        throw std::runtime_error("A live MultiSweep puffer még nem telt meg");
    }
    if (fixed_frame.empty()) {
        throw std::invalid_argument("A fixed_frame paraméter nem lehet üres");
    }

    const auto & current = sweeps_.back();
    const rclcpp::Time current_time(current->header.stamp);
    const std::int64_t current_ns = timestampNanoseconds(current->header.stamp);
    std::vector<Point> combined;

    for (auto iter = sweeps_.rbegin(); iter != sweeps_.rend(); ++iter) {
        const auto & sweep = *iter;
        const std::int64_t sweep_ns = timestampNanoseconds(sweep->header.stamp);
        const double dt_seconds = static_cast<double>(current_ns - sweep_ns) / 1.0e9;
        if (dt_seconds < 0.0) {
            throw std::runtime_error("A sweep időbélyege későbbi a kulcsképkockánál");
        }

        auto points = decodePointCloud(
            *sweep, intensity_field, static_cast<float>(dt_seconds));
        const rclcpp::Time sweep_time(sweep->header.stamp);
        const auto transform = tf_buffer.lookupTransform(
            current->header.frame_id,
            current_time,
            sweep->header.frame_id,
            sweep_time,
            fixed_frame,
            rclcpp::Duration::from_nanoseconds(0));
        tf2::Transform transform_matrix;
        tf2::fromMsg(transform.transform, transform_matrix);

        for (auto & point : points) {
            const auto transformed = transform_matrix * tf2::Vector3(point.x, point.y, point.z);
            point.x = static_cast<float>(transformed.x());
            point.y = static_cast<float>(transformed.y());
            point.z = static_cast<float>(transformed.z());
        }
        combined.insert(combined.end(), points.begin(), points.end());
    }
    return combined;
}

}  // namespace perception
