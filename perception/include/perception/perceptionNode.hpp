#ifndef PERCEPTION_NODE_HPP_
#define PERCEPTION_NODE_HPP_

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <crp_msgs/msg/scenario.hpp>

#include "perception/pointcloud_preprocessor.hpp"

namespace perception
{

class PerceptionNode : public rclcpp::Node
{
public:
    PerceptionNode();

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void publishScenario();
    void loadModel(const std::string & model_path, bool use_cuda);
    void runInference(
        const VoxelizedPointCloud & voxelized,
        std::int64_t timestamp_ns);

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pointcloud_;
    rclcpp::Publisher<crp_msgs::msg::Scenario>::SharedPtr pub_scenario_;
    rclcpp::TimerBase::SharedPtr timer_;

    // ONNX Runtime: az Env-nek a Session-nél tovább kell élnie
    Ort::Env ort_env_{ORT_LOGGING_LEVEL_WARNING, "perception"};
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::unique_ptr<PointCloudPreprocessor> preprocessor_;
    std::string intensity_field_;
    std::string dt_field_;
};

}  // namespace perception

#endif  // PERCEPTION_NODE_HPP_
