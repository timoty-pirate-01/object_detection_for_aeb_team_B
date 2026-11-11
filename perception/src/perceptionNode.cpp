#include "perception/perceptionNode.hpp"

#include <chrono>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace perception
{

namespace
{
std::string shapeToString(const std::vector<int64_t> & shape)
{
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        ss << (i ? ", " : "") << (shape[i] < 0 ? std::string("dyn") : std::to_string(shape[i]));
    }
    ss << "]";
    return ss.str();
}
}  // namespace

PerceptionNode::PerceptionNode()
: Node("perception_node")
{
    // Az MMDeploy által exportált end2end.onnx útvonala
    const auto model_path = this->declare_parameter<std::string>("model_path", "");
    const auto use_cuda = this->declare_parameter<bool>("use_cuda", false);
    loadModel(model_path, use_cuda);

    sub_pointcloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "pointcloud",
        10,
        std::bind(&PerceptionNode::pointCloudCallback, this, std::placeholders::_1));

    pub_scenario_ = this->create_publisher<crp_msgs::msg::Scenario>(
        "scenario",
        10);

    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&PerceptionNode::publishScenario, this));
}

void PerceptionNode::loadModel(const std::string & model_path, bool use_cuda)
{
    if (model_path.empty()) {
        throw std::runtime_error("A 'model_path' paraméter nincs megadva (MMDeploy end2end.onnx)");
    }

    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (use_cuda) {
        try {
            OrtCUDAProviderOptions cuda_options{};
            options.AppendExecutionProvider_CUDA(cuda_options);
        } catch (const Ort::Exception & e) {
            RCLCPP_WARN(get_logger(), "CUDA nem elérhető, CPU-n fut: %s", e.what());
        }
    }

    try {
        session_ = std::make_unique<Ort::Session>(ort_env_, model_path.c_str(), options);
    } catch (const Ort::Exception & e) {
        throw std::runtime_error("A modell betöltése sikertelen (" + model_path + "): " + e.what());
    }

    // A betöltött modell be- és kimeneteinek kiírása ellenőrzéshez
    Ort::AllocatorWithDefaultOptions allocator;
    RCLCPP_INFO(get_logger(), "Modell betöltve: %s", model_path.c_str());
    for (size_t i = 0; i < session_->GetInputCount(); ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator);
        auto shape = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        RCLCPP_INFO(get_logger(), "  bemenet: %s %s", name.get(), shapeToString(shape).c_str());
    }
    for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator);
        auto shape = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        RCLCPP_INFO(get_logger(), "  kimenet: %s %s", name.get(), shapeToString(shape).c_str());
    }
}

void PerceptionNode::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    (void)msg;
}

void PerceptionNode::publishScenario()
{
}

}  // namespace perception

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<perception::PerceptionNode>());
    rclcpp::shutdown();
    return 0;
}
