#include "perception/perceptionNode.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <tf2/exceptions.h>

namespace perception
{

namespace
{
std::size_t configuredPreviousSweeps(rclcpp::Node & node)
{
    const int sweeps_num = node.declare_parameter<int>("sweeps_num", 9);
    if (sweeps_num < 0) {
        throw std::invalid_argument("A sweeps_num nem lehet negatív");
    }
    return static_cast<std::size_t>(sweeps_num);
}

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
: Node("perception_node"),
  tf_buffer_(this->get_clock()),
  tf_listener_(tf_buffer_),
  sweep_buffer_(configuredPreviousSweeps(*this))
{
    // Az MMDeploy által exportált end2end.onnx útvonala
    const auto model_path = this->declare_parameter<std::string>("model_path", "model/model.onnx");
    const auto use_cuda = this->declare_parameter<bool>("use_cuda", false);
    intensity_field_ = this->declare_parameter<std::string>("intensity_field", "intensity");
    dt_field_ = this->declare_parameter<std::string>("dt_field", "dt");
    input_is_multisweep_ = this->declare_parameter<bool>("input_is_multisweep", false);
    fixed_frame_ = this->declare_parameter<std::string>("fixed_frame", "odom");
    const auto pointcloud_topic =
        this->declare_parameter<std::string>("pointcloud_topic", "pointcloud");
    preprocessor_ = std::make_unique<PointCloudPreprocessor>(
        centerPointPillarNuscenesConfig());
    loadModel(model_path, use_cuda);

    sub_pointcloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        pointcloud_topic,
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
        throw std::runtime_error("A 'model_path' paraméter nincs megadva");
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
        input_names_.push_back(name.get());
        RCLCPP_INFO(get_logger(), "  bemenet: %s %s", name.get(), shapeToString(shape).c_str());
    }
    for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator);
        auto shape = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        output_names_.push_back(name.get());
        RCLCPP_INFO(get_logger(), "  kimenet: %s %s", name.get(), shapeToString(shape).c_str());
    }
}

void PerceptionNode::pointCloudCallback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
    if (!msg || msg->data.empty()) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Üres PointCloud2 üzenet érkezett");
        return;
    }

    std::vector<Point> points;
    if (input_is_multisweep_) {
        try {
            points = decodePointCloud(*msg, intensity_field_, 0.0F, dt_field_);
        } catch (const std::exception & e) {
            RCLCPP_ERROR(get_logger(), "MultiSweep PointCloud2 dekódolási hiba: %s", e.what());
            return;
        }
    } else {
        try {
            sweep_buffer_.push(msg);
        } catch (const std::exception & e) {
            RCLCPP_WARN(get_logger(), "PointCloud2 pufferelési hiba: %s", e.what());
            return;
        }

        if (!sweep_buffer_.ready()) {
            RCLCPP_INFO_THROTTLE(
                get_logger(), *get_clock(), 5000,
                "Live MultiSweep puffer feltöltése: %zu/%zu sweep",
                sweep_buffer_.size(), sweep_buffer_.capacity());
            return;
        }

        try {
            points = sweep_buffer_.assemble(tf_buffer_, fixed_frame_, intensity_field_);
        } catch (const tf2::TransformException & e) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000,
                "A MultiSweep TF transzformációja nem érhető el: %s", e.what());
            return;
        } catch (const std::exception & e) {
            RCLCPP_ERROR(get_logger(), "MultiSweep feldolgozási hiba: %s", e.what());
            return;
        }
    }

    const auto voxelized = preprocessor_->process(points);
    if (voxelized.num_points.empty()) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "A PointCloud2 nem tartalmaz a model tartományába eső pontot");
        return;
    }

    runInference(voxelized, msg->header.stamp.nanosec +
        static_cast<std::int64_t>(msg->header.stamp.sec) * 1000000000LL);
}

void PerceptionNode::runInference(
    const VoxelizedPointCloud & voxelized,
    std::int64_t timestamp_ns)
{
    (void)timestamp_ns;
    if (input_names_.size() != session_->GetInputCount()) {
        throw std::runtime_error("Az ONNX bemenetnevek száma nem egyezik a modellével");
    }

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputs;
    std::vector<const char *> input_name_ptrs;
    inputs.reserve(input_names_.size());
    input_name_ptrs.reserve(input_names_.size());

    for (std::size_t i = 0; i < input_names_.size(); ++i) {
        const std::string & name = input_names_[i];
        input_name_ptrs.push_back(name.c_str());

        if (name.find("voxels") != std::string::npos) {
            const std::array<int64_t, 3> shape{
                static_cast<int64_t>(voxelized.num_points.size()),
                static_cast<int64_t>(VoxelizedPointCloud::kPointsPerVoxel),
                static_cast<int64_t>(VoxelizedPointCloud::kFeaturesPerPoint)};
            inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float *>(voxelized.voxels.data()),
                voxelized.voxels.size(),
                shape.data(),
                shape.size()));
        } else if (name.find("num_points") != std::string::npos) {
            const std::array<int64_t, 1> shape{
                static_cast<int64_t>(voxelized.num_points.size())};
            inputs.push_back(Ort::Value::CreateTensor<std::int32_t>(
                memory_info,
                const_cast<std::int32_t *>(voxelized.num_points.data()),
                voxelized.num_points.size(),
                shape.data(),
                shape.size()));
        } else if (name.find("coors") != std::string::npos) {
            const std::array<int64_t, 2> shape{
                static_cast<int64_t>(voxelized.num_points.size()),
                4};
            inputs.push_back(Ort::Value::CreateTensor<std::int32_t>(
                memory_info,
                const_cast<std::int32_t *>(voxelized.coors.data()),
                voxelized.coors.size(),
                shape.data(),
                shape.size()));
        } else {
            throw std::runtime_error(
                "Ismeretlen ONNX bemenetnév: " + name);
        }
    }

    std::vector<const char *> output_name_ptrs;
    output_name_ptrs.reserve(output_names_.size());
    for (const auto & name : output_names_) {
        output_name_ptrs.push_back(name.c_str());
    }

    Ort::RunOptions run_options;
    auto outputs = session_->Run(
        run_options,
        input_name_ptrs.data(),
        inputs.data(),
        inputs.size(),
        output_name_ptrs.empty() ? nullptr : output_name_ptrs.data(),
        output_name_ptrs.size());

    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "ONNX inference kész: %zu voxel, %zu modellkimenet",
        voxelized.num_points.size(),
        outputs.size());
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
