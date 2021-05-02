#pragma once

#include <opencv4/opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <tuple>

namespace Gaia::DNNHelper
{
    /**
     * @brief Helper object for network of Darknet classifier model.
     * @details
     *  This class encapsulate the feed data and acquire output operations of Darknet model.
     */
    class Darknet
    {
    private:
        /// Size of the picture that network input layer accepts.
        cv::Size NetworkInputSize;
        /// Output names of the network.
        std::vector<std::string> NetworkOutputNames;

    protected:
        /// Network object.
        std::unique_ptr<cv::dnn::Net> Network;

    public:
        /**
         * @brief Initialize the network object.
         * @param cfg_path The path to the *.cfg file of the network model.
         * @param weights_path The path to the *.weights file of the network model.
         * @param input_size The size of the input layer of the network model.
         * @param backend_preference The preferable backend for cv::dnn to use.
         * @param target_preference The preferable target device for cv::dnn to compute on.
         */
        void Initialize(const std::string& cfg_path, const std::string& weights_path, const cv::Size& input_size,
                        cv::dnn::Backend backend_preference = cv::dnn::DNN_BACKEND_CUDA,
                        cv::dnn::Target target_preference = cv::dnn::DNN_TARGET_CUDA);

        /**
         * @brief Detect objects in the given picture using the network.
         * @param picture The raw BGR picture used to detect objects in.
         * @return A tuple of the most possible class id and the corresponding confidence.
         * @pre Initialize the network first.
         */
        std::tuple<int, float> Classify(const cv::Mat& picture);
    };
}
