#pragma once

#include <opencv4/opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <tuple>

namespace Gaia::DNNHelper
{
    /**
     * @brief Helper object for network of YoloV4 model.
     * @details
     *  This class encapsulate the feed data and acquire output operations of YoloV4 model.
     */
    class YoloV4
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
         * @brief The objects interfered from the input picture.
         * @details
         *  The first element is the class ID of the object.
         *  The second element is the bounding box of the object.
         *  The third element is the confidence of the object.
         */
        struct Object
        {
        public:
            /// The id of the class names which this object belongs to.
            unsigned int ClassID;
            /// The bounding box of this object in the given raw input picture.
            cv::Rect BoundingBox;
            /// This variable describes how likely this object belongs to the  class.
            float Confidence;
        };

        /**
         * @brief Detect objects in the given picture using the network.
         * @param picture The raw BGR picture used to detect objects in.
         * @param confidence_threshold Only objects with confidence bigger than it will be put into the objects list.
         * @return Detected objects.
         * @pre Initialize the network first.
         */
        std::list<Object> Detect(const cv::Mat& picture, float confidence_threshold = 0.0f);
    };
}
