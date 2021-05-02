#include "Darknet.hpp"

#include <GaiaExceptions/GaiaExceptions.hpp>

#include <iostream>

namespace Gaia::DNNHelper
{
    /// Initialize the network object.
    void Darknet::Initialize(const std::string& cfg_path, const std::string& weights_path, const cv::Size& input_size,
                            cv::dnn::Backend backend_preference, cv::dnn::Target target_preference)
    {
        NetworkInputSize = input_size;

        Network = std::make_unique<cv::dnn::Net>(cv::dnn::readNetFromDarknet(cfg_path, weights_path));
        Network->setPreferableBackend(backend_preference);
        Network->setPreferableTarget(target_preference);

        NetworkOutputNames = Network->getUnconnectedOutLayersNames();
    }

    /// Classify the given picture.
    std::tuple<int, float> Darknet::Classify(const cv::Mat &picture)
    {
        Exceptions::NullPointerException::ThrowIfNull(Network.get(), "Network");

        auto picture_width = static_cast<float>(picture.cols);
        auto picture_height = static_cast<float>(picture.rows);

        // Feed the network.
        auto input_blob = cv::dnn::blobFromImage(picture, 1/255.f,
                                                 NetworkInputSize, cv::Scalar(),
                                                 true, false);
        Network->setInput(input_blob);
        // Start the forward broadcast and get the output data.
        std::vector<cv::Mat> output_blob;
        Network->forward(output_blob /*, NetworkOutputNames*/);
        cv::Mat result = output_blob[0].reshape(1, 1);
        cv::Point most_likely_position;
        double most_likely_confidence = 0.0f;
        cv::minMaxLoc(result, nullptr, &most_likely_confidence, nullptr, &most_likely_position);

        return std::tuple{most_likely_position.x, most_likely_confidence};
    }
}