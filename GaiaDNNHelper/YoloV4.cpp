#include "YoloV4.hpp"

#include <GaiaExceptions/GaiaExceptions.hpp>

namespace Gaia::DNNHelper
{
    /// Initialize the network object.
    void YoloV4::Initialize(const std::string& cfg_path, const std::string& weights_path, const cv::Size& input_size,
                            cv::dnn::Backend backend_preference, cv::dnn::Target target_preference)
    {
        NetworkInputSize = input_size;

        Network = std::make_unique<cv::dnn::Net>(cv::dnn::readNetFromDarknet(cfg_path, weights_path));
        Network->setPreferableBackend(backend_preference);
        Network->setPreferableTarget(target_preference);

        NetworkOutputNames = Network->getUnconnectedOutLayersNames();
    }

    /// Detect objects.
    std::list<YoloV4::Object> YoloV4::Detect(const cv::Mat& picture, float confidence_threshold, float nms_threshold,
                                             unsigned int top_k)
    {
        Exceptions::NullPointerException::ThrowIfNull(Network.get(), "Network");

        std::list<Object> results;

        auto picture_width = static_cast<float>(picture.cols);
        auto picture_height = static_cast<float>(picture.rows);

        // Feed the network.
        auto input_blob = cv::dnn::blobFromImage(picture, 1/255.f,
                                                 NetworkInputSize, cv::Scalar(),
                                                 true, false);
        Network->setInput(input_blob);
        // Start the forward broadcast and get the output data.
        std::vector<cv::Mat> output_blob;
        Network->forward(output_blob, NetworkOutputNames);

        std::vector<cv::Rect> box_list;
        std::vector<int> id_list;
        std::vector<float> confidence_list;

        for (auto& output_data : output_blob)
        {
            auto* data_address = reinterpret_cast<float*>(output_data.data);

            for (int item_index = 0; item_index < output_data.rows; ++item_index, data_address += output_data.cols)
            {
                // An array of confidences of all corresponding classes.
                cv::Mat class_confidences = output_data.row(item_index).colRange(5, output_data.cols);

                cv::Point most_likely_class;
                double biggest_confidence;
                // This object most likely belongs to the class with biggest confidence.
                cv::minMaxLoc(class_confidences,
                              nullptr, &biggest_confidence,
                              nullptr, &most_likely_class);

                // Filter out the confidence lower than the confidence threshold.
                if (biggest_confidence < confidence_threshold)
                {
                    continue;
                }

                int center_x = static_cast<int>(data_address[0] * picture_width);
                int center_y = static_cast<int>(data_address[1] * picture_height);
                int box_width = static_cast<int>(data_address[2] * picture_width);
                int box_height = static_cast<int>(data_address[3] * picture_height);
                int box_x = center_x - box_width / 2;
                int box_y = center_y - box_height / 2;

                id_list.push_back(most_likely_class.x);
                box_list.emplace_back(box_x, box_y, box_width, box_height);
                confidence_list.push_back(static_cast<float>(biggest_confidence));
            }
        }

        std::vector<int> indices_to_keep;
        cv::dnn::NMSBoxes(box_list, confidence_list, confidence_threshold, 0.2, indices_to_keep, 1.0f,
                          static_cast<int>(top_k));
        for (int id : indices_to_keep)
        {
            results.push_back(Object{
                .ClassID = static_cast<unsigned int>(id_list[id]),
                .Confidence = confidence_list[id],
                .BoundingBox = box_list[id]
            });
        }

        return results;
    }
}
