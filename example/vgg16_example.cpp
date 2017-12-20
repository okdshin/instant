#include <iostream>
#include <queue>
#include <vector>

#include <opencv2/opencv.hpp>

#include <instant/instant.hpp>

#include "../external/cmdline.h"

auto crop_and_resize(cv::Mat mat, cv::Size const& size) {
    auto short_edge = std::min(mat.size().width, mat.size().height);
    cv::Rect roi;
    roi.x = (mat.size().width - short_edge) / 2;
    roi.y = (mat.size().height - short_edge) / 2;
    roi.width = roi.height = short_edge;
    cv::Mat cropped = mat(roi);
    cv::Mat resized;
    cv::resize(cropped, resized, size);
    return resized;
}

auto reorder_to_nchw(cv::Mat const& mat) {
    assert(mat.channels() == 3);
    std::vector<float> data(mat.channels() * mat.rows * mat.cols);
    for(int y = 0; y < mat.rows; ++y) {
        for(int x = 0; x < mat.cols; ++x) {
            // cv::imread loads image BGR order so reordering here to RGB
            for(int c = mat.channels() - 1; c >= 0; --c) {
                data[c * (mat.rows * mat.cols) + y * mat.cols + x] =
                  static_cast<float>(
                    mat.data[y * mat.step + x * mat.elemSize() + c]);
            }
        }
    }
    return data;
}

template <typename InIter>
auto extract_top_k_index_list(
  InIter first, InIter last,
  typename std::iterator_traits<InIter>::difference_type k) {
    using diff_t = typename std::iterator_traits<InIter>::difference_type;
    std::priority_queue<
      std::pair<typename std::iterator_traits<InIter>::value_type, diff_t>>
      q;
    for(diff_t i = 0; first != last; ++first, ++i) {
        q.push({*first, i});
    }
    std::vector<diff_t> indices;
    for(diff_t i = 0; i < k; ++i) {
        indices.push_back(q.top().second);
        q.pop();
    }
    return indices;
}

auto load_category_list(std::string const& synset_words_path) {
    std::ifstream ifs(synset_words_path);
    if(!ifs) {
        throw std::runtime_error("File open error: " + synset_words_path);
    }
    std::vector<std::string> categories;
    std::string line;
    while(std::getline(ifs, line)) {
        categories.push_back(std::move(line));
    }
    return categories;
}

int main(int argc, char** argv) {
    std::cout << "vgg16 example" << std::endl;

    constexpr auto batch_size = 1;
    constexpr auto channel_num = 3;
    constexpr auto height = 224;
    constexpr auto width = 224;

    std::vector<int> input_dims{batch_size, channel_num, height, width};

    cmdline::parser a;
    a.add<std::string>("input_image", 'i', "input image path", false,
                       "../data/Light_sussex_hen.jpg");
    a.add<std::string>("model", 'm', "onnx model path", false,
                       "../data/VGG16.onnx");
    a.add<std::string>("synset_words", 's', "synset words path", false,
                       "../data/synset_words.txt");
    a.parse_check(argc, argv);

    auto input_image_path = a.get<std::string>("input_image");
    auto onnx_model_path = a.get<std::string>("model");
    auto synset_words_path = a.get<std::string>("synset_words");

    cv::Mat image_mat =
      cv::imread(input_image_path.c_str(), CV_LOAD_IMAGE_COLOR);
    if(!image_mat.data) {
        throw std::runtime_error("Invalid input image path: " +
                                 input_image_path);
    }
    image_mat = crop_and_resize(std::move(image_mat), cv::Size(width, height));
    auto image_data = reorder_to_nchw(image_mat);

    // Alias to onnx's node input and output tensor name
    auto conv1_1_in_name = "140326425860192";
    auto fc6_out_name = "140326200777976";
    auto softmax_out_name = "140326200803680";

    // Load ONNX model
    auto onnx_model = instant::load_onnx(onnx_model_path);

    // Construct computation primitive list and memories
    auto model = instant::make_model(
      onnx_model,
      {std::make_tuple(conv1_1_in_name, instant::dtype_t::float_, input_dims,
        mkldnn::memory::format::nchw)},  // input's (name, dtype, dims, format)
                                         // list
      {fc6_out_name, softmax_out_name}); // required output's name list

    // Copy input image data to model's input array
    auto& input_array = model.input(conv1_1_in_name);
    std::copy(image_data.begin(), image_data.end(),
              instant::fbegin(input_array));

    // Run inference
    auto const& output_table = model.run();

    // Get output
    auto const& fc6_out_arr = instant::find_value(output_table, fc6_out_name);
    std::cout << "fc6_out: ";
    for(int i = 0; i < 5; ++i) {
        std::cout << instant::fat(fc6_out_arr, i) << " ";
    }
    std::cout << "...\n";

    auto const& softmax_out_arr =
      instant::find_value(output_table, softmax_out_name);

    auto categories = load_category_list(synset_words_path);
    auto top_k = 5;
    auto top_k_indices = extract_top_k_index_list(
      instant::fbegin(softmax_out_arr), instant::fend(softmax_out_arr), top_k);
    std::cout << "top " << top_k << " categories are\n";
    for(auto ki : top_k_indices) {
        std::cout << ki << " " << instant::fat(softmax_out_arr, ki) << " "
                  << categories.at(ki) << std::endl;
    }
}
