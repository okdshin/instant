#include <iostream>
#include <queue>
#include <vector>
#include <opencv2/opencv.hpp>

#include <instant/array.hpp>
#include <instant/model.hpp>

#include "cmdline.h"

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
    std::unique_ptr<float[]> data(
        new float[mat.channels() * mat.rows * mat.cols]);
    for (int y = 0; y < mat.rows; ++y) {
        for (int x = 0; x < mat.cols; ++x) {
            // cv::imread loads image BGR order so reordering here to RGB
            for (int c = mat.channels() - 1; c >= 0; --c) {
                *(data.get() + (c * (mat.rows * mat.cols) + y * mat.cols + x)) =
                    static_cast<float>(
                        mat.data[y * mat.step + x * mat.elemSize() + c]);
            }
        }
    }
    return data;
}

template <typename InIter>
auto extract_top_k_index_list(InIter first, InIter last,
                 typename std::iterator_traits<InIter>::difference_type k) {
    using diff_t = typename std::iterator_traits<InIter>::difference_type;
    std::priority_queue<
        std::pair<typename std::iterator_traits<InIter>::value_type, diff_t>>
        q;
    for (diff_t i = 0; first != last; ++first, ++i) {
        q.push({*first, i});
    }
    std::vector<diff_t> indices;
    for (diff_t i = 0; i < k; ++i) {
        indices.push_back(q.top().second);
        q.pop();
    }
    return indices;
}

auto load_category_list(std::string const& synset_words_path) {
    std::ifstream ifs(synset_words_path);
    if(!ifs) {
        throw std::runtime_error("File open error: "+synset_words_path);
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

    cmdline::parser a;
    a.add<std::string>("input_image", 'i', "input image path", false, "../data/Light_sussex_hen.jpg");
    a.add<std::string>("model", 'm', "onnx model path", false, "../data/VGG16.onnx");
    a.add<std::string>("synset_words", 's', "synset words path", false, "../data/synset_words.txt");
    a.parse_check(argc, argv);

    auto input_image_path = a.get<std::string>("input_image");
    auto onnx_model_path = a.get<std::string>("model");
    auto synset_words_path = a.get<std::string>("synset_words");

    cv::Mat image_mat = cv::imread(input_image_path.c_str(), CV_LOAD_IMAGE_COLOR);
    if(!image_mat.data) {
        throw std::runtime_error("Invalid input image path: "+input_image_path);
    }
    image_mat = crop_and_resize(std::move(image_mat), cv::Size(width, height));
    instant::array input_image(instant::dtype_t::float_,
                               {batch_size, channel_num, height, width},
                               reorder_to_nchw(image_mat));
    assert(input_image.dtype() == instant::dtype_t::float_);

    auto onnx_model = instant::load_onnx(onnx_model_path);

    std::cout << "model input and output" << std::endl;
    for (auto const& node : onnx_model.graph().node()) {
        std::cout << node.op_type() << "\t";
        for (auto const& i : node.input()) {
            std::cout << i << " ";
        }
        std::cout << "-> ";
        for (auto const& i : node.output()) {
            std::cout << i << " ";
        }
        std::cout << "\n";
    }

    auto parameter_table = instant::make_parameter_table(onnx_model.graph());
    auto parameter_memory_table = instant::make_parameter_memory_table(
        onnx_model.graph(), parameter_table, ::instant::get_context().engine());

    auto conv1_1_in_name = "140326425860192";
    std::vector<std::tuple<std::string, instant::array, mkldnn::memory::format>>
        input_list{
            {conv1_1_in_name, input_image, mkldnn::memory::format::nchw}};
    auto variable_memory_table = instant::make_variable_memory_table(
        input_list, ::instant::get_context().engine());

    auto fc6_out_name = "140326200777976";
    auto softmax_out_name = "140326200803680";
    std::vector<std::string> required_output_name_list{fc6_out_name,
                                                       softmax_out_name};
    auto output_table = instant::run_model(
        onnx_model.graph(), parameter_memory_table, variable_memory_table,
        std::set<std::string>(required_output_name_list.begin(),
                              required_output_name_list.end()));
    for (auto const& required_output_name : required_output_name_list) {
        std::cout << required_output_name << ": ";
        for (int i = 0; i < 5; ++i) {
            std::cout << *(static_cast<float*>(
                               output_table[required_output_name].data()) +
                           i);
        }
        std::cout << "...\n";
    }

    auto const& softmax_out = output_table[softmax_out_name];

    auto categories = load_category_list(synset_words_path);
    auto top_k = 5;
    auto top_k_indices = extract_top_k_index_list(instant::fbegin(softmax_out), instant::fend(softmax_out), top_k);
    for(auto ki : top_k_indices) {
        std::cout << ki << " " << instant::fat(softmax_out, ki) << " " << categories.at(ki) << std::endl;
    }
}
