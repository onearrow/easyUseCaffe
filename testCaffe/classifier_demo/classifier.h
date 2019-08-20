#include <caffe/caffe.hpp>
#ifdef USE_OPENCV  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#endif  
// USE_OPENCV  
#include <algorithm>  
#include <iosfwd>  
#include <memory>  
#include <string>  
#include <utility>  
#include <vector>  

#ifdef USE_OPENCV  
using namespace caffe; 
// NOLINT(build/namespaces)  
using std::string; 
/* Pair (label, confidence) representing a prediction. */ 
typedef std::pair<string, float> Prediction;//记录每一个类的名称以及概率  

//Classifier为构造函数，主要进行模型初始化，读入训练完毕的模型参数，均值文件和标签文件  
class Classifier 
{
public: 
    Classifier(const string& model_file, //model_file为测试模型时记录网络结构的prototxt文件路径  
    const string& trained_file, //trained_file为训练完毕的caffemodel文件路径  
    const string& mean_file, //mean_file为记录数据集均值的文件路径，数据集均值的文件的格式通常为binaryproto  
    const string& label_file); //label_file为记录类别标签的文件路径，标签通常记录在一个txt文件中，一行一个  

    std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);//Classify函数去进行网络前传，得到img属于各个类的概率  
private:
    void SetMean(const string& mean_file);//SetMean函数主要进行均值设定，每张检测图输入后会进行减去均值的操作，这个均值可以是模型使用的数据集图像的均值  
    std::vector<float> Predict(const cv::Mat& img);//Predict函数是Classify函数的主要组成部分，将img送入网络进行前向传播，得到最后的类别  
    void WrapInputLayer(std::vector<cv::Mat>* input_channels);//WrapInputLayer函数将img各通道(input_channels)放入网络的输入blob中  
    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);//Preprocess函数将输入图像img按通道分开(input_channels)  

private: 
    shared_ptr<Net<float> > net_; //net_表示caffe中的网络  
    cv::Size input_geometry_; //input_geometry_表示了输入图像的高宽，同时也是网络数据层中单通道图像的高宽  
    int num_channels_; //num_channels_表示了输入图像的通道数  
    cv::Mat mean_; //mean_表示了数据集的均值，格式为Mat  
    std::vector<string> labels_; //字符串向量labels_表示了各个标签
}; 