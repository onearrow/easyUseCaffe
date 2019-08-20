#include<classifier.h>

//构造函数Classifier进行了各种各样的初始化工作，并对网络的安全进行了检验  
Classifier::Classifier(const string& model_file,
                        const string& trained_file, 
                        const string& mean_file, 
                        const string& label_file)
{ 
#ifdef CPU_ONLY  
    Caffe::set_mode(Caffe::CPU);//如果caffe是只在cpu上运行的，将运行模式设置为CPU  
#else  
    Caffe::set_mode(Caffe::GPU);//一般我们都是用的GPU模式  
#endif  

    /* Load the network. */ 
    net_.reset(new Net<float>(model_file, TEST));//从model_file路径下的prototxt初始化网络结构  
    net_->CopyTrainedLayersFrom(trained_file);//从trained_file路径下的caffemodel文件读入训练完毕的网络参数  
    
    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";//核验是不是只输入了一张图像，输入的blob结构为(N,C,H,W)，在这里，N只能为1  
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";//核验输出的blob结构，输出的blob结构同样为(N,C,W,H)，在这里，N同样只能为1  

    Blob<float>* input_layer = net_->input_blobs()[0];//获取网络输入的blob，表示网络的数据层  
    num_channels_ = input_layer->channels();//获取输入的通道数  

    //核验输入图像的通道数是否为3或者1，网络只接收3通道或1通道的图片  
    CHECK(num_channels_ == 3 || num_channels_ == 1)<< "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());//获取输入图像的尺寸(宽与高)  

    /* Load the binaryproto mean file. */ 
    SetMean(mean_file);//进行均值的设置  
    
    /* Load labels. */ 
    std::ifstream labels(label_file.c_str());
    
    //从标签文件路径读入定义的标签文件  
    CHECK(labels) << "Unable to open labels file " << label_file; 
    
    string line;//line获取标签文件中的每一行(每一个标签)  
    while (std::getline(labels, line)) 
        labels_.push_back(string(line));
        
    //将所有的标签放入labels_  
    /*output_layer指向网络最后的输出，举个例子，最后的分类器采用softmax分类，且类别有10类，那么，输出的blob就会有10个通道，每个通道的长 
  宽都为1(因为是10个数，这10个数表征输入属于10类中每一类的概率，这10个数之和应该为1)，输出blob的结构为(1,10,1,1)*/ 
  
    Blob<float>* output_layer = net_->output_blobs()[0]; //在这里核验最后网络输出的通道数是否等于定义的标签的通道数  
    CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
} 

static bool PairCompare(const std::pair<float, int>& lhs, 
                        const std::pair<float, int>& rhs) 
{ 
    return lhs.first > rhs.first; 
}

//PairCompare函数比较分类得到的物体属于某两个类别的概率的大小，若属于lhs的概率大于属于rhs的概率，返回真，否则返回假  
/* Return the indices of the top N values of vector v. */ 

/*Argmax函数返回前N个得分概率的类标*/ 
static std::vector<int> Argmax(const std::vector<float>& v, int N) 
{ 
    std::vector<std::pair<float, int> > pairs; 
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
        
    //按照分类结果存储输入属于每一个类的概率以及类标  
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);
    /*partial_sort函数按照概率大小筛选出pairs中概率最大的N个组合，并将它们按照概率从大到小放在pairs的前N个位置*/ 
    
    //将前N个较大的概率对应的类标放在result中
    std::vector<int> result; 
    for (int i = 0; i < N; ++i) 
        result.push_back(pairs[i].second);
    return result;
} 

/* Return the top N predictions. */ 

std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) 
{ 
    //进行网络的前向传输，得到输入属于每一类的概率，存储在output中
    std::vector<float> output = Predict(img);
    //找到想要得到的概率较大的前N类，这个N应该小于等于总的类别数目  
    N = std::min<int>(labels_.size(), N);
    //找到概率最大的前N类，将他们按概率由大到小将类标存储在maxN中
    std::vector<int> maxN = Argmax(output, N);  
    std::vector<Prediction> predictions; 
    
    for (int i = 0; i < N; ++i) 
    { 
        int idx = maxN[i]; 
        predictions.push_back(std::make_pair(labels_[idx], output[idx]));
        //在labels_找到分类得到的概率最大的N类对应的实际的名称  
    } 
    return predictions; 
} 

/* Load the mean file in binaryproto format. */ 
void Classifier::SetMean(const string& mean_file) 
{
    //设置数据集的平均值  
    BlobProto blob_proto; 
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    
    //用定义的均值文件路径将均值文件读入proto中  
    /* Convert from BlobProto to Blob<float> */ 
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    
    //将proto中存储的均值文件转移到blob中  
    //核验均值的通道数是否等于输入图像的通道数，如果不相等的话则为异常
    CHECK_EQ(mean_blob.channels(), num_channels_)<< "Number of channels of mean file doesn't match input layer.";
    
    /* The format of the mean file is planar 32-bit float BGR or grayscale. */ 
    std::vector<cv::Mat> channels;
    
    //将mean_blob中的数据转化为Mat时的存储向量  
    float* data = mean_blob.mutable_cpu_data();
    
    //指向均值blob的指针  
    for (int i = 0; i < num_channels_; ++i) 
    { 
        /* Extract an individual channel. */ 
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        //存储均值文件的每一个通道转化得到的
        Mat channels.push_back(channel);
        
        //将均值文件的所有通道转化成的Mat一个一个地存储到channels中  
        data += mean_blob.height() * mean_blob.width();
        //在均值文件上移动一个通道  
    } 
    
    /* Merge the separate channels into a single image. */ 
    cv::Mat mean; 
    cv::merge(channels, mean);//将得到的所有通道合成为一张图  
    
    /* Compute the global mean pixel value and create a mean image * filled with this value. */ 
    cv::Scalar channel_mean = cv::mean(mean);//求得均值文件的每个通道的平均值，记录在channel_mean中  
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);//用上面求得的各个通道的平均值初始化mean_，作为数据集图像的均值  
} 
    
std::vector<float> Classifier::Predict(const cv::Mat& img)
{ 
    Blob<float>* input_layer = net_->input_blobs()[0];//input_layer是网络的输入blob  
    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);//表示网络只输入一张图像，图像的通道数是num_channels_，高为input_geometry_.height，宽为input_geometry_.width  
    
    /* Forward dimension change to all layers. */ 
    net_->Reshape();//初始化网络的各层  
    
    std::vector<cv::Mat> input_channels;
    //存储输入图像的各个通道  
    WrapInputLayer(&input_channels);//将存储输入图像的各个通道的input_channels放入网络的输入blob中  
    
    Preprocess(img, &input_channels);//将img的各通道分开并存储在input_channels中  
    
    net_->Forward();//进行网络的前向传输  
    
    /* Copy the output layer to a std::vector */ 
    
    Blob<float>* output_layer = net_->output_blobs()[0];//output_layer指向网络输出的数据，存储网络输出数据的blob的规格是(1,c,1,1)  
    const float* begin = output_layer->cpu_data();//begin指向输入数据对应的第一类的概率  
    const float* end = begin + output_layer->channels();//end指向输入数据对应的最后一类的概率  
    return std::vector<float>(begin, end);//返回输入数据经过网络前向计算后输出的对应于各个类的分数  
} 

/* Wrap the input layer of the network in separate cv::Mat objects 
 * (one per channel). This way we save one memcpy operation and we 
 * don't need to rely on cudaMemcpy2D. The last preprocessing 
 * operation will write the separate channels directly to the input 
 * layer. */ 

void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) 
{ 
    Blob<float>* input_layer = net_->input_blobs()[0];
    //input_layer指向网络输入的blob  
    
    int width = input_layer->width();//得到网络指定的输入图像的宽  
    int height = input_layer->height();//得到网络指定的输入图像的高  
    float* input_data = input_layer->mutable_cpu_data();//input_data指向网络的输入blob  
    for (int i = 0; i < input_layer->channels(); ++i) 
    { 
        cv::Mat channel(height, width, CV_32FC1, input_data);//将网络输入blob的数据同Mat关联起来  
        input_channels->push_back(channel);//将上面的Mat同input_channels关联起来  
        input_data += width * height;//一个一个通道地操作  
    } 
} 

void Classifier::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) 
{ /* Convert the input image to the input image format of the network. */ 
    cv::Mat sample; 
    if(img.channels() == 3 && num_channels_ == 1) 
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY); 
    else if (img.channels() == 4 && num_channels_ == 1) 
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY); 
    else if (img.channels() == 4 && num_channels_ == 3) 
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR); 
    else if (img.channels() == 1 && num_channels_ == 3) 
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR); 
    else sample = img;
    
    //if-else嵌套表示了要将输入的img转化为num_channels_通道的  
    cv::Mat sample_resized; 
    if (sample.size() != input_geometry_) 
        cv::resize(sample, sample_resized, input_geometry_); 
        //将输入图像的尺寸强制转化为网络规定的输入尺寸  
    else 
        sample_resized = sample; 

    cv::Mat sample_float; 
        
    if (num_channels_ == 3) 
        sample_resized.convertTo(sample_float, CV_32FC3); 
    else 
        sample_resized.convertTo(sample_float, CV_32FC1);//将输入图像转化成为网络前传合法的数据规格  
        
    cv::Mat sample_normalized; 
    cv::subtract(sample_float, mean_, sample_normalized);//将图像减去均值  
    
    /* This operation will write the separate BGR planes directly to the 
   * input layer of the network because it is wrapped by the cv::Mat 
   * objects in input_channels. */ 
  
    cv::split(sample_normalized, *input_channels);
    /*将减去均值的图像分散在input_channels中，由于在WrapInputLayer函数中，input_channels已经和网络的输入blob关联起来了，因此在这里实际上是把图像送入了网络的输入blob*/ 
    CHECK(reinterpret_cast<float*>(input_channels->at(0).data) 
        == net_->input_blobs()[0]->cpu_data()) << "Input channels are not wrapping the input layer of the network.";//核验图像是否被送入了网络作为输入  
        
} 

int main(int argc, char** argv) 
{//主函数  
    if (argc != 6) {
        /*核验命令行参数是否为6，这6个参数分别为classification编译生成的可执行文件，测试模型时记录网络结构的prototxt文件路径，训练完毕的caffemodel文件路径，记录数据集均值的文件路径，记录类别标签的文件路径，需要送入网络进行分类的图片文件路径*/ 
        std::cerr << "Usage: " << argv[0] << " deploy.prototxt network.caffemodel"
        << " mean.binaryproto labels.txt img.jpg" << std::endl;
        return 1; 
    } 
    ::google::InitGoogleLogging(argv[0]);
    //InitGoogleLogging做了一些初始化glog的工作  
    
    //取四个参数  
    string model_file = argv[1]; 
    string trained_file = argv[2]; 
    string mean_file = argv[3]; 
    string label_file = argv[4]; 
    
    Classifier classifier(model_file, trained_file, mean_file, label_file);//进行检测网络的初始化  
    string file = argv[5];//取得需要进行检测的图片的路径  
    std::cout << "---------- Prediction for " << file 
    << " ----------" << std::endl; 
    
    cv::Mat img = cv::imread(file, -1);
    //读入图片  
    
    CHECK(!img.empty()) << "Unable to decode image " 
    << file; std::vector<Prediction> predictions = classifier.Classify(img);
    
    //进行网络的前向计算，并且取到概率最大的前N类对应的类别名称  
    
    /* Print the top N predictions. */ 
    
    for (size_t i = 0; i < predictions.size(); ++i) 
    {
        //打印出概率最大的前N类并给出概率  
        Prediction p = predictions[i]; 
        std::cout << std::fixed << std::setprecision(4) << p.second 
        << " - \"" << p.first << "\"" << std::endl; 
    } 
} 

#else  
int main(int argc, char** argv) 
{ 
    LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV."; 
}
#endif  
// USE_OPENCV  
