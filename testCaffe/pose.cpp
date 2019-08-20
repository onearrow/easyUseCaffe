/*-------------------------------------------------------- 
作者：hanss2 
来源：CSDN 
原文：https://blog.csdn.net/hanss2/article/details/78927225 
版权声明：本文为博主原创文章，转载请附上博文链接！
-----------------------------------------------------------*/

#include <caffe/caffe.hpp>
#include <caffe/blob.hpp>
#include <caffe/common.hpp>
#include <caffe/net.hpp>
#include <caffe/proto/caffe.pb.h>
#include <caffe/util/io.hpp>

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <algorithm>  
#include <iosfwd>
#include <iostream>
#include <iomanip>
#include <memory>  
#include <string>  
#include <utility>  
#include <vector>  

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

using namespace caffe;
using std::string;

#define PRINT_DATA(x) \
    std::cout << (x)[0] << "\t" << (x)[1] << "\t"<< (x)[2] << "\t"<< (x)[3] << "\t"<< (x)[4] << "\n";
void caffe_forward(boost::shared_ptr< Net<float>> & net, float *data_ptr);
void file_operator(unsigned int index,const float* ldata,unsigned int data_sum);

void WrapInputLayer(shared_ptr< Net<float>> & net,std::vector<cv::Mat>* input_channels){
    Blob<float>* input_layer = net->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    //std::cout<< "width:" << width <<std::endl;
    //std::cout<< "height:" << height <<std::endl;

    float* input_data = input_layer->mutable_cpu_data();

    for (int i = 0; i < input_layer->channels(); ++i) 
    {
      cv::Mat channel(height, width, CV_32FC1, input_data);
      input_channels->push_back(channel);
      input_data += width * height;
    }
}

void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels, int num_channels_) 
{
	/* Convert the input image to the input image format of the network. */
  //std::cout << "img.channels() ="<<img.channels()<<std::endl;
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
    
    //cv::Mat sample_resized;
    //if-else嵌套表示了要将输入的img转化为num_channels_通道的
    //cv::Mat sample_resized= cv::imread("test.png",-1); 
    //sample = cv::imread("test.png"); 
    //if (sample.size() != input_geometry_)
    //将输入图像的尺寸强制转化为网络规定的输入尺寸
    //cv::resize(sample, sample_resized, cv::Size(224, 224));

    /*float data_temp[150528];
    for (int h = 0; h < 150528; ++h)
    {
      data_temp[h] = sample_resized.data[h];
    }
    file_operator(500,data_temp,150528);*/
    //cv::imwrite("COresize.jpg",sample_resized);

    cv::Mat sample_float;     
    if (num_channels_ == 3) 
        sample.convertTo(sample_float, CV_32FC3); 
    else 
        sample.convertTo(sample_float, CV_32FC1);//将输入图像转化成为网络前传合法的数据规格  
    
    //cv::imwrite("resize.jpg",sample_float);
    //cv::Mat sample_normalized; 
    //cv::subtract(sample_float, mean_, sample_normalized);//将图像减去均值  
    
    /* This operation will write the separate BGR planes directly to the 
   * input layer of the network because it is wrapped by the cv::Mat 
   * objects in input_channels. */ 

    /*将减去均值的图像分散在input_channels中，由于在WrapInputLayer函数中，
    input_channels已经和网络的输入blob关联起来了，因此在这里实际上是把图像送入了网络的输入blob*/ 
    cv::split(sample_float, *input_channels);

    //核验图像是否被送入了网络作为输入 
//    CHECK(reinterpret_cast<float*>(input_channels->at(0).data) 
//       == net_->input_blobs()[0]->cpu_data()) << "Input channels are not wrapping the input layer of the network.";      
} 

//### 此函数只有两个参数，训练的时候使用meanvalue的可以使用这个函数
template<typename Dtype>
void CvMat2blobVec(const vector<cv::Mat>& srcs, vector<Blob<Dtype>* >& transformed_blob_vec)
{
	// get size
	const int num = transformed_blob_vec[0]->num();
	const int height = transformed_blob_vec[0]->height();
	const int width  = transformed_blob_vec[0]->width();
	const int channel  = transformed_blob_vec[0]->channels();
	CHECK_EQ(num, srcs.size());
	// mean value

	//### 下面定义的mean_values存放的是训练的时候的meanvalue的值，根据自己的情况决定
	vector<Dtype> mean_values;
	for(int i = 0 ; i< 3; i++)
		mean_values.push_back(0);

	Dtype* transformed_data = transformed_blob_vec[0]->mutable_cpu_data();
	for(int n=0; n<num; n++){
		// resize
		cv::Mat cv_img;
		cv::resize(srcs[n], cv_img, cv::Size(width, height)); 
		// transform into blob
		int num_offset = transformed_blob_vec[0]->offset(n, 0, 0, 0);
		int top_index;
		for (int h = 0; h < height; ++h) {
			const uchar* ptr = cv_img.ptr<uchar>(h);
			int img_index = 0;
			for (int w = 0; w < width; ++w) {
				for (int c = 0; c < channel; ++c) {
					top_index = (c * height + h) * width + w;
					Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
					transformed_data[num_offset+top_index] = (pixel - mean_values[c]);
				}
			}
		}
	}
}

unsigned int get_blob_index(boost::shared_ptr<Net<float>> & net, std::string query_blob_name)
{
  std::string str_query(query_blob_name);
  vector< string > const & blob_names = net->blob_names();

  for(unsigned int i = 0; i != blob_names.size(); ++i )
  {
    //std::cout<< "--->"<< blob_names[i]<<std::endl;
    if( str_query == blob_names[i] )
      {
        return i;
      }
  }
  LOG(FATAL) << "Unknown blob name: " << str_query;
  return -1;
}


void query_Layer_data(boost::shared_ptr<Net<float>> & net,std::string lname)
{
  unsigned int layer_id = get_blob_index(net,lname);

  const float *weight_ptr;
  const float *bias_ptr;

  boost::shared_ptr<Layer<float>> l = net->layers()[layer_id];

  std::cout<<"The "<<layer_id<<" layer name: "<<"-->"<<net->blob_names()[layer_id]<<":"<<std::endl;
  if((l->blobs().size()>0) && (l->type() == "Convolution"))
  {
    boost::shared_ptr<Blob<float>> blob =l->blobs()[0];
    weight_ptr = (const float*)(blob->mutable_cpu_data());
    file_operator(layer_id,weight_ptr,blob->count());
    if(l->blobs().size()>1)
    {
      boost::shared_ptr<Blob<float>> blob1 =l->blobs()[1];
      bias_ptr = (const float*)(blob1->mutable_cpu_data());
      //file_operator(layer_id,bias_ptr,blob1->count());
    }
  }

  /*float fdata;
  for(int i=0;i<blob->count();i++)
  {
    fdata = blob->cpu_data()[i];
    printf("%f\t",fdata);
    if(0==(i+1)%5)
      printf("\n");
  }*/
}

void readNetParam(const char *model)
{
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(model, &param);

  WriteProtoToTextFile(param,"pose.txt");
  /*int num_layers = param.layer_size();

  for (int i = 0; i < num_layers; ++i)
  {
  // 结构配置参数:name，type，kernel size，pad，stride等
    std::cout << "Layer " << i << ":" << param.layer(i).name() << "   " << param.layer(i).type();
    if (param.layer(i).type() == "Convolution")
    {
      ConvolutionParameter conv_param = param.layer(i).convolution_param();
      //uint ckname = conv_param.kernel_size();
      //std::cout << "kernel size: " << (unsigned int)(conv_param.kernel_size())<< ", pad: " << (unsigned int)(conv_param.pad())<< ", stride: " << (unsigned int)(conv_param.stride());
    }
  }*/
}



//### 此函数有三个参数，训练的时候使用meanfile的可以使用这个函数
// override
/*template<typename Dtype>
void CvMat2blobVec(const vector<cv::Mat>& srcs, vector<Blob<Dtype>* >& transformed_blob_vec,
  string meanfilePath)
{
  // get size
  const int num = transformed_blob_vec[0]->num();
  const int height = transformed_blob_vec[0]->height();
  const int width  = transformed_blob_vec[0]->width();
  const int channel  = transformed_blob_vec[0]->channels();
  CHECK_EQ(num, srcs.size());
  //### meanfile
 
  Blob<Dtype> data_mean_;
  //### meanfile to blob_proto
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(meanfilePath.c_str(),&blob_proto);
  //### blob_proto to data_mean_ 
  data_mean_.FromProto(blob_proto);
 
  //### get meanfile value
  Dtype* mean =NULL;
  CHECK_EQ(channel, data_mean_.channels());
  CHECK_EQ(height, data_mean_.height());
  CHECK_EQ(width, data_mean_.width());
  mean = data_mean_.mutable_cpu_data();
 
 
  Dtype* transformed_data = transformed_blob_vec[0]->mutable_cpu_data();
  for(int n=0; n<num; n++){
    // resize
    cv::Mat cv_img;
    cv::resize(srcs[n], cv_img, cv::Size(width, height)); 
    // transform into blob
    int num_offset = transformed_blob_vec[0]->offset(n, 0, 0, 0);
    int top_index;
    for (int h = 0; h < height; ++h) {
      const uchar* ptr = cv_img.ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < channel; ++c) {
            top_index = (c * height + h) * width + w;   
            Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
            transformed_data[num_offset+top_index] = (pixel - mean[num_offset+top_index]);
        }
      }
    }
  } 
}*/

/*copyLayerParam(const char* trained_file)
{

  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(trained_file, &param);
  
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
  const LayerParameter& source_layer = param.layer(i);
  const string& source_layer_name = source_layer.name();
  int target_layer_id = 0;
  while (target_layer_id != layer_names_.size() &&
    layer_names_[target_layer_id] != source_layer_name) {
    ++target_layer_id;
  }
  if (target_layer_id == layer_names_.size()) {
    LOG(INFO) << "Ignoring source layer " << source_layer_name;
    continue;
  }

  DLOG(INFO) << "Copying source layer " << source_layer_name;
  vector<shared_ptr<Blob<Dtype> > >& target_blobs =
    layers_[target_layer_id]->blobs();
  CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
      << "Incompatible number of blobs for layer " << source_layer_name;
  for (int j = 0; j < target_blobs.size(); ++j) {
  if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
    Blob<Dtype> source_blob;
    const bool kReshape = true;
    source_blob.FromProto(source_layer.blobs(j), kReshape);
    LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
            << source_layer_name << "'; shape mismatch.  Source param shape is "
            << source_blob.shape_string() << "; target param shape is "
            << target_blobs[j]->shape_string() << ". "
            << "To learn this layer's parameters from scratch rather than "
            << "copying from a saved net, rename the layer.";
    }
    const bool kReshape = false;
    target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
  }
  }
}*/

int main()
{
	Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(0);

	//const string model_file = "/home/hup/project/openpose/models/pose/body_25/pose_deploy.prototxt";
	//const char* trained_file ="/home/hup/project/openpose/models/pose/body_25/pose_iter_584000.caffemodel";
	//const string img_file="/home/hup/project/openpose/caffe/testPose/COCO.jpg";

  const string model_file = "/home/hup/project/ReID/reidS/pytorch2caffe_MGN/PED_EXT_006.prototxt";
  const char* trained_file ="/home/hup/project/ReID/reidS/pytorch2caffe_MGN/PED_EXT_006.caffemodel";
  const string img_file="/home/hup/project/openpose/caffe/testPose/pos_f.jpg";

  //const string model_file = "/home/hup/project/ReID/pytorch2caffe_MGN/agender004.prototxt";
  //const char* trained_file ="/home/hup/project/ReID/pytorch2caffe_MGN/agender004.caffemodel";
  //const string img_file="test224.jpg";

	shared_ptr<Net<float>> net;
	net.reset(new caffe::Net<float>(model_file,TEST));
	net->CopyTrainedLayersFrom(trained_file);


  //Blob<float>* input_layer = net->input_blobs()[0];
  //input_layer->Reshape(1, 3, 656, 368);
  net->Reshape();

	//核验是不是只输入了一张图像，输入的blob结构为(N,C,H,W)，在这里，N只能为1  
	CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly one input.";
	//核验输出的blob结构，输出的blob结构同样为(N,C,W,H)，在这里，N同样只能为1  
	//CHECK_EQ(net->num_outputs(), 1) << "Network should have exactly one output.";

  unsigned int index = get_blob_index(net,"cat1");
  unsigned int index1 = get_blob_index(net,"conv1");
  unsigned int index2 = get_blob_index(net,"blob1");
  std::cout<<"conv1_1 "<<index1<<" conv1_2: "<<index2 <<std::endl;
	
	cv::Mat img = cv::imread(img_file,-1);
	CHECK(!img.empty()) << "Unable to decode image " << img_file;

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(net, &input_channels);

	Preprocess(img, &input_channels, 3);

	net->Forward();
  //readNetParam(trained_file);
	//PRINT_DATA((const float *)(net->blobs()[0]->cpu_data()));

  string strl[] = {"conv49","batch_norm49","relu45","conv70","batch_norm70","relu66","conv95","batch_norm95","relu90"}; //,"bn_scale49""bn_scale70","bn_scale95",
  int strCount = sizeof(strl)/sizeof(strl[1]);
  for(int i=0;i<strCount;i++)
  {
    //query_Layer_data(net,strl[i]);
  }
   
  for(int i=0;i<index+1;i++)
  {
      boost::shared_ptr<Blob<float> > blob = net->blobs()[i];
      unsigned int num_data = blob->count();
      
      if(Caffe::mode()==Caffe::GPU)
      {
        unsigned int blobid = get_blob_index(net,net->blob_names()[i]);
        //std::cout<<i<<":The "<<blobid<<" layer data size: "<<num_data<<"-->"<<net->blob_names()[i]<<":"<<blob->height()<<"X"<<blob->width()<<"X"<< blob->channels()<<std::endl;

        float* data_ptr = new float[num_data];
        const float *blob_ptr = (const float *)blob->mutable_gpu_data();
        caffe_copy(blob->count(), blob_ptr, data_ptr);

          //double* data_ptr = new double[num_data];
          //const double *blob_ptr = (const double *)blob->mutable_gpu_data();
          //caffe_copy(blob->count(), blob_ptr, data_ptr);
          //PRINT_DATA(blob_ptr);
          //std::cout<<"The "<<i<<" layer data size: "<<num_data<<":"<<blob->height()<<"X"<<blob->width()<<"X"<< blob->channels()<<std::endl;
          //query_Layer_data(net,net->blob_names()[i]);

        file_operator(i,data_ptr,num_data);
        delete data_ptr;
      }
      else if(Caffe::mode()==Caffe::CPU)
      {
        const float *blob_ptr = (const float *)blob->mutable_cpu_data();
        //std::cout<<"The "<<i<<" layer data size: "<<num_data<<std::endl;
        std::cout<<"The "<<i<<" layer data size: "<<num_data<<"-->"<<net->blob_names()[i]<<":"<<blob->height()<<"X"<<blob->width()<<"X"<< blob->channels()<<std::endl;

        file_operator(i,blob_ptr,num_data);
      }
  }
	return 0;
}


//! Note: data_ptr指向已经处理好（去均值的，符合网络输入图像的长宽和Batch Size）的数据
void caffe_forward(boost::shared_ptr< Net<float>> & net,float *data_ptr)
{
    Blob<float>* input_blobs = net->input_blobs()[0];
    switch (Caffe::mode())
    {
    case Caffe::CPU:
        memcpy(input_blobs->mutable_cpu_data(), data_ptr,
            sizeof(float) * input_blobs->count());
        break;
    case Caffe::GPU:
        //cudaMemcpy(input_blobs->mutable_gpu_data(), data_ptr,	sizeof(float) * input_blobs->count(), cudaMemcpyHostToDevice);
        break;
    default:
        LOG(FATAL) << "Unknown Caffe mode.";
    } 
    net->ForwardPrefilled();
}

void file_operator(unsigned int index,const float* ldata,unsigned int data_sum)
{
  char cfname[30];
  sprintf(cfname,"./reid/dat%d.txt",index);

  if(NULL==ldata)
  {
    printf("The layer is NULL!\n");
    return;
  }
  std::ofstream f_out(cfname);
  f_out.setf(ios::fixed, ios::floatfield);
  f_out.precision(5);
  if(f_out.is_open())
  {
    const float* f_data=ldata;
    int i=0;
    while(i<data_sum)
    {
      if(0==i)
        f_out<<"---------"<<cfname<<":"<<data_sum<<"-----------\n";
      f_out<<f_data[i]<<std::endl; //setiosflags(ios::fixed)<<
      i++;
    }
    f_out.close();
  }
}