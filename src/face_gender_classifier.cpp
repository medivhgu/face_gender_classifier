#include <cstring>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "face_gender_classifier.hpp"
#include "caffe/caffe.hpp"

using namespace std;
using namespace caffe;


int FaceGenderClassifier::init(const string& dir, const int gpuid) {
  // Read net and parameters
  string mydeploy = dir + "/mydeploy.prototxt";
  caffe_test_net = new Net<float>(mydeploy, caffe::TEST);
  string mymodel = dir + "/mymodel.caffemodel";
  caffe_test_net->CopyTrainedLayersFrom(mymodel);

  // Set caffe mode
  if (gpuid != -1) {
    Caffe::SetDevice(gpuid);
    Caffe::set_mode(Caffe::GPU);
    LOG(INFO) << "Using GPU " << gpuid;
  } else {
    Caffe::set_mode(Caffe::CPU);
    LOG(INFO) << "Using CPU";
  }

  // Read mean
  mean[0] = 104; //blue mean
  mean[1] = 117; //green mean
  mean[2] = 123; //red mean

  batch_size = caffe_test_net->blob_by_name("data")->num(); //fixed to 1
  n_channel = caffe_test_net->blob_by_name("data")->channels(); //fixed to 3
  image_size_h = image_size_w = 256;
  image_crop_size_h = caffe_test_net->blob_by_name("data")->height();
  image_crop_size_w = caffe_test_net->blob_by_name("data")->width();
  if (image_size_h < image_crop_size_h || image_size_w < image_crop_size_w) {
    image_size_h = (int)round(image_crop_size_h * 1.1);
    image_size_w = (int)round(image_crop_size_w * 1.1);
  }
  height_offset = (image_size_h - image_crop_size_h) / 2;
  width_offset = (image_size_w - image_crop_size_w) / 2;
  
  LOG(INFO) << caffe_test_net->name();
  LOG(INFO) << caffe_test_net->blob_by_name("data")->num();
  LOG(INFO) << caffe_test_net->blob_by_name("data")->channels();
  LOG(INFO) << caffe_test_net->blob_by_name("data")->height();
  LOG(INFO) << caffe_test_net->blob_by_name("data")->width();
  
  return 0;
}


int FaceGenderClassifier::get_gender(const string& path) { 
  cv::Mat cv_img, cv_img_orig;
  cv_img_orig = cv::imread(path, CV_LOAD_IMAGE_COLOR);
  cv::resize(cv_img_orig, cv_img, cv::Size(image_size_w, image_size_h));

  vector<Blob<float>*> batch_frame_blobs;
  Blob<float>* frame_blob = new Blob<float>(batch_size, n_channel, image_crop_size_h, image_crop_size_w);
  float* frame_blob_data = frame_blob->mutable_cpu_data();
  for (int i_channel = 0; i_channel < n_channel; i_channel++) {
    for (int height = 0; height < image_crop_size_h; height++) {
      for (int width = 0; width < image_crop_size_w; width++) {
        int insertion_index = (i_channel * image_crop_size_h + height) * image_crop_size_w + width;
        frame_blob_data[insertion_index] = static_cast<float>(cv_img.at<cv::Vec3b>(height + height_offset, width + width_offset)[i_channel]) - mean[i_channel];
      }
    }
  }
  
  batch_frame_blobs.push_back(frame_blob);
  caffe_test_net->Forward(batch_frame_blobs);

  const shared_ptr<Blob<float> > prob_blob = caffe_test_net->blob_by_name("prob");
  const float* prob_blob_ptr = prob_blob->cpu_data();
  int face_gender = 0;
  if (prob_blob_ptr[0] < prob_blob_ptr[1]) {
    face_gender = 1;
  }
  
  delete frame_blob;
  return face_gender;
}


int FaceGenderClassifier::release() {
  delete caffe_test_net;
  return 0;
}


FaceGenderClassifier::FaceGenderClassifier() {
}
