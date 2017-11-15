#ifndef FACE_GENDER_CLASSIFIER_HPP_
#define FACE_GENDER_CLASSIFIER_HPP_

#include "caffe/caffe.hpp"

#include <string>
using namespace std;
using namespace caffe;

class FaceGenderClassifier {
  public:
    explicit FaceGenderClassifier();
    virtual ~FaceGenderClassifier(){}
    
//    static FaceGenderClassifier* _pInstance;

    /**
     * initialize FaceGenderClassifier
     * dir, the directory of caffemodel and prototxt
     * gpuid, -1 for cpu mode, other for gpu mode and the number is gpuid
     */
    int init(const string& dir,const int gpuid = -1);

    /**
     * aquire gender of face image
     * @return gender (0 for female, 1 for male)
     * path, path of face image
     */
    int get_gender(const string& path);

    int release();

  protected:
    Net<float>* caffe_test_net;
    float mean[3];
    int batch_size, n_channel;
    int image_size_h, image_size_w;
    int image_crop_size_h, image_crop_size_w;
    int height_offset, width_offset;
};

//FaceGenderClassifier* FaceGenderClassifier::_pInstance = NULL;
//
//extern "C" {
//  extern FaceGenderClassifier* create() {
//    if (NULL == FaceGenderClassifier::_pInstance) {
//      FaceGenderClassifier::_pInstance = new FaceGenderClassifier();
//    }
//    return FaceGenderClassifier::_pInstance;
//  }
//  extern void destroy(FaceGenderClassifier* p) {
//    if (p != NULL) {
//      p->release();
//      delete p;
//      p = NULL;
//    }
//  }
//}
//
//typedef FaceGenderClassifier* create_t();
//typedef void destroy_t(FaceGenderClassifier* p);

#endif
