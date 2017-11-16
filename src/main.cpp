#include <iostream>
#include <string>
#include <cstdlib>

#include "face_gender_classifier.hpp"

using namespace std;

int main(int argc, char** argv) {
  FaceGenderClassifier fg_classifier;
  fg_classifier.init(argv[1], atoi(argv[2]));
  string image_path;
  while(cin >> image_path) {
    cout << fg_classifier.get_gender(image_path) << endl;
  }
  fg_classifier.release();
  return 0;  
}
