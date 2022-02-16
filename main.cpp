#include <iostream>
#include <opencv2/opencv.hpp>
#include "Image.hpp"
#include <chrono>



using namespace std::chrono;
using namespace cv;
using namespace std;

void loadTrainingCatImages(vector<Image>* dataset){
    vector<cv::String> fn;
    glob("dataset\\training_set\\cats\\*.jpg", fn, false);


    size_t count = fn.size(); //number of png files in images folder
    for (size_t i=0; i<count; i++){
        Mat image = imread(fn[i]);
        dataset->emplace_back(image,"cat");
  }
}


void loadTrainingDogImages(vector<Image>* dataset){
    vector<cv::String> fn;
    glob("dataset\\training_set\\dogs\\*.jpg", fn, false);


    size_t count = fn.size(); //number of png files in images folder
    for (size_t i=0; i<count; i++){
        Mat image = imread(fn[i]);
        dataset->emplace_back(image,"dog");
  }
}


void calculateDistances(vector<Image>* dataset,Mat& test_image){
  
  int total_test_pixels=test_image.rows * test_image.cols * test_image.channels();  //total number of elements in the test image
  for(int i=0; i<dataset->size(); i++){
    
    Mat resized_image;
    
    resize( (*dataset)[i].img_data , resized_image, Size(test_image.size().width, test_image.size().height), INTER_LINEAR);  //resize the dataset to the test image dimensions
    
    int distance=0;
    
    for(int j=0; j<total_test_pixels ; j+=3){   
      
      //test image rgb values
      int test_red=test_image.data[j];
      int test_green=test_image.data[j+1];
      int test_blue=test_image.data[j+2];
      
      //dataset image[i] rgb values
      int current_red=resized_image.data[j];
      int current_green=resized_image.data[j+1];
      int current_blue=resized_image.data[j+2];
      
      distance+=sqrt( pow((current_red-test_red),2) + pow((current_green-test_green),2) + pow((current_blue-test_blue),2) );  //calculate the distance between two pixels
       
    }
    
    (*dataset)[i].distance=sqrt(distance);   //the total distance between the 2 images
    
        
  }
}

void getClassification(int k, vector<Image>* dataset){
  double cat_count=0;
  double dog_count=0;
  
  for(int i=0; i<k ; i++){
    string current_type=(*dataset)[i].label;
    
    if(current_type=="cat") cat_count++;
    else dog_count++; 
    
  }
  
  
  if(cat_count>dog_count) cout<<"Classification is : Cat, Confidence= "<<double((cat_count/k))*100<<endl;
  else   cout<<"Classification is : Dog, Confidence= "<<double((dog_count/k))*100<<endl;
   
}


void classifyFolder(vector<Image>* dataset, string file_path){
  vector<cv::String> fn;
  glob(file_path+"\\*.jpg", fn, false); //get all file paths in the folder
  
  size_t count = fn.size(); //number of jpg files in images folder
    for (size_t i=0; i<count; i++){
        Mat test_image = imread(fn[i]);
        
        calculateDistances(dataset,test_image);  //calculate the distances to the test image
        std::sort((*dataset).begin(), (*dataset).end());  //sort the vector by distances
        getClassification(12,dataset);   //get the classification according to the given K value
         
  }
  
  
}

void classifyImage(vector<Image>* dataset, string file_path){
  
        Mat test_image = imread(file_path);
        calculateDistances(dataset,test_image);
        std::sort((*dataset).begin(), (*dataset).end());
        getClassification(12,dataset);
  
}

void printDataset(vector<Image>* dataset){
  for(int i=0; i<100 ; i++){
    cout<<i<<"-"<<(*dataset)[i].distance<<"-"<<(*dataset)[i].label<<endl;
  }
  
}

//build command : g++ --std c++17 -g main.cpp -o output.exe -I C:\OpenCV-MinGW-Build-OpenCV-4.5.2-x64\include -L C:\OpenCV-MinGW-Build-OpenCV-4.5.2-x64\x64\mingw\bin -llibopencv_calib3d452 -llibopencv_core452 -llibopencv_dnn452 -llibopencv_features2d452 -llibopencv_flann452 -llibopencv_highgui452 -llibopencv_imgcodecs452 -llibopencv_imgproc452 -llibopencv_ml452 -llibopencv_objdetect452 -llibopencv_photo452 -llibopencv_stitching452 -llibopencv_video452 -llibopencv_videoio452
int main(){
  
//allocate the vector and the image objects on the heap
  vector<Image>* dataset = new vector<Image>;
  
  auto start = high_resolution_clock::now();
  
  //load the training cat images to the vector
  loadTrainingCatImages(dataset);
  
  //load the training dog images to the vector
  loadTrainingDogImages(dataset);
  

  
  
  classifyImage(dataset,"dataset\\test_set\\cats\\cat.4301.jpg"); // one image classification function
  
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<seconds>(stop - start);
  cout << "time taken to classify one image serially is: "<<duration.count() << endl;
  
  // classifyFolder(dataset,"dataset\\test_set\\dogs");
  
  
  //clear vector by swapping it by an empty one 
  vector<Image>().swap( *dataset );
  

    return 0;
    
}