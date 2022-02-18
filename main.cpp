#include <iostream>
#include <opencv2/opencv.hpp>
#include "Image.hpp"
#include "ImageDistance.hpp"
#include <chrono>



using namespace std::chrono ;
using namespace cv;
using namespace std;


Mat convert_to_grayscale(const Mat& rgb) { // &: reference. 

    // create a temp. image. 
    Mat gray_image(rgb.size().height, rgb.size().width, CV_8UC1, Scalar(0));

    // loop through all pixels in RGB image
    for (int row = 0; row < rgb.rows; ++row) {
        for (int col = 0; col < rgb.cols; ++col) {

            // extract a single RGB value
            Vec3b bgrpixel = rgb.at<Vec3b>(row, col); // {0,0}, {0,1}, .. {0,2}..
            
            //access each, blue, green, red and convert them to grayscale. 
            // (int)(0.114 * blue + 0.587 * green + 0.299 * red);
            int gray_value = (int)(0.114 * bgrpixel[0] + 0.587 * bgrpixel[1] + 0.299 * bgrpixel[2]);
            gray_image.at<uchar>(row, col) = gray_value;
        }
    }
    return gray_image;
}


void loadTrainingCatImagesSerially(vector<Image>* dataset){
    vector<cv::String> fn;
    glob("dataset\\training_set\\cats\\*.jpg", fn, false);


    size_t count = fn.size(); //number of png files in images folder
    for (size_t i=0; i<count; i++){
        Mat image = imread(fn[i]);
        image=convert_to_grayscale(image);
        dataset->emplace_back(image,"cat");
  }
}


void loadTrainingDogImagesSerially(vector<Image>* dataset){
    vector<cv::String> fn;
    glob("dataset\\training_set\\dogs\\*.jpg", fn, false);


    size_t count = fn.size(); //number of png files in images folder
    for (size_t i=0; i<count; i++){
        Mat image = imread(fn[i]);
        image=convert_to_grayscale(image);
        dataset->emplace_back(image,"dog");
  }
}




void loadDataSerially(vector<Image>* dataset){
  loadTrainingDogImagesSerially(dataset);
  loadTrainingCatImagesSerially(dataset);
}


void loadDataParallaly(vector<Image>* dataset){
  
  
}


//calculates the distances to one image
vector<ImageDistance> calculateDistances_serially(vector<Image>* dataset,Mat& test_image){
  vector<ImageDistance> distances;
  int total_test_pixels=test_image.rows * test_image.cols;  //total number of elements in the test image
  for(int i=0; i<dataset->size(); i++){
    
    Mat resized_image;
    
    resize( (*dataset)[i].img_data , resized_image, Size(test_image.size().width, test_image.size().height), INTER_LINEAR);  //resize the dataset to the test image dimensions
    
    
    
    double distance=0;
    
    for(int j=0; j<total_test_pixels ; j++){   
      
      distance+= pow(resized_image.data[j]-test_image.data[j],2);  //calculate the distance between two pixels
       
    }
    
    distance=sqrt(distance);
    distances.emplace_back((*dataset)[i].label,distance);   //the total distance between the 2 images
    
        
  }
  
  return distances;
}

void getClassification(int k, vector<ImageDistance>& distances){
  double cat_count=0;
  double dog_count=0;
  
  for(int i=0; i<k ; i++){
    string current_type=distances[i].label;
    
    if(current_type=="cat") cat_count++;
    else dog_count++; 
    
  }
  
  
  if(cat_count>dog_count) cout<<"Classification is : Cat, Confidence= "<<double((cat_count/k))*100<<endl;
  else   cout<<"Classification is : Dog, Confidence= "<<double((dog_count/k))*100<<endl;
   
}


// void classifyFolderSerially(vector<Image>* dataset, string file_path){
  
//   loadDataSerially(dataset);
  
//   vector<cv::String> fn;
//   glob(file_path+"\\*.jpg", fn, false); //get all file paths in the folder
  
//   size_t count = fn.size(); //number of jpg files in images folder
//     for (size_t i=0; i<count; i++){
//         Mat test_image = imread(fn[i]);
        
//         calculateDistances(dataset,test_image);  //calculate the distances to the test image
//         std::sort((*dataset).begin(), (*dataset).end());  //sort the vector by distances
//         getClassification(12,dataset);   //get the classification according to the given K value
         
//   }
  
  
// }

void classifyImageSerially(vector<Image>* dataset, string file_path, int k){
        
        auto start = high_resolution_clock::now();
        
        loadDataSerially(dataset);
        
        Mat test_image = imread(file_path);
        test_image=convert_to_grayscale(test_image);
        
        
        
        vector<ImageDistance> distances = calculateDistances_serially(dataset,test_image);
        std::sort(distances.begin(), distances.end());
        getClassification(k,distances);
        
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<seconds>(stop - start);
        cout << "time of serial implementation for one image: "<<duration.count() << endl;
  
}



int main(){
  
//allocate the vector and the image objects on the heap
  vector<Image>* dataset = new vector<Image>;
  
  
  classifyImageSerially(dataset,"dataset\\test_set\\dogs\\dog.4321.jpg",9);
   
  
  
  //clear vector by swapping it by an empty one 
  vector<Image>().swap( *dataset );
  

    return 0;
    
}

//build command : g++ --std c++17 -g main.cpp -o output.exe -I C:\OpenCV-MinGW-Build-OpenCV-4.5.2-x64\include -L C:\OpenCV-MinGW-Build-OpenCV-4.5.2-x64\x64\mingw\bin -llibopencv_calib3d452 -llibopencv_core452 -llibopencv_dnn452 -llibopencv_features2d452 -llibopencv_flann452 -llibopencv_highgui452 -llibopencv_imgcodecs452 -llibopencv_imgproc452 -llibopencv_ml452 -llibopencv_objdetect452 -llibopencv_photo452 -llibopencv_stitching452 -llibopencv_video452 -llibopencv_videoio452