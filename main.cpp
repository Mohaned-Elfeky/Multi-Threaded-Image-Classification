#include <iostream>
#include <opencv2/opencv.hpp>
#include "Image.hpp"
#include "ImageDistance.hpp"
#include <chrono>
#include <thread>
#include <future>



using namespace std::chrono ;
using namespace cv;
using namespace std;

Mutex dataloading_mutex; //mutex used by the threads that will load the data
Mutex distance_mutex;  //mutex used by the threads that will calculate the distances

//convert an rgb image to a single channel greyscale image
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

//load the training images for cats into the dataset vector
void loadTrainingCatImagesSerially(vector<Image>* dataset){ //pointer to the dataset vector
    vector<cv::String> fn;  //vector to hold file names of all the images in a given folder
    
    glob("dataset\\training_set\\cats\\*.jpg", fn, false);  //load all the image paths in the given path to the fn vector 


    size_t count = fn.size(); //number of jpg files in images folder
    for (size_t i=0; i<count; i++){
        Mat image = imread(fn[i]);
        image=convert_to_grayscale(image);
        dataset->emplace_back(image,"cat"); //add the greyscale image to the dataset vector
  }
}

//same as load cats but for the dogs folder instead
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



//function to be called by the threads that will load the images into the dataset parallelly by reading the images in thr fn vector 
//from the start to the end value provided
void loadImages(vector<Image>* dataset,string label, vector<cv::String>& fn ,int start, int end){
        for(int i=start; i<end; i++){
          
            Mat image = imread(fn[i]);   //read the image of each path in the folder
            image=convert_to_grayscale(image);  //convert the image to grey scale
            unique_lock<Mutex> lck(dataloading_mutex);   //lock the dataset vector to prevent race condition between the multiple threads trying to inset into it
            dataset->emplace_back(image,label); //insert the image to the dataset vector
            
        }
       
}


//loads the training cat images by dividing the folder among 6 threads 
//this is the function that achieves parallellism for loading the data
void loadTrainingCatImagesParallelly(vector<Image>* dataset){
    vector<cv::String> fn;
    glob("dataset\\training_set\\cats\\*.jpg", fn, false); //training images for cats folder
    size_t count = fn.size(); //number of jpg files in images folder
    
    //create 6 threads and divide the folder among them to be loaded parallelly into the dataset vector
    thread t1(loadImages,dataset,"cat",std::ref(fn),0,count/6);
    thread t2(loadImages,dataset,"cat",std::ref(fn),count/6,count*2/6);
    thread t3(loadImages,dataset,"cat",std::ref(fn),count*2/6,count*3/6);
    thread t4(loadImages,dataset,"cat",std::ref(fn),count*3/6,count*4/6);
    thread t5(loadImages,dataset,"cat",std::ref(fn),count*4/6,count*5/6);
    thread t6(loadImages,dataset,"cat",std::ref(fn),count*5/6,count);
    
    
    //wait until all the 6 threads have done executing
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
    t6.join();
}

//same as previous function but for dogs training fodler
void loadTrainingDogImagesparallelly(vector<Image>* dataset){
     
    vector<cv::String> fn;
    glob("dataset\\training_set\\dogs\\*.jpg", fn, false);
  

    size_t count = fn.size(); //number of png files in images folder
    thread t1(loadImages,dataset,"dog",std::ref(fn),0,count/6);
    thread t2(loadImages,dataset,"dog",std::ref(fn),count/6,count*2/6);
    thread t3(loadImages,dataset,"dog",std::ref(fn),count*2/6,count*3/6);
    thread t4(loadImages,dataset,"dog",std::ref(fn),count*3/6,count*4/6);
    thread t5(loadImages,dataset,"dog",std::ref(fn),count*4/6,count*5/6);
    thread t6(loadImages,dataset,"dog",std::ref(fn),count*5/6,count);
    
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
    t6.join();
   
    
}



void loadTrainingDataSerially(vector<Image>* dataset){
  loadTrainingDogImagesSerially(dataset);
  loadTrainingCatImagesSerially(dataset);
}

//calls the parallell loading functions for both the cats and dogs folders 
void loadTrainingDataparallelly(vector<Image>* dataset){
  //create a seperate thread for each folder (one for dogs and one for cats folder)
  thread t1(loadTrainingCatImagesParallelly,dataset); 
  thread t2(loadTrainingDogImagesparallelly,dataset);
  
  t1.join();
  t2.join();
}



//calculates the distances to the test image serially
vector<ImageDistance> calculateDistances_serially(vector<Image>* dataset,Mat& test_image){
  vector<ImageDistance> distances; //a vector to hold the distances and the type of that distance(cat or dog)
  int total_test_pixels=test_image.rows * test_image.cols;  //total number of elements in the test image
  
  //iterate every image in the dataset vector and calculate its distance from the test image
  for(int i=0; i<dataset->size(); i++){
    
    Mat resized_image;  
    //resize each image in the dataset to match the test image dimensions
    resize( (*dataset)[i].img_data , resized_image, Size(test_image.size().width, test_image.size().height), INTER_LINEAR);  //resize the dataset to the test image dimensions
    
    
    
    double distance=0;
    //calculate the distance by summing the difference of powers
    for(int j=0; j<total_test_pixels ; j++){   
      
      distance+= pow(resized_image.data[j]-test_image.data[j],2);  //calculate the distance between two pixels
       
    }
    
    distance=sqrt(distance);
    
    //adding the distance value and the label of the image to the distances vector
    distances.emplace_back((*dataset)[i].label,distance);   
    
        
  }
  
  return distances; //return the distances vector
}



//function to be called by the threads that will calculate the distance between an image in the dataset and the test image
//this achieves parallelism pixel level wise
double getPixlesDistance(unsigned char* img,unsigned char* test_img,int start,int end,int depth){
  
  //stop at depth 2 with total of 2^2=4 threads running
  if(depth>=2){
      double distance=0;
      for(int i=start; i<end ; i++){
              distance+=pow((img[i]-test_img[i]),2); //sum of differences between the pixels between tha start and the end. 
        }
        
    return distance;
  }
  else{
    depth ++;
    
    //divide the number of pixels to to halves and each half is operated on by a seperate thread
    future<double>  left_distance =std::async(std::launch::async,getPixlesDistance,img,test_img,start,start+(end-start)/2,depth);
    double right_distance =getPixlesDistance(img,test_img,start+(end-start)/2,end,depth);
  
    return left_distance.get()+right_distance; //accumalate the sum of the left and right side of the pixels 
  }
  
    
}


 //function to be called by the threads that will calculate the distances between the training images and the test image
void getImagesDistance(vector<Image>* dataset,Mat& test_image,vector<ImageDistance>& distances,int total_test_pixels ,int start, int end ){
  
  //iterate the dataset vector from the start given to the end given
  for(int i=start; i<end; i++){
    
    Mat resized_image;
    //resize each image in the dataset to match the test image dimensions
    resize( (*dataset)[i].img_data , resized_image, Size(test_image.size().width, test_image.size().height), INTER_LINEAR);  //resize the dataset to the test image dimensions
    
    //get the distance between dataset[i] image and the test image parallelly
    double distance=sqrt(getPixlesDistance(resized_image.data,test_image.data,0,total_test_pixels,0));
    
    unique_lock<Mutex> lck(dataloading_mutex);  //lock the distances vector to prevent race condition between the multiple threads trying to inset into it
    distances.emplace_back((*dataset)[i].label,distance);   //add the distance and the label
    
        
  }
}

//calculates the distances of the dataset to the test image parallelly
vector<ImageDistance> calculateDistances_parallelly(vector<Image>* dataset,Mat& test_image){
  vector<ImageDistance> distances;
  int total_test_pixels=test_image.rows * test_image.cols;  //total number of elements in the test image
  int n=(*dataset).size();
  
  
    //divide the images in the dataset into 4 threads inorder to parallelize the process of getting the distances
    thread t1(getImagesDistance,dataset,std::ref(test_image),std::ref(distances),total_test_pixels,0,n/4);
    thread t2(getImagesDistance,dataset,std::ref(test_image),std::ref(distances),total_test_pixels,n/4,n/2);
    thread t3(getImagesDistance,dataset,std::ref(test_image),std::ref(distances),total_test_pixels,n/2,n*(3/4));
    thread t4(getImagesDistance,dataset,std::ref(test_image),std::ref(distances),total_test_pixels,n*(3/4),n);
   
    
   //wait until all the threads have finished and all the distances have been calculated 
    t1.join();
    t2.join();
    t3.join();
    t4.join();
  


  return distances; //return the distances vector
}


//get the classification depending on the K value 
void getClassification(int k, vector<ImageDistance>& distances,string img_name){
  double cat_count=0;
  double dog_count=0;
  
  //get the first k elements in the sorted distances vector
  for(int i=0; i<k ; i++){
    string current_type=distances[i].label;
    
    if(current_type.compare("cat") == 0) cat_count++; 
    else dog_count++; 
    
  }
  
  cout<<img_name;
  if(cat_count>dog_count) cout<<" classification is : Cat, Confidence= "<<double((cat_count/k))*100<<endl;
  else   cout<<" classification is : Dog, Confidence= "<<double((dog_count/k))*100<<endl;
   
}







//classify a single test image serially
void classifyImageSerially(vector<Image>* dataset, string file_path, int k){
        
      
        Mat test_image = imread(file_path);
        test_image=convert_to_grayscale(test_image);
        
        
        
        vector<ImageDistance> distances = calculateDistances_serially(dataset,test_image);
        //sort the distances vector
        std::sort(distances.begin(), distances.end());
        
        //get the classification  by passing the sorted distances vector and the desired K value
        getClassification(k,distances,file_path);
        
      
  
}

//classify a single image parallelly
void classifyImageParallelly(vector<Image>* dataset, string file_path, int k){
        
        
        Mat test_image = imread(file_path);
  
        
        test_image=convert_to_grayscale(test_image);
        
        
        
        vector<ImageDistance> distances = calculateDistances_parallelly(dataset,test_image);
        //sort the distances vector
        std::sort(distances.begin(), distances.end());
        
        //get the classification  by passing the sorted distances vector and the desired K value
        getClassification(k,distances,file_path);
        
      
  
}
//classify a test folder
void classifyFolderSerially(vector<Image>* dataset, string file_path,int k){
  
  
  
  vector<cv::String> fn;
  glob(file_path+"\\*.jpg", fn, false); //get all file paths in the folder
  
  size_t count = fn.size(); //number of jpg files in images folder
    for (size_t i=0; i<count; i++){
      classifyImageSerially(dataset,fn[i],k); //classify each image in the test folder
         
  }
  
  
}

void classifyFolderParallelly(vector<Image>* dataset, string file_path,int k){
  
  
  
  vector<cv::String> fn;
  glob(file_path+"\\*.jpg", fn, false); //get all file paths in the folder
  
  size_t count = fn.size(); //number of jpg files in images folder
    for (size_t i=0; i<count; i++){
        
        std::async(std::launch::async,
            [&](){
        classifyImageParallelly(dataset,fn[i],7);   //classify each image in the test folder  
                                                    //async was used to try to classify each image on a seperate thread but due to large number of images we use async as it automatically alocates threads if available
            });
         
  }
  
  
}


int main(int argc, char* argv[]){
  
  //allocate a vector in the heap for the dataset images to be stored in.
  vector<Image>* dataset = new vector<Image>;
   
   string test_type(argv[1]); //the type of test images provided:'image' for a single image or 'folder' for a folder of images
   string path(argv[2]); //the path for the image or the folder
   string k_s(argv[3]); //the k value
   int k=stoi( k_s );
   
   
   loadTrainingDataparallelly(dataset);//load the training images into the dataset vector parallelly
   
   //checks the test_type argument whether it is a single image or folder
   if(test_type.compare("image") == 0){
      
    classifyImageParallelly(dataset,path,k);
   }
   else if(test_type.compare("folder") == 0){
     
    classifyFolderParallelly(dataset,path,k);
    
   }
   
   
    //clear vector by swapping it by an empty one inorder to prevent memory leaks
    vector<Image>().swap( *dataset );
  

    return 0;
    
}

//build command : g++ --std c++17 -g main.cpp -o output.exe -I C:\OpenCV-MinGW-Build-OpenCV-4.5.2-x64\include -L C:\OpenCV-MinGW-Build-OpenCV-4.5.2-x64\x64\mingw\bin -llibopencv_calib3d452 -llibopencv_core452 -llibopencv_dnn452 -llibopencv_features2d452 -llibopencv_flann452 -llibopencv_highgui452 -llibopencv_imgcodecs452 -llibopencv_imgproc452 -llibopencv_ml452 -llibopencv_objdetect452 -llibopencv_photo452 -llibopencv_stitching452 -llibopencv_video452 -llibopencv_videoio452
//run command for testing an image: .\output.exe image image_path  k
//run command for testing a folder: .\output.exe folder folder_path  k