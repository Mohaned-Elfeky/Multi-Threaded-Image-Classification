#ifndef IMAGE_H
#define IMAGE_H

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Image{
    
private:

   

public:
    Mat img_data; //the opencv mat object
    String label;  //the classification of the image
    
    
    Image(Mat img, String lbl){
        img_data=img;
        label=lbl;
        
    }
    
    //overload the < operator inorder to facilitate sorting by distance
    // bool operator< (const Image& other){
    //     return distance< other.distance;
    // }
    
};


#endif