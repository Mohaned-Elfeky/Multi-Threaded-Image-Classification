#ifndef IMAGEDISTANCE_H
#define IMAGEDISTANCE_H

#include <iostream>



using namespace std;

class ImageDistance{
    
private:

   

public:
    
    string label;  //the classification of the image
    double distance;
    
    ImageDistance(string lbl, double dist){
        
        label=lbl;
        distance=dist;
        
    }
    
    // overload the < operator inorder to facilitate sorting by distance
    bool operator< (const ImageDistance& other){
        return distance< other.distance;
    }
    
};


#endif