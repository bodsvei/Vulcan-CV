#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
    VideoCapture cap(0);
    while (true){
        Mat frame;
        cap.read(frame);
        imshow("Camera", frame);
        if(waitKey(30)==27){
            return 0;
        }
    }
}
