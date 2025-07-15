#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>

using namespace std;
using namespace cv;

//face detection function
void detectAndDraw(){
    
}


int main(){
    VideoCapture cap(0);
    CascadeClassifier facedetect;
    Mat frame;
    facedetect.load("/System/Volumes/Data/opt/homebrew/Cellar/opencv/4.11.0_1/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"); //add your 

    while (true){
        cap.read(frame);

        vector<Rect> faces;

        facedetect.detectMultiScale(frame, faces, 1.3, 5);

        for(int i=0; i<faces.size(); i++){
            rectangle(frame, faces[i].tl(), faces[i].br(), Scalar(50, 50, 255), 3);
        }

        imshow("Camera", frame);
        if(waitKey(30)==27){
            return 0;
        }
        waitKey(2);
    }
}
