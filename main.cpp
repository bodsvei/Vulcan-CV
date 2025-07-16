#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

// Defining classifier paths as constants
const std::string FACE_CASCADE_PATH = "/System/Volumes/Data/opt/homebrew/Cellar/opencv/4.11.0_1/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
const std::string EYE_CASCADE_PATH = "/System/Volumes/Data/opt/homebrew/Cellar/opencv/4.11.0_1/share/opencv4/haarcascades/haarcascade_eye.xml";


int main() {
    // 1. --- INITIALIZATION ---
    // Open the default camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    // Load pre-trained classifiers once, outside the loop
    cv::CascadeClassifier face_detect;
    if (!face_detect.load(FACE_CASCADE_PATH)) {
        std::cerr << "Error: Could not load face cascade from " << FACE_CASCADE_PATH << std::endl;
        return -1;
    }

    cv::CascadeClassifier eye_detect;
    if (!eye_detect.load(EYE_CASCADE_PATH)) {
        std::cerr << "Error: Could not load eye cascade from " << EYE_CASCADE_PATH << std::endl;
        return -1;
    }

    cv::Mat frame;

    // 2. --- MAIN PROCESSING LOOP ---
    while (true) {
        // Capture a new frame
        if (!cap.read(frame)) {
            std::cerr << "Error: Could not read frame from camera." << std::endl;
            break;
        }

        // --- Pre-processing for Detection ---
        // Convert to grayscale and equalize for better contrast, which helps the detector
        cv::Mat gray_frame;
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray_frame, gray_frame);

        // --- Face Detection ---
        std::vector<cv::Rect> faces;
        face_detect.detectMultiScale(gray_frame, faces, 1.1, 5, 0, cv::Size(30, 30));

        // Find the index of the closest face
        int largest_face_index = -1;
        double max_area = 0.0;
        for (size_t i = 0; i < faces.size(); ++i) {
            double area = faces[i].area();
            if (area > max_area) {
                max_area = area;
                largest_face_index = static_cast<int>(i);
            }
        }

        // --- Drawing and Eye Detection within each Face ---
        for (size_t i = 0; i < faces.size(); ++i) {
            // Determine color for the face rectangle
            cv::Scalar face_rect_color = (static_cast<int>(i) == largest_face_index) ? cv::Scalar(50, 250, 50) : cv::Scalar(50, 50, 250);
            
            // Draw rectangle around the face on the original color frame
            cv::rectangle(frame, faces[i], face_rect_color, 3);

            // --- Eye Detection within the Face ROI ---
            // Create a Region of Interest (ROI) for the detected face
            cv::Mat faceROI = gray_frame(faces[i]);
            std::vector<cv::Rect> eyes;

            // Detect eyes within the face ROI, not the whole image
            eye_detect.detectMultiScale(faceROI, eyes, 1.1, 5, 0, cv::Size(20, 20));

            for (size_t j = 0; j < eyes.size(); ++j) {
                // Get the eye rectangle's coordinates relative to the face ROI
                cv::Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
                int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
                
                cv::circle(frame, eye_center, radius, cv::Scalar(255, 0, 0), 2); // Blue circle
            }
        }
        
        // Display the number of detected faces on the frame
        cv::putText(frame, "Faces: " + std::to_string(faces.size()), cv::Point(10, 40), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 255, 255), 1);

        // 3. --- DISPLAY AND EXIT ---
        // Display the resulting frame
        cv::imshow("Camera", frame);

        // Exit loop if ESC key
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // 4. --- CLEANUP ---
    // Release the camera and destroy all windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}