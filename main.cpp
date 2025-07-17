#include <iostream>
#include <vector>
#include <string>
#include <iomanip> // For std::setprecision
#include <sstream> // For std::stringstream

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

// --- Best Practice: Define constants for easy modification ---
const std::string FACE_CASCADE_PATH = "/System/Volumes/Data/opt/homebrew/Cellar/opencv/4.11.0_1/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
const cv::Size SIDEBAR_FACE_SIZE(250, 250); // The size for the isolated face in the sidebar


int main() {
    // 1. --- INITIALIZATION ---
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    cv::CascadeClassifier face_detect;
    if (!face_detect.load(FACE_CASCADE_PATH)) {
        std::cerr << "Error: Could not load face cascade from " << FACE_CASCADE_PATH << std::endl;
        return -1;
    }

    cv::Mat frame;

    // 2. --- MAIN PROCESSING LOOP ---
    while (true) {
        int64 start_time = cv::getTickCount();

        if (!cap.read(frame)) {
            std::cerr << "Error: Could not read frame from camera." << std::endl;
            break;
        }

        // --- Pre-processing for Detection ---
        cv::Mat gray_frame;
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray_frame, gray_frame);

        // --- Face Detection ---
        std::vector<cv::Rect> faces;
        face_detect.detectMultiScale(gray_frame, faces, 1.1, 5, 0, cv::Size(30, 30));

        // --- Analysis: Find the largest face ---
        int largest_face_index = -1;
        double max_area = 0.0;
        for (size_t i = 0; i < faces.size(); ++i) {
            double area = faces[i].area();
            if (area > max_area) {
                max_area = area;
                largest_face_index = static_cast<int>(i);
            }
        }

        // --- Drawing Logic for all detected faces on the main frame ---
        for (size_t i = 0; i < faces.size(); ++i) {
            cv::Scalar face_rect_color = (static_cast<int>(i) == largest_face_index) ? cv::Scalar(50, 250, 50) : cv::Scalar(50, 50, 250);
            cv::rectangle(frame, faces[i], face_rect_color, 3);

            // --- Crosshair ---
            cv::Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
            int crosshair_size = faces[i].width / 5;
            cv::line(frame, cv::Point(center.x - crosshair_size, center.y), cv::Point(center.x + crosshair_size, center.y), cv::Scalar(0, 255, 255), 2);
            cv::line(frame, cv::Point(center.x, center.y - crosshair_size), cv::Point(center.x, center.y + crosshair_size), cv::Scalar(0, 255, 255), 2);
        }

        // --- Performance and Info Display on the main frame ---
        double ms = (static_cast<double>(cv::getTickCount()) - start_time) / cv::getTickFrequency() * 1000.0;
        std::stringstream ss;
        ss << "Faces: " << faces.size() << " | Time: " << std::fixed << std::setprecision(2) << ms << " ms";
        cv::putText(frame, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(255, 255, 255), 1);

        // --- Final View Composition ---
        // This will be the final image we show. Initialize it with the main frame.
        cv::Mat final_view = frame;

        if (largest_face_index != -1) {
            // Create a black canvas with extra width for the sidebar
            cv::Mat canvas(frame.rows, frame.cols + SIDEBAR_FACE_SIZE.width, frame.type(), cv::Scalar(0, 0, 0));

            // Copy the main frame to the left side of the canvas
            frame.copyTo(canvas(cv::Rect(0, 0, frame.cols, frame.rows)));

            // Get the ROI for the largest face and create a display copy
            cv::Rect largest_face_rect = faces[largest_face_index];
            cv::Mat closest_face_display = frame(largest_face_rect).clone();
            cv::resize(closest_face_display, closest_face_display, SIDEBAR_FACE_SIZE, 0, 0, cv::INTER_AREA);

            // Add a title to the sidebar face display
            cv::putText(closest_face_display, "Closest Face", cv::Point(10, 25), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(255, 255, 255), 1);

            // Copy the resized face to the top-right side of the canvas (the sidebar)
            closest_face_display.copyTo(canvas(cv::Rect(frame.cols, 0, SIDEBAR_FACE_SIZE.width, SIDEBAR_FACE_SIZE.height)));
            
            // The canvas is now our final view
            final_view = canvas;
        }

        // 3. --- DISPLAY AND EXIT ---
        cv::imshow("Face Detection", final_view);

        if (cv::waitKey(1) == 27) { // Exit on ESC key
            break;
        }
    }

    // 4. --- CLEANUP ---
    cap.release();
    cv::destroyAllWindows();

    return 0;
}