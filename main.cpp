#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <filesystem> // C++17 for directory iteration

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp> // Required for FaceRecognizer

namespace fs = std::filesystem; // For easier use of filesystem

// --- Constants ---
const std::string FACE_CASCADE_PATH = "/System/Volumes/Data/opt/homebrew/Cellar/opencv/4.11.0_1/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
const cv::Size SIDEBAR_FACE_SIZE(300, 300);
const std::string TRAINING_DATA_PATH = "training_faces"; // Directory to store training images
const double RECOGNITION_THRESHOLD = 80.0; // Adjust this threshold for better/worse matching

// Function to get training images and labels
// Assumes directory structure: TRAINING_DATA_PATH/label_name/image.jpg
// E.g., training_faces/Alice/1.jpg, training_faces/Alice/2.jpg, training_faces/Bob/1.jpg
void read_training_data(const std::string& path, std::vector<cv::Mat>& images, std::vector<int>& labels, std::vector<std::string>& label_names) {
    int current_label_id = 0;
    std::map<std::string, int> name_to_label_id; // Map names to integer labels

    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.is_directory()) {
            std::string person_name = entry.path().filename().string();
            if (name_to_label_id.find(person_name) == name_to_label_id.end()) {
                name_to_label_id[person_name] = current_label_id++;
                label_names.push_back(person_name); // Store name corresponding to label_id
            }
            int label = name_to_label_id[person_name];

            for (const auto& img_entry : fs::directory_iterator(entry.path())) {
                if (img_entry.is_regular_file()) {
                    cv::Mat img = cv::imread(img_entry.path().string(), cv::IMREAD_GRAYSCALE);
                    if (!img.empty()) {
                        images.push_back(img);
                        labels.push_back(label);
                    } else {
                        std::cerr << "Warning: Could not read image " << img_entry.path().string() << std::endl;
                    }
                }
            }
        }
    }
    std::cout << "Loaded " << images.size() << " training images from " << label_names.size() << " individuals." << std::endl;
}


int main() {
    // 1. --- INITIALIZATION ---
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }
    cv::CascadeClassifier face_detect;
    if (!face_detect.load(FACE_CASCADE_PATH)){
        std::cerr << "Error: Could not load face cascade from " << FACE_CASCADE_PATH << std::endl;
        return -1;
    }

    // --- Face Recognition Setup ---
    cv::Ptr<cv::face::FaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
    std::vector<cv::Mat> training_images;
    std::vector<int> training_labels;
    std::vector<std::string> label_names; // Maps integer label to actual name

    // Load training data and train the model
    try {
        read_training_data(TRAINING_DATA_PATH, training_images, training_labels, label_names);
        if (training_images.empty() || training_labels.empty()) {
            std::cerr << "Error: No training data found. Please create '" << TRAINING_DATA_PATH << "' with subdirectories for each person (e.g., training_faces/Alice/1.jpg)." << std::endl;
            // Optionally, allow running in detection-only mode if no training data
            // Or exit if recognition is mandatory. For now, we'll exit.
            return -1;
        }
        model->train(training_images, training_labels);
        std::cout << "FaceRecognizer model trained successfully." << std::endl;
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error accessing training data: " << e.what() << std::endl;
        std::cerr << "Please ensure '" << TRAINING_DATA_PATH << "' exists and contains training data." << std::endl;
        return -1;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error during training: " << e.what() << std::endl;
        return -1;
    }


    cv::Mat frame;
    cv::Size recognition_face_size(100, 100); // Standardize face size for recognition input

    // 2. --- MAIN PROCESSING LOOP ---
    while (true) {
        int64 start_time = cv::getTickCount();

        if (!cap.read(frame)){
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
            int crosshair_size = faces[i].width / 20;
            cv::line(frame, cv::Point(center.x - crosshair_size, center.y), cv::Point(center.x + crosshair_size, center.y), cv::Scalar(0, 255, 255), 2);
            cv::line(frame, cv::Point(center.x, center.y - crosshair_size), cv::Point(center.x, center.y + crosshair_size), cv::Scalar(0, 255, 255), 2);

            // --- Facial Recognition for each detected face ---
            cv::Mat face_roi = gray_frame(faces[i]);
            cv::Mat resized_face;
            cv::resize(face_roi, resized_face, recognition_face_size, 0, 0, cv::INTER_AREA);

            int predicted_label = -1;
            double confidence = 0.0;
            model->predict(resized_face, predicted_label, confidence);

            std::string recognized_name = "Unknown";
            cv::Scalar text_color = cv::Scalar(0, 0, 255); // Red for unknown

            if (predicted_label != -1 && confidence < RECOGNITION_THRESHOLD) {
                 if (static_cast<size_t>(predicted_label) < label_names.size()) {
                    recognized_name = label_names[predicted_label];
                    text_color = cv::Scalar(0, 255, 0); // Green for known
                 }
            }

            // Display name and confidence
            std::stringstream reco_ss;
            reco_ss << recognized_name;
            if (recognized_name != "Unknown") {
                 reco_ss << " (" << std::fixed << std::setprecision(0) << confidence << ")";
            }

            cv::putText(frame, reco_ss.str(), cv::Point(faces[i].x, faces[i].y - 10),
                        cv::FONT_HERSHEY_DUPLEX, 0.7, text_color, 2);
        }

        // --- Performance and Info Display on the main frame ---
        double ms = (static_cast<double>(cv::getTickCount()) - start_time) / cv::getTickFrequency() * 1000.0;
        std::stringstream ss;
        ss << "Faces: " << faces.size() << " | Time: " << std::fixed << std::setprecision(2) << ms << " ms";
        cv::putText(frame, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(255, 255, 255), 1);

        // --- Final View Composition (sidebar for closest face) ---
        cv::Mat final_view = frame;

        if (largest_face_index != -1) {
            cv::Mat canvas(frame.rows, frame.cols + SIDEBAR_FACE_SIZE.width, frame.type(), cv::Scalar(0, 0, 0));
            frame.copyTo(canvas(cv::Rect(0, 0, frame.cols, frame.rows)));

            cv::Rect largest_face_rect = faces[largest_face_index];
            cv::Mat closest_face_display = frame(largest_face_rect).clone();
            cv::resize(closest_face_display, closest_face_display, SIDEBAR_FACE_SIZE, 0, 0, cv::INTER_AREA);

            cv::putText(closest_face_display, "Closest Face", cv::Point(10, 25), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(255, 255, 255), 1);
            closest_face_display.copyTo(canvas(cv::Rect(frame.cols, 0, SIDEBAR_FACE_SIZE.width, SIDEBAR_FACE_SIZE.height)));

            final_view = canvas;
        }

        // 3. --- DISPLAY AND EXIT ---
        cv::imshow("Face Recognition", final_view);

        if (cv::waitKey(1) == 27) { // Exit on ESC key
            break;
        }
    }

    // 4. --- CLEANUP ---
    cap.release();
    cv::destroyAllWindows();

    return 0;
}