#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <algorithm>

#include "opencv2/opencv.hpp"
#include "face_recognition.hpp"
#include "boost/filesystem.hpp"
#include "face_repository.hpp"
#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "face_align.h"

#define DIST_THRESHOLD 0.6
#define FACE_ALIGN_SIZE 192
#define FACE_ALIGN_LANDMARK FaceAlign::INNER_EYES_AND_TOP_LIP
//#define FACE_ALIGN_SCALE_FACTOR 0.2734
#define FACE_ALIGN_SCALE_FACTOR 0.26 // 25:100 as in CASIA dataset paper. Use 96*96 as input in small CNN, that's 25/96.
//#define FACE_ALIGN_LANDMARK FaceAlign::INNER_EYES_AND_BOTTOM_LIP
//#define FACE_ALIGN_SCALE_FACTOR 0.3385

namespace fs = ::boost::filesystem;

using namespace cv;
using namespace std;
using namespace face_rec_srzn;

std::vector<std::string> &split(const std::string &s, char delim,
                                std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

void getAllFiles(const fs::path &root, const string &ext,
                 vector<fs::path> &ret) {
    if (!fs::exists(root) || !fs::is_directory(root))
        return;

    fs::recursive_directory_iterator it(root);
    fs::recursive_directory_iterator endit;

    while (it != endit) {
        if (fs::is_regular_file(*it) && it->path().extension() == ext)
            ret.push_back(it->path());
        ++it;
    }
}

string getNewFileName(string ext) {
    srand((int)time(0));
    int num = rand() % 1000;
    struct tm *p;
    time_t second;
    time(&second);
    p = localtime(&second);
    char buf[100] = {'\0'};
    sprintf(buf, "%04d%02d%02d_%02d%02d%02d_%03d", 1900 + p->tm_year,
            1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec, num);
    return string(buf) + ext;
}

float dist2sim(float dist) {
    float maxDist = 2;
    if (dist < 0 || dist > maxDist)
        return 0;
    return (maxDist - dist) / maxDist;
}

// TODO
float affineDist(const Mat &H1, const Mat &H2) {
    assert(H1.size() == Size(2, 3));
    assert(H2.size() == Size(2, 3));
    return -1;
}

Mat affine2square(const Mat &H) {
    assert(H.size() == Size(2, 3));

    Mat M(3, 3, H.type());
    H.row(0).copyTo(M.row(0));
    H.row(1).copyTo(M.row(1));
    M.at<double>(2, 0) = 0.0;
    M.at<double>(2, 1) = 0.0;
    M.at<double>(2, 2) = 1.0;

    return M;
}


bool faceVerfication(LightFaceRecognizer &recognizer, FaceAlign &face_align,
                     fs::path &path, Mat & frame) {
    // Parameter settings.
    int num_select_sample = 5;
    double dist_threshold = DIST_THRESHOLD;

    cout << "Verification: " << path.string() << endl;

    string winName = "Face Verification";
    namedWindow(winName);

    // Get register faces.
    vector<fs::path> all_image_path;
    getAllFiles(path, ".jpg", all_image_path);
    if (0 >= all_image_path.size()) {
        cerr << "The person with name " << path.filename() << " does not exist!"
             << endl;
        return false;
    }

    // Select face samples and compute their feature.
    vector<Mat> features;
    Mat sample_to_show;
    srand(time(0));
    for (int i = 0; i < num_select_sample && i < all_image_path.size(); i++) {
        int j = rand() % all_image_path.size();
        Mat face = imread(all_image_path[j].string());
        Mat feature;
        recognizer.ExtractFaceFeature(face, feature);
        features.push_back(feature);
        if (0 == i) {
            resize(face, sample_to_show, Size(100, 100));
        }
    }

    // Detect face and show.
    Rect face_rect;
    face_align.detectFace(frame, face_rect);
    if (face_rect.width <= 0 || face_rect.height <= 0) {
        imshow(winName, frame);
        cout<<" No valid face detected!"<<endl;
        waitKey(5000);
        return false;
    }

    // Extract face feature
    Mat face = face_align.align(frame, face_rect, FACE_ALIGN_SIZE,
                                FACE_ALIGN_LANDMARK, FACE_ALIGN_SCALE_FACTOR);
    Mat f;
    recognizer.ExtractFaceFeature(face, f);

    // Compare with register face features.
    vector<double> dists;
    for (int j = 0; j < features.size(); j++) {
        dists.push_back(norm(f, features[j]));
    }
    vector<double>::iterator min_iter = min_element(dists.begin(), dists.end());
    cout << ": " << *min_iter << endl;

    // Show image.
    Mat im_show;
    frame.copyTo(im_show);
    rectangle(im_show, face_rect, Scalar(255, 0, 255));
    resize(im_show, im_show, Size(480, 480));
    Mat roi = Mat(im_show, Rect(0, im_show.rows - 100, 100, 100));
    sample_to_show.copyTo(roi);
    char buf[20] = {'\0'};
    sprintf(buf, "%4f", dist2sim(*min_iter));
    putText(im_show, buf, Point(20, 50), CV_FONT_HERSHEY_COMPLEX, 1,
            Scalar(255, 0, 0));
    string result = *min_iter < dist_threshold ? "SUCCESS" : "FAIL";
    putText(im_show, result, Point(im_show.cols / 2, 100), CV_FONT_HERSHEY_COMPLEX,
            1, Scalar(255, 0, 0));
    imshow(winName, im_show);
    waitKey(5000);

    return *min_iter < dist_threshold;
}

bool checkFacePose(FaceAlignTransVar & V) {
    float t_rotation = 15;        // Rotation angle
    float t_pitching = 0.2; // Pitching rate (good in raise face). Use the change of mid_eye_to_top_lip/whole_face_height as the approximate estimate of the face pitching rate.
    float t_ar_change = 0.1; // Change of face aspect ratio (fair in down face, good in face fragment).
    float t_lr_change = 0.5; // Change of width_of_left_face / width_of_right_face

    if(abs(V.rotation) > t_rotation ||
            abs(V.pitching) > t_pitching ||
            abs(V.ar_change) > t_ar_change ||
            abs(V.lr_change) > t_lr_change)
        return false;

    return true;
}

bool faceVerfication(LightFaceRecognizer &recognizer, FaceAlign &face_align,
                     fs::path &path, VideoCapture &cap) {
    // Parameter settings.
    int num_max_check_frame = 150;
    int num_select_sample = 5;
    int num_min_accept_frame = 5;
    double dist_threshold = DIST_THRESHOLD;
    int size_face_least = 160;

    cout << "Verification: " << path.string() << endl;

    string winName = "Face Verification";
    namedWindow(winName);

    // Get register faces.
    vector<fs::path> all_image_path;
    getAllFiles(path, ".jpg", all_image_path);
    if (0 >= all_image_path.size()) {
        cerr << "The person with name " << path.filename() << " does not exist!"
             << endl;
        return false;
    }

    // Select face samples and compute their feature.
    vector<Mat> features;
    Mat sample_to_show;
    srand(time(0));
    for (int i = 0; i < num_select_sample && i < all_image_path.size(); i++) {
        int j = rand() % all_image_path.size();
        Mat face = imread(all_image_path[j].string());
        Mat feature;
        recognizer.ExtractFaceFeature(face, feature);
        features.push_back(feature);
        if (0 == i) {
            resize(face, sample_to_show, Size(100, 100));
        }
    }

    Mat frame;
    bool is_start = false;
    int num_accepted = 0;
    for (int i = 0; i <= num_max_check_frame; i = i + (is_start ? 1 : 0)) {
        // Get frame.
        cap >> frame;
        Mat origin;
        frame.copyTo(origin);

        // Sample face to compare. Show it in the corner.
        Mat roi = Mat(frame, Rect(0, frame.rows - 100, 100, 100));
        sample_to_show.copyTo(roi);

        // Detect face and show.
        Rect face_rect;
        face_align.detectFace(origin, face_rect);
        if (face_rect.width < size_face_least || face_rect.height < size_face_least) {
            imshow(winName, frame);
            waitKey(30);
            continue;
        }

        // Extract face feature
        Mat H;  // Affine matrix in face alignment
        FaceAlignTransVar V;    // Face pose variable
        Mat face = face_align.align(origin, H, V, face_rect, FACE_ALIGN_SIZE,
                                    FACE_ALIGN_LANDMARK, FACE_ALIGN_SCALE_FACTOR);
        Mat f;  // feature
        recognizer.ExtractFaceFeature(face, f);

        // Check face pose.
        bool is_valid_pose = checkFacePose(V);

        // Compare with register face features.
        vector<double> dists;
        double min_dist; // Minimum distance to register faces.
        for (int j = 0; j < features.size(); j++) {
            dists.push_back(norm(f, features[j]));
        }
        vector<double>::iterator min_iter = min_element(dists.begin(), dists.end());
        min_dist = *min_iter;
        cout << i << ": " << *min_iter << endl;

        // Check verification criteria.
        if (is_start && min_dist < dist_threshold && is_valid_pose) {
            num_accepted++;
        }

        // Show image.
        rectangle(frame, face_rect, Scalar(255, 0, 255));
        stringstream buf;
        buf<<"Similarity: "<<dist2sim(min_dist);
        putText(frame, buf.str(), Point(10, 30), CV_FONT_HERSHEY_COMPLEX, 0.7, Scalar(255, 0, 0));
        buf.str("");
        buf<<"Accept faces: "<<num_accepted<<"/"<<num_min_accept_frame;
        putText(frame, buf.str(), Point(10, 60), CV_FONT_HERSHEY_COMPLEX, 0.7, Scalar(255, 0, 0));
        if (is_start) {
            buf.str("");
            buf << (num_accepted == num_min_accept_frame
                    ? "SUCCESS"
                    : (i == num_max_check_frame ? "FAIL" : ""));
            putText(frame, buf.str(), Point(10, 90), CV_FONT_HERSHEY_COMPLEX, 0.7, Scalar(255, 0, 0));
        } else {
            putText(frame, "Press SPACE to start.", Point(10, 90), CV_FONT_HERSHEY_COMPLEX, 0.7, Scalar(0, 0, 255));
        }
        // Show face pose variables.
        buf.str("");
        buf <<"Rotation: "<<V.rotation;
        putText(frame, buf.str(), Point(frame.cols-250, 30), CV_FONT_HERSHEY_COMPLEX, 0.7, Scalar(255, 0, 0));
        buf.str("");
        buf <<"Left/right: "<<V.lr_change;
        putText(frame, buf.str(), Point(frame.cols-250, 60), CV_FONT_HERSHEY_COMPLEX, 0.7, Scalar(255, 0, 0));
        buf.str("");
        buf <<"Aspect: "<<V.ar_change;
        putText(frame, buf.str(), Point(frame.cols-250, 90), CV_FONT_HERSHEY_COMPLEX, 0.7, Scalar(255, 0, 0));
        buf.str("");
        buf <<"Pitching: "<<V.pitching;
        putText(frame, buf.str(), Point(frame.cols-250, 120), CV_FONT_HERSHEY_COMPLEX, 0.7, Scalar(255, 0, 0));
        if (!is_valid_pose) {
            putText(frame, "Plz face agaist cam.", Point(frame.cols-250, 150), CV_FONT_HERSHEY_COMPLEX, 0.7, Scalar(0, 0, 255));
            if (is_start)
                putText(frame, "Press SPACE to use the face.", Point(10, 90), CV_FONT_HERSHEY_COMPLEX, 0.7, Scalar(0, 0, 255));
        }
        imshow(winName, frame);

        // Wait key.
        char key;
        if (is_start &&
                (num_accepted == num_min_accept_frame || i == num_max_check_frame)) {
            key = (char)waitKey(-1);
            return num_accepted == num_min_accept_frame;
        } else
            key = (char)waitKey(30);
        switch (key) {
        case 27:
            return false;
        case ' ':
            if (!is_start)
                is_start = true;
            else if (!is_valid_pose) {
                num_accepted += min_dist < dist_threshold ? 1 : 0;
            }
            break;
        }
    }
}

bool faceRegister(LightFaceRecognizer &recognizer, FaceAlign &face_align,
                  fs::path &path, Mat & frame) {
    cout << "Register: " << path.string() << endl;
    string winName = "Face Register";
    namedWindow(winName);

    Rect face_rect;
    face_align.detectFace(frame, face_rect);
    if (face_rect.width <=0 || face_rect.height <= 0) {
        cout<<"No valid face detected in the given image."<<endl;
        return false;
    }

    Mat image;
    frame.copyTo(image);
    rectangle(image, face_rect, Scalar(255, 0, 255));
    imshow(winName, image);
    Mat face = face_align.align(frame, face_rect, FACE_ALIGN_SIZE,
                                FACE_ALIGN_LANDMARK, FACE_ALIGN_SCALE_FACTOR);
    string filename = getNewFileName(".jpg");
    fs::path save_path = path / filename;
    cout << "Save face image to: " << save_path.string() << endl;
    fs::create_directories(path);
    imwrite(save_path.string(), face);
    waitKey(5000);
    return true;
}

bool faceRegister(LightFaceRecognizer &recognizer, FaceAlign &face_align,
                  fs::path &path, VideoCapture &cap) {
    int size_face_least = 160;

    cout << "Register: " << path.string() << endl;
    string winName = "Face Register";
    namedWindow(winName);
    Mat frame;
    for (;;) {
        cap >> frame;
        Mat origin = frame.clone();

        Rect face_rect;
        face_align.detectFace(frame, face_rect);
        bool face_detected = true;
        if (face_rect.width < size_face_least || face_rect.height < size_face_least )
            face_detected = false;
        if (face_detected)
            rectangle(frame, face_rect, Scalar(255, 0, 255));
        imshow(winName, frame);

        char key = (char)waitKey(30);
        switch (key) {
        case 27:
            return false;
        case ' ':
            if (face_detected) {
                Mat face = face_align.align(origin, face_rect, FACE_ALIGN_SIZE,
                                            FACE_ALIGN_LANDMARK, FACE_ALIGN_SCALE_FACTOR);
                string filename = getNewFileName(".jpg");
                fs::path save_path = path / filename;
                cout << "Save face image to: " << save_path.string() << endl;
                fs::create_directories(path);
                imwrite(save_path.string(), face);
                waitKey(-1);
                return true;
            }
        }
    }
}

int main(int argc, char **argv) {
    // Initial face recognizer and face aligner
    LightFaceRecognizer recognizer(
                "../../models/cnn", "../../models/bayesian_model_lfw.bin", "prob", false);
    FaceAlign face_align(
                "../../models/dlib_shape_predictor_68_face_landmarks.dat");
    string image_root("../../images");
    int camera_index = 0;

    string action, name, image_path;
    Mat image;
    if (3 <= argc) {
        action = argv[1];
        name = argv[2];
    }
    if ( 4 == argc )
        image_path = argv[3];

    if (action.empty() || name.empty() || "ver" != action && "reg" != action) {
        cout << "Usage: ./face_ver [reg|ver] [person_name] [image_path]" << endl;
        cout << "Press Space to start process, ESC to close window" << endl;
        return -1;
    }

    // Get face image path
    fs::path folder(image_root);
    folder /= name;

    // Get image
    if ( !image_path.empty() ){
        image = imread(image_path);
    }

    // Get a handle to the camera
    VideoCapture * cap = NULL;
    if ( image.empty() ) {
        cap = new VideoCapture(camera_index);
        if (!cap->isOpened()) {
            cerr << "Camera " << camera_index << " cannot be opened." << endl;
            delete cap;
            return -1;
        }
    }

    // Action
    if ("ver" == action) {
        if (NULL != cap)
            faceVerfication(recognizer, face_align, folder, *cap);
        else
            faceVerfication(recognizer, face_align, folder, image);
    } else {
        if (NULL != cap)
            faceRegister(recognizer, face_align, folder, *cap);
        else
            faceRegister(recognizer, face_align, folder, image);
    }

    delete cap;
    return 0;
}
