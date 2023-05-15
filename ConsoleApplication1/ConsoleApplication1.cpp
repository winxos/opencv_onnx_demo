#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace cv;
using namespace cv::dnn;
using namespace std;

int main()
{
    String model_path = "cifar.onnx";
    Net net = readNetFromONNX(model_path);
    vector<string> labels = { "plane","car","bird","cat","deer","dog","frog","horse","ship","truck" };
    Mat img = imread("2.jpg");
    cv::Scalar mean = cv::Scalar(0.5, 0.5, 0.5);
    cv::Scalar std = cv::Scalar(0.5, 0.5, 0.5);

    //cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Resize the image to the required size
    cv::resize(img, img, cv::Size(224, 224));

    // Convert the image to float32 data type
    img.convertTo(img, CV_32F);
    img /= 255;
    cv::subtract(img, mean, img);
    cv::divide(img, std, img);
    cv::Mat input_blob = cv::dnn::blobFromImage(img);
    //cv::Mat blob = input_blob.reshape(1, 3, 224, 224);
    Mat blob = blobFromImage(img, 1, Size(224, 224), Scalar(), true, false);
    net.setInput(blob, "input");
    Mat outputMat = net.forward();
    cout << outputMat << endl;
    Mat softmaxOutput;
    cv::exp(outputMat, softmaxOutput);
    softmaxOutput /= cv::sum(softmaxOutput)[0];
    cout << softmaxOutput << endl;
    double maxVal;
    Point  maxLoc;
    cv::minMaxLoc(softmaxOutput.reshape(1, 1), nullptr, &maxVal, nullptr, &maxLoc);
    cout << maxLoc.x <<","<<maxVal<< endl;
    putText(img, labels[maxLoc.x], Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    imshow("Result", img);
    waitKey(0);
    return 0;
}

