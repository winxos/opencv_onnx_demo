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
    Mat image = imread("3.jpg");

    Size input_size = Size(224, 224);
    Scalar mean = Scalar(0.5, 0.5, 0.5);
    Mat blob = blobFromImage(image, 1 / 255.0, input_size, mean, true, false);
    net.setInput(blob, "input");
    Mat outputMat = net.forward();
    cout << outputMat << endl;
    cout << outputMat.reshape(1, 1) << endl;
    Mat softmaxOutput;
    cv::exp(outputMat, softmaxOutput);
    softmaxOutput /= cv::sum(softmaxOutput)[0];
    cout << softmaxOutput << endl;
    double maxVal;
    Point  maxLoc;
    cv::minMaxLoc(softmaxOutput.reshape(1, 1), nullptr, &maxVal, nullptr, &maxLoc);
    cout << maxLoc.x <<","<<maxVal<< endl;
    putText(image, labels[maxLoc.x], Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    imshow("Result", image);
    waitKey(0);

    return 0;
}
