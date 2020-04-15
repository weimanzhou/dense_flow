//
// Created by snowflake on 2020/4/15.
//
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <zconf.h>

using namespace cv;
using namespace std;

static void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y, double lowerBound,
                               double higherBound) {
#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
    for (int i = 0; i < flow_x.rows; ++i) {
        for (int j = 0; j < flow_y.cols; ++j) {
            float x = flow_x.at<float>(i, j);
            float y = flow_y.at<float>(i, j);
            img_x.at<uchar>(i, j) = CAST(x, lowerBound, higherBound);
            img_y.at<uchar>(i, j) = CAST(y, lowerBound, higherBound);
        }
    }
#undef CAST
}

static void drawOptFlowMap(const Mat &flow, Mat &cflowmap, int step, double, const Scalar &color) {
    for (int y = 0; y < cflowmap.rows; y += step)
        for (int x = 0; x < cflowmap.cols; x += step) {
            const Point2f &fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),
                 color);
            circle(cflowmap, Point(x, y), 2, color, -1);
        }
}

static int convert(const string vid_file, string x_flow_path, string y_flow_path, string image_path, int bound) {
    VideoCapture capture(vid_file);
    if (!capture.isOpened()) {
        printf("Could not initialize capturing..\n");
        return -1;
    }

    int frame_num = 0;
    Mat image, prev_image, prev_grey, grey, frame, flow, cflow;

    while (true) {
        capture >> frame;
        if (frame.empty())
            break;

        if (frame_num == 0) {
            image.create(frame.size(), CV_8UC3);
            grey.create(frame.size(), CV_8UC1);
            prev_image.create(frame.size(), CV_8UC3);
            prev_grey.create(frame.size(), CV_8UC1);

            frame.copyTo(prev_image);
            cvtColor(prev_image, prev_grey, CV_BGR2GRAY);

            frame_num++;
            continue;
        }

        frame.copyTo(image);
        cvtColor(image, grey, CV_BGR2GRAY);

        // calcOpticalFlowFarneback(prev_grey,grey,flow,0.5, 3, 15, 3, 5, 1.2, 0 );
        calcOpticalFlowFarneback(prev_grey, grey, flow, 0.702, 5, 10, 2, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN);

        // prev_image.copyTo(cflow);
        // drawOptFlowMap(flow, cflow, 12, 1.5, Scalar(0, 255, 0));

        Mat flows[2];
        split(flow, flows);
        Mat imgX(flows[0].size(), CV_8UC1);
        Mat imgY(flows[0].size(), CV_8UC1);
        convertFlowToImage(flows[0], flows[1], imgX, imgY, -bound, bound);
        char tmp[20];
        sprintf(tmp, "_%04d.jpg", int(frame_num));
        imwrite(x_flow_path + "/" + tmp, imgX);
        imwrite(y_flow_path + "/" + tmp, imgY);
        imwrite(image_path + "/" + tmp, image);

        std::swap(prev_grey, grey);
        std::swap(prev_image, image);
        frame_num = frame_num + 1;
    }
}

static int avi_file_count(string source) {
    int count = 0;
    // 遍历源文件下的所有 avi 格式的视频
    struct stat s;
    stat(source.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) throw "路径不是一个文件夹";
    DIR *open_dir = opendir(source.c_str());
    if (NULL != open_dir) {
        dirent *p = nullptr;
        while ((p = readdir(open_dir)) != nullptr) {
            // 获取到文件名
            string name = string(p->d_name);
            // 如果文件的后缀名是 avi 则处理
            if (strncmp(name.c_str(), ".", 1) == 0 || strncmp(name.c_str(), "..", 2) == 0) continue;
            if (strncmp(name.substr(name.find_last_of(".") + 1, name.length()).c_str(), "avi", 3) == 0) {
                count++;
            }
        }
    }
    return count;
}

int main(int argc, char **argv) {
    // IO operation
    const char *keys =
            {
                    "{ S sourceFile   | source  | directory of video }"
                    "{ T targetFile   | target  | directory of flow x component, flow y component, image }"
                    "{ b bound        | 15      | specify the maximum of optical flow}"
                    "{ t type         | 0       | specify the optical flow algorithm }"
                    "{ d device_id    | 0       | set gpu id}"
                    "{ s step         | 1       | specify the step for frame sampling}"
            };

    CommandLineParser cmd(argc, argv, keys);
    // 提示软件版本
    cmd.about("Application name v1.0.0");

//    if (cmd.has("help")) {
//        cmd.printMessage();
//        return 0;
//    }

    // 获取到源文件夹
    string source = cmd.get<string>("sourceFile");
    string target = cmd.get<string>("targetFile");
    int bound = cmd.get<int>("bound");
    int type = cmd.get<int>("type");
    int gpu_device_id = cmd.get<int>("device_id");
    int step = cmd.get<int>("step");

//    printf("%s\n", source.c_str());
//    printf("%s\n", target.c_str());
//    printf("%d\n", bound);

    if (!cmd.check()) {
        cmd.printErrors();
        return 0;
    }

    // 遍历源文件下的所有 avi 格式的视频
    struct stat s;
    stat(source.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) throw "路径不是一个文件夹";
    DIR *open_dir = opendir(source.c_str());
    if (NULL != open_dir) {
        dirent *p = nullptr;
        // 循环读取某个文件夹下的文件
        // 获取有多少个 avi 文件
        int total_count = avi_file_count(source);
        int cur_count = 0;
        while ((p = readdir(open_dir)) != nullptr) {
            // 获取到文件名
            string name = string(p->d_name);
            // 如果文件的后缀名是 avi 则处理
            if (strncmp(name.c_str(), ".", 1) == 0 || strncmp(name.c_str(), "..", 2) == 0) continue;
            if (strncmp(name.substr(name.find_last_of(".") + 1, name.length()).c_str(), "avi", 3) == 0) {
                // 根据文件名创建文件夹以及子文件（x_flow, y_flow, image）；
                // 在目标文件夹下创建 x_flow, y_flow, image
                string path = source + "/" + name;
                string x_flow_path = target + "/" + name.substr(0, name.find_last_of(".")) + "/x_flow";
                string y_flow_path = target + "/" + name.substr(0, name.find_last_of(".")) + "/y_flow";
                string image_path = target + "/" + name.substr(0, name.find_last_of(".")) + "/image";

//                printf("%s\n", path.c_str());
//                printf("%s\n", x_flow_path.c_str());
//                printf("%s\n", y_flow_path.c_str());
//                printf("%s\n", image_path.c_str());

                if (0 != access(x_flow_path.c_str(), F_OK)) {
                    system(("mkdir -p " + x_flow_path).c_str());
                    printf("%d", 1);
                }
                if (0 != access(y_flow_path.c_str(), F_OK)) {
                    system(("mkdir -p " + y_flow_path).c_str());
                    printf("%d", 2);
                }
                if (0 != access(image_path.c_str(), F_OK)) {
                    system(("mkdir -p " + image_path).c_str());
                    printf("%d", 3);
                }

                char tmp[100];
                sprintf(tmp, "[%03d / %03d][ ing] : %-40s", cur_count, total_count, path.c_str());
                printf("%s\n", tmp);
                convert(path, x_flow_path, y_flow_path, image_path, bound);
                cur_count++;
                sprintf(tmp, "\r\r[%03d / %03d][done] : %-40s", cur_count, total_count, path.c_str());
                printf("%s\n", tmp);
            }
        }
    }

    return 0;
}
