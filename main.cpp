/*********************************
*C++ code to detect human fall from real time video.The code tested and worked with cylinderical shape as demonstration.
* Written by: Tadewos Somano
* Date:Sept/2018
*********************************/
#include <math.h>
#include <time.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <numeric>

#include <opencv2/video/tracking.hpp>
#include <opencv2/optflow/motempl.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

const float MHI_DURATION =  0.5;
int DEFAULT_THRESHOLD = 32;
const float MAX_TIME_DELTA = 12500.0;
const float MIN_TIME_DELTA = 5;
int visual_trackbar = 2;
const bool isFall=false;
double frameSkipFactor=1.17;



/* For conclusion if it is actually fall or not */
Mat frame_20;
Mat frame_30;
Mat image;

std::vector<double> y_coordinates;
std::vector<double> num_cols_per_width;

void processImage(Mat);
double calculateAxisDeviation(Mat,Mat);
double calculateThetaDeviation(Mat,Mat);
double calculateMotionCoefficient(Mat,Mat);

void calculate_y_deviation(Mat);
bool checkMinimumContourSize(Mat);
bool isYCoordGreater();
bool detectHuman(Mat);

int main()
{

    double thetaDeviation=0;
    double axisDeviation=0;
    double motionCoefficient=0;

    int counter=0;
    Mat fgMaskMOG;
    Mat fgMaskMOG2;
    Mat fgMaskMOG3;
    Mat fgMaskMOG4;

    Ptr<BackgroundSubtractor> bg_model1 = createBackgroundSubtractorMOG2();
    Ptr<BackgroundSubtractor> bg_model2=createBackgroundSubtractorMOG2();
    Ptr<BackgroundSubtractor> bg_model3=createBackgroundSubtractorMOG2();
    Ptr<BackgroundSubtractor> bg_model4 = createBackgroundSubtractorMOG2();
    namedWindow("Motion_tracking",CV_WINDOW_AUTOSIZE);
    string values[4] = {"input", "frame_diff", "motion_hist", "grad_orient"};
    createTrackbar( "visual", "Motion_tracking", &visual_trackbar, 3, NULL );
    createTrackbar("threshold", "Motion_tracking", &DEFAULT_THRESHOLD, 255, NULL);
    
     VideoCapture cap;
     cap.open(0); //capture from camera 

    const int skipRate=cap.get(CV_CAP_PROP_FPS)*frameSkipFactor;
    Mat frames[skipRate];
    std::cout<<"frame rate is :  "<<cap.get(CV_CAP_PROP_FPS)<<std::endl;

    if ( !cap.isOpened() )  // if not success, exit program
     {
        cout << "Cannot open the video file" << endl;
            return -1;
      }
    Mat frame, ret, frame_diff, gray_diff, motion_mask, current_frame;

    cap.read(frame);
    ret = frame.clone();
    Size frame_size = frame.size();
    int h = frame_size.height;
    int w = frame_size.width;

    Mat prev_frame(h,w, CV_8UC3,Scalar(0,0,0));// = frame.clone();
    Mat motion_history(h,w, CV_32FC1,Scalar(0,0,0));
    Mat hsv(h,w, CV_8UC3,Scalar(0,255,0));
    Mat mg_mask(h,w, CV_8UC1,Scalar(0,0,0));
    Mat mg_orient(h,w, CV_32FC1,Scalar(0,0,0));
    Mat seg_mask(h,w, CV_32FC1,Scalar(0,0,0));
    vector<Rect> seg_bounds;
    String visual_name;
    Mat vis(h,w,CV_32FC3);
    Mat vis1(h,w,CV_8UC1);
    Mat silh_roi,orient_roi,mask_roi,mhi_roi;

    while(1)
     {
        cap.retrieve(frame);
        cap.read(frame);

        ret = frame.clone();
        if (!ret.data) //if not success, break loop
         {
            cout << "video ended" << endl;
            break;
          }
        frames[counter%skipRate]=frame.clone();
        absdiff(frame, prev_frame, frame_diff);
        cvtColor(frame_diff,gray_diff, CV_BGR2GRAY );
        threshold(gray_diff,ret,DEFAULT_THRESHOLD,255,0);

        motion_mask = ret.clone();

        double timestamp = 1000.0*clock()/CLOCKS_PER_SEC;
        cv::motempl::updateMotionHistory(motion_mask, motion_history, timestamp, MHI_DURATION);

        if(counter>=(skipRate-1)){
         prev_frame= frames[counter%skipRate];
         current_frame= frames[(counter+1)%skipRate];

         fgMaskMOG.create( current_frame.size(),  current_frame.type());
         bg_model1->apply( current_frame, fgMaskMOG, true ? -1 : 0);

         fgMaskMOG2.create(prev_frame.size(), prev_frame.type());
         bg_model2->apply(prev_frame, fgMaskMOG2, true ? -1 : 0);


        fgMaskMOG4.create(frame.size(), frame.type());
        bg_model4->apply(motion_history, fgMaskMOG4, true ? -1 : 0);
        imshow("Motion_tracking",fgMaskMOG);
 
     //The minimum number of contour points required by OpenCV fitEllipse function  is 5
       if( checkMinimumContourSize(fgMaskMOG.clone())&& checkMinimumContourSize(fgMaskMOG2.clone())){
         processImage(fgMaskMOG.clone());

         motionCoefficient=calculateMotionCoefficient(motion_history,fgMaskMOG2);
         thetaDeviation=calculateThetaDeviation(fgMaskMOG2, fgMaskMOG);

         axisDeviation=calculateAxisDeviation(fgMaskMOG2, fgMaskMOG);

         std::cout<<"Motion coefficient is :"<<motionCoefficient<<std::endl;
         std::cout<<"Theta deviation is :"<<thetaDeviation<<std::endl;
         std::cout<<"Axis deviation is :"<<axisDeviation<<std::endl;
         std:: cout<<"Frame number is: "<<cap.get(CV_CAP_PROP_POS_FRAMES)<< std::endl;

         if(motionCoefficient>0.8)
          {
            if(thetaDeviation>15 && axisDeviation>0.8)
              {               

                  std:: cout<<"Detected human-fall at frame number: "<<cap.get(CV_CAP_PROP_POS_FRAMES)<< std::endl;
            }
          }
        }
      }
       //std::cout<<"out of need to skip frames"<<std::endl;
       cv::motempl::calcMotionGradient(motion_history, mg_mask, mg_orient, MIN_TIME_DELTA, MAX_TIME_DELTA, 3);
       cv::motempl::segmentMotion(motion_history, seg_mask, seg_bounds, timestamp, 32);
       visual_name = values[visual_trackbar];

        if(visual_name == "input")
            vis = frame.clone();
        else if(visual_name == "frame_diff")
            vis = frame_diff.clone();
        else if(visual_name == "motion_hist")
        {

            for(int i=0; i< motion_history.cols; i++)
                {
                    for(int j=0; j< motion_history.rows ; j++)
                    {
                       float a = motion_history.at<float>(j,i);
                       if((a-timestamp-MHI_DURATION)/MHI_DURATION <= -5)
                           vis1.at<uchar>(j,i) = 0;
                        else
                            vis1.at<uchar>(j,i) = (a-timestamp-MHI_DURATION)/MHI_DURATION;
                    }
               }

            cvtColor(vis1,vis,COLOR_GRAY2BGR);
        }
        else if(visual_name == "grad_orient")
        {
            for(int i=0; i< hsv.cols; i++)
                {
                    for(int j=0; j< hsv.rows ; j++)
                    {
                      float a = (mg_orient.at<float>(j,i))/2;

                       hsv.at<Vec3b>(j,i)[0] = a;
                       float b = (mg_mask.at<uchar>(j,i))*255;
                       hsv.at<Vec3b>(j,i)[2] = b;
                   }
             }
            cvtColor(hsv,vis,COLOR_HSV2BGR);
        }

        for(unsigned int h = 0; h < seg_bounds.size(); h++)
         {
            Rect rec = seg_bounds[h];
            if(rec.area() > 5000 && rec.area() < 70000)
               {
                 silh_roi = motion_mask(rec);
                 orient_roi = mg_orient(rec);
                 mask_roi = mg_mask(rec);
                 mhi_roi = motion_history(rec);
                 if(norm(silh_roi, NORM_L2, noArray()) > rec.area()*0.5)
                  {
                    double angle = cv::motempl::calcGlobalOrientation(orient_roi, mask_roi, mhi_roi,timestamp, MHI_DURATION);

                  }
              }
        }

    counter++;

            if(waitKey(30) >= 0)
        {
            cout << "esc key is pressed by user" << endl;
            break;
        }

    }

    return 0;

}

bool checkMinimumContourSize(Mat foreGroundImage){

    bool hasMoreThan5Contours=true;
    vector<vector<Point>> contours;
    findContours(foreGroundImage, contours,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);
    double x=contourArea(contours[0]);
         for(size_t i=0;i<contours.size();i++){
           if(contourArea(contours[i])>x)
            {
              x=contourArea(contours[i]);

              contours[0]=contours[i];
           }
         }
   if(contours[0].size()<6){
       hasMoreThan5Contours=false;
   }
         return hasMoreThan5Contours;
}

void processImage(Mat bimage )
{
    RotatedRect box;

    vector<vector<Point>> contours;

    findContours(bimage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

     double x=contourArea(contours[0]);

     for(size_t i=0;i<contours.size();i++){
         if(contourArea(contours[i])>x)
         {
           x=contourArea(contours[i]);

           contours[0]=contours[i];
         }
     }

    Mat cimage=Mat::zeros(bimage.size(), CV_8UC3);

    box = fitEllipse(contours[0]);

    Point2f vtx[4];

    box.points(vtx);

    for( int j = 0; j < 4; j++ )
            line(cimage, vtx[j], vtx[(j+1)%4], Scalar(0,255,0), 1, CV_AA);

   std::cout<<"Angle is  :"<<box.angle<<std::endl;

   std::cout<<"width is  :"<<box.size.width<<":  height is :"<<box.size.height<<std::endl;
   imshow("result", cimage);
}

double calculateThetaDeviation(Mat frame1,Mat frame2){

    double theta_deviation=0;

    Mat img1,img2;

    vector<Vec4i> hierarchy;

    RotatedRect box;

    RotatedRect box2;

    double mean_theta;

    vector<vector<Point>>contours;

    vector< vector<Point>>contours2;

    findContours(frame1, contours,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);

    findContours(frame2, contours2,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);
   
    //select contour with largest area

     double x=contourArea(contours[0]);

     for(size_t i=0;i<contours.size();i++){
       if(contourArea(contours[i])>x)
        {
          x=contourArea(contours[i]);

          contours[0]=contours[i];
       }
     }

     box = fitEllipse(contours[0]);
     double y=contourArea(contours2[0]);
      for(size_t i=0;i<contours2.size();i++){
        if(contourArea(contours2[i])>y)
         {
           y=contourArea(contours2[i]);
           contours2[0]=contours2[i];

         }
      }

    box2 = fitEllipse(contours2[0]);
     
    double angle1=box.angle;
    
    double angle2=box2.angle;

    mean_theta=(angle1+angle2)/2;
    theta_deviation=sqrt((pow((angle1-mean_theta),2)+pow((angle2-mean_theta),2))/2);
    return theta_deviation;
}

double calculateAxisDeviation(Mat frame1,Mat frame2){
    double axis_deviation=0;
    Mat img1,img2;
    vector<Vec4i> hierarchy;
    RotatedRect box;
    RotatedRect box2;
    double ratio1;
    double ratio2;
    double mean_axis;
    double box1_height,box1_width,box2_height,box2_width;
    vector<vector<Point>>contours;
    vector< vector<Point>>contours2;
    findContours(frame1, contours, CV_RETR_LIST,CV_CHAIN_APPROX_NONE);

    findContours(frame2, contours2,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);
    double x=contourArea(contours[0]);
     for(size_t i=0;i<contours.size();i++){
       if(contourArea(contours[i])>x)
       {
         x=contourArea(contours[i]);
         contours[0]=contours[i];
       }
   }

       box = fitEllipse(contours[0]);
       double y=contourArea(contours2[0]);
        for(size_t i=0;i<contours2.size();i++){
       if(contourArea(contours2[i])>y)
       {
         y=contourArea(contours2[i]);
         contours2[0]=contours2[i];
       }
   }


       box2=fitEllipse(contours2[0]);
       if((box.angle>45&&box.angle<=90)||(box.angle<135 && box.angle >=90)){
           box1_width=box.size.height;
           box1_height=box.size.width;
        }
       else{
           box1_width=box.size.width;
           box1_height=box.size.height;

       }
       //TO DO: how to fitEllipse Mat image
       ratio1= box1_height/box1_width;
       if((box2.angle>45&&box2.angle<=90)||(box2.angle<135 && box2.angle >=90)){
           box2_width=box2.size.height;
           box2_height=box2.size.width;
        }
       else{
           box2_width=box2.size.width;
           box2_height=box2.size.height;

       }
       ratio2=box2_height/box2_width;
       mean_axis=(ratio1+ratio2)/2;

       axis_deviation=sqrt((pow((ratio1-mean_axis),2)+pow((ratio2-mean_axis),2))/2);

       return axis_deviation;
}

double calculateMotionCoefficient(Mat motion_history,Mat fgMaskMOG)
{
    double sum1=0,sum2=0;
    for(int i = 0; i < motion_history.rows; i++)
    {
        for(int j = 0; j <motion_history.cols; j++)

        sum1+= motion_history.at<uchar>(i,j);
    }
    for(int i = 0; i < fgMaskMOG.rows; i++)
    {
        for(int j = 0; j <fgMaskMOG.cols; j++)

       sum2+= fgMaskMOG.at<uchar>(i,j);
    }
   return (sum1/sum2);

}

void calculate_y_deviation(Mat frame){


     std::vector<std::vector<Point>>contours;
     RotatedRect box;
     findContours(frame, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
     for(size_t i = 0; i < contours.size(); i++)
     {
        size_t count = contours[i].size();
        if( count < 50 )
        {
             continue;
        }
        Mat pointsf1;
        Mat(contours[i]).convertTo(pointsf1, CV_32F);
        box = fitEllipse(pointsf1);
    }

   /*y_deviation for 10 frames after fall > (frame.cols/8) for frames f0 tp f9 after fall*/
   y_coordinates.push_back(box.center.y);
   num_cols_per_width.push_back(frame.cols);

}

bool isYCoordGreater(){
   double deviation_of_y_coords=0;
   double average_of_y_coords = (std::accumulate(y_coordinates.begin(), y_coordinates.end(), 0))/y_coordinates.size();
   for(auto n:y_coordinates)
     deviation_of_y_coords+=sqrt(pow((n-average_of_y_coords),2)/y_coordinates.size());
   for(size_t i=0;i<num_cols_per_width.size();i++)
     {
        if((num_cols_per_width[i]/8)>=deviation_of_y_coords)
           return false;
      }

   return true;
}

bool detectHuman(Mat img)
{
   bool isHuman=true;
   HOGDescriptor hog;
   hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
        if (!img.data) {
             std::cout<<"Could not read image"<<std::endl;
             isHuman= false;
          }
    vector<Rect> found, found_filtered;
    hog.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.05, 2);
    size_t i, j;
    for (i=0; i<found.size(); i++)
       {
         Rect r = found[i];
         for (j=0; j<found.size(); j++){
            if (j!=i && (r & found[j])==r) {
               std::cout<<"Not human"<<std::endl;
               isHuman= false;
               break;
              }
            if (j==found.size())
                found_filtered.push_back(r);
         }
        }

    return isHuman;
    }




