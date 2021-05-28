TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp

INCLUDEPATH += /usr/local/include/opencv /usr/local/include/opencv2
LIBS += `pkg-config  opencv --libs`

LIBS += -L/usr/local/lib/ \
        -lopencv_core \
        -lopencv_highgui \
        -lopencv_imgproc \
        -lopencv_objdetect \
        -lopencv_highgui \
         -lopencv_calib3d \
         -lopencv_video\
         -lopencv_videoio\
          -lopencv_imgcodecs\
         -lopencv_features2d \
          -lopencv_optflow
