#pragma once
#include "FaceTransformer.h"

class PoseTransformer : public FaceTransformer
{
public:
    PoseTransformer();

    PoseTransformer(string imagePath,
                    int normalizeHeight,
                    int normalizeWidth,
                    int pointNum);

    PoseTransformer(cv::Mat image,
                    int normalizeHeight,
                    int normalizeWidth,
                    int pointNum);
};
