#include "FeatureExtractor.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace mgnc::slam
{
    void FeatureExtractor::ComputePyramid(cv::Mat image)
    {
        using namespace mgnc::slam;
        for (int level = 0; level < nlevels; ++level)
        {
            float scale = mvInvScaleFactor[level];
            cv::Size sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));
            cv::Size wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
            cv::Mat temp(wholeSize, image.type()), masktemp;
            mvImagePyramid[level] = temp(cv::Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

            // Compute the resized image
            if( level != 0 )
            {
                cv::resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, cv::INTER_LINEAR);

                cv::copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               cv::BORDER_REFLECT_101+cv::BORDER_ISOLATED);
            }
            else
            {
                cv::copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               cv::BORDER_REFLECT_101);
            }
        }

    }
}