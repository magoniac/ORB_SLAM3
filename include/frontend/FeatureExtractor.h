#pragma once
#include <vector>
#include <opencv2/core/mat.hpp>
// #include <opencv2/core.hpp>

namespace mgnc::slam
{
    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;
    const int EDGE_THRESHOLD = 19;

    class FeatureExtractor
    {
    public:
        
        FeatureExtractor(int nfeatures, float scaleFactor, int nlevels,
                    int iniThFAST, int minThFAST);

        virtual ~FeatureExtractor() { }

        // Compute the ORB features and descriptors on an image.
        // ORB are dispersed on the image using an octree.
        // Mask is ignored in the current implementation.
        virtual int operator()( cv::InputArray _image, cv::InputArray _mask,
                        std::vector<cv::KeyPoint>& _keypoints,
                        cv::OutputArray _descriptors, std::vector<int> &vLappingArea) = 0;

        int inline GetLevels(){ return nlevels; }

        float inline GetScaleFactor() { return scaleFactor; }

        std::vector<float> inline GetScaleFactors() { return mvScaleFactor; }

        std::vector<float> inline GetInverseScaleFactors() { return mvInvScaleFactor; }

        std::vector<float> inline GetScaleSigmaSquares() { return mvLevelSigma2; }

        std::vector<float> inline GetInverseScaleSigmaSquares() { return mvInvLevelSigma2; }

        std::vector<cv::Mat> mvImagePyramid;

    protected:

        void ComputePyramid(cv::Mat image);
        void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);    
        std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                            const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

        void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
        std::vector<cv::Point> pattern;

        int nfeatures;
        double scaleFactor;
        int nlevels;
        int iniThFAST;
        int minThFAST;

        std::vector<int> mnFeaturesPerLevel;

        std::vector<int> umax;

        std::vector<float> mvScaleFactor;
        std::vector<float> mvInvScaleFactor;    
        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSigma2;
    };

}