/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>
#include "FeatureExtractor.h"


namespace mgnc::slam
{
    class ExtractorNode
    {
    public:

        ExtractorNode() : bNoMore(false) { }

        void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

        std::vector<cv::KeyPoint> vKeys;
        cv::Point2i UL, UR, BL, BR;
        std::list<ExtractorNode>::iterator lit;
        bool bNoMore;
    };

    class OrbExtractor : public FeatureExtractor
    {
    public:
        
        enum { HARRIS_SCORE = 0, FAST_SCORE = 1 };

        OrbExtractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);

        // Compute the ORB features and descriptors on an image.
        // ORB are dispersed on the image using an octree.
        // Mask is ignored in the current implementation.
        int operator()(cv::InputArray _image, cv::InputArray _mask,
                       std::vector<cv::KeyPoint>& _keypoints,
                       cv::OutputArray _descriptors, std::vector<int> &vLappingArea);

    protected:

        void ComputePyramid(cv::Mat image);
        void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);    
        std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                            const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

        void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
        std::vector<cv::Point> pattern;
    };
} //namespace mgnc::slam
