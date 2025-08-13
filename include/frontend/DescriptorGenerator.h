#pragma once

#include <vector>
#include <opencv2/core/mat.hpp>
// #include <opencv2/core.hpp>

namespace mgnc::slam
{
    class KeypointDetector
    {
    public:

        class KeypointDetector() { }

        virtual ~class KeypointDetector() { }

        // Compute the ORB features and descriptors on an image.
        // ORB are dispersed on the image using an octree.
        // Mask is ignored in the current implementation.
        virtual int operator()() = 0;
    };

}