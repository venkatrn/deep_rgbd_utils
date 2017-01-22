/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

// This is a modification of PCL's RANSAC implementation for obtaining
// the top-k models, as opposed to the single best one.

#ifndef PCL_SAMPLE_CONSENSUS_MULTIPLE_RANSAC_H_
#define PCL_SAMPLE_CONSENSUS_MULTIPLE_RANSAC_H_

#include <pcl/sample_consensus/sac.h>
#include <pcl/sample_consensus/sac_model.h>

#include <deep_rgbd_utils/bounded_priority_queue.h>

namespace pcl
{

  class RANSACMatch {
    public:
      std::vector<int> samples;
      Eigen::VectorXf model_coefficients;
      int num_inliers = 0;
      RANSACMatch() {}
      RANSACMatch(const std::vector<int>& selection, const Eigen::VectorXf& coefficients, int n_inliers) {
        samples = selection;
        model_coefficients = coefficients;
        num_inliers = n_inliers;
      }
      bool operator <(const RANSACMatch& other) const {
        return num_inliers < other.num_inliers;
      }
      bool operator >(const RANSACMatch& other) const {
        return !(*this < other);
      }
  };

  template <typename PointT, std::size_t num_matches>
  class RandomSampleConsensusMultiple : public SampleConsensus<PointT>
  {
    typedef typename SampleConsensusModel<PointT>::Ptr SampleConsensusModelPtr;

    public:
      typedef boost::shared_ptr<RandomSampleConsensusMultiple> Ptr;
      typedef boost::shared_ptr<const RandomSampleConsensusMultiple> ConstPtr;

      using SampleConsensus<PointT>::max_iterations_;
      using SampleConsensus<PointT>::threshold_;
      using SampleConsensus<PointT>::iterations_;
      using SampleConsensus<PointT>::sac_model_;
      using SampleConsensus<PointT>::model_;
      using SampleConsensus<PointT>::model_coefficients_;
      using SampleConsensus<PointT>::inliers_;
      using SampleConsensus<PointT>::probability_;

      /** \brief RANSAC (RAndom SAmple Consensus) main constructor
        * \param[in] model a Sample Consensus model
        */
      RandomSampleConsensusMultiple (const SampleConsensusModelPtr &model) 
        : SampleConsensus<PointT> (model)
      {
        // Maximum number of trials before we give up.
        max_iterations_ = 10000;
      }

      /** \brief RANSAC (RAndom SAmple Consensus) main constructor
        * \param[in] model a Sample Consensus model
        * \param[in] threshold distance to model threshold
        */
      RandomSampleConsensusMultiple (const SampleConsensusModelPtr &model, double threshold) 
        : SampleConsensus<PointT> (model, threshold)
      {
        // Maximum number of trials before we give up.
        max_iterations_ = 10000;
      }

      /** \brief Compute the actual model and find the inliers
        * \param[in] debug_verbosity_level enable/disable on-screen debug information and set the verbosity level
        */
      bool 
      computeModel (int debug_verbosity_level = 0);

      // Obtain the top matches, in sorted order (best to worst).
      std::vector<RANSACMatch> getTopMatches();


    private:
      // Number of top models to save.
      bounded_priority_queue<RANSACMatch, num_matches> best_matches_;

  };

}

#include <deep_rgbd_utils/ransac.hpp>

#endif  //#ifndef PCL_SAMPLE_CONSENSUS_MULTIPLE_RANSAC_H_

