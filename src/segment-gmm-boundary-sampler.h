// baud/segment-gmm-boundary-sampler.h

// Author: David Harwath

#ifndef KALDI_BAUD_SEGMENT_GMM_BOUNDARY_SAMPLER_H
#define KALDI_BAUD_SEGMENT_GMM_BOUNDARY_SAMPLER_H

#include <iostream>
#include <vector>

#include "base/kaldi-common.h"
#include "baud/segment-feature-extractor.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "landmarks/landmark-utils.h"
#include "matrix/kaldi-matrix.h"
#include "tree/context-dep.h"
#include "util/common-utils.h"

namespace kaldi {

struct SegmentGmmBoundarySamplerConfig {
  BaseFloat cluster_gamma;
  BaseFloat major_boundary_alpha;
  BaseFloat minor_boundary_alpha;
  int32 max_segment_length;
  std::string posteriors_wspecifier;
  SegmentGmmBoundarySamplerConfig(BaseFloat cluster_gamma = 1,
                                	BaseFloat major_boundary_alpha = 0.8,
                                  BaseFloat minor_boundary_alpha = 0.5,
                                  int32 max_segment_length = 20,
                                	std::string posteriors_wspecifier = "none"):
      cluster_gamma(cluster_gamma), major_boundary_alpha(major_boundary_alpha), minor_boundary_alpha(minor_boundary_alpha),
      max_segment_length(max_segment_length), posteriors_wspecifier(posteriors_wspecifier) {}
  
  void Register (OptionsItf *po) {
    po->Register("cluster-gamma", &cluster_gamma,
                 "Concentration parameter for Dirichlet Process generating clusters");
    po->Register("posteriors-wspecifier", &posteriors_wspecifier,
    						"If specified, write a table of matrices containing the posterior probabilities of "
    						"each cluster for each speech segment");
    po->Register("major-boundary-alpha", &major_boundary_alpha,
                 "Prior probability of major landmarks being turned on as boundaries");
    po->Register("minor-boundary-alpha", &minor_boundary_alpha,
                  "Prior probability of minor landmarks being turned on as boundaries");
    po->Register("max-segment-length", &max_segment_length,
                  "Maximum number of frames allowed per segment");
  }
};

class SegmentGmmBoundarySampler {
public:
  SegmentGmmBoundarySampler(const SegmentGmmBoundarySamplerConfig &config):
    config_(config), extractor_() {
    if (config_.posteriors_wspecifier != "none") {
      posteriors_writer_.Open(config_.posteriors_wspecifier); 
    }
  }

  ~SegmentGmmBoundarySampler() {
    if (posteriors_writer_.IsOpen()) {
      posteriors_writer_.Close();
    }
  }
                       
  void SetOptions(const SegmentGmmBoundarySamplerConfig &config) {
    config_ = config;
    if (posteriors_writer_.IsOpen()) {
      posteriors_writer_.Close();
    }
    if (config_.posteriors_wspecifier != "none") {
      posteriors_writer_.Open(config_.posteriors_wspecifier);
    }
  }

  SegmentGmmBoundarySamplerConfig GetOptions() {
    return config_;
  }

  // Computes a resampling of the boundary variables and segment cluster assignments
  // for the given utterance. Warning: modifies the class_counts vector to reflect
  // the updated counts after the resampling.
  bool ResampleAlignment(const Matrix<BaseFloat> &features,
                         const std::vector<int32> &old_bounds,
                         const LandmarkSeq &landmarks,
                         const AmDiagGmm &am_gmm,
                         const std::string &utt_id,
                         const std::vector<int32> class_counts,
                         std::vector<int32> *new_class_counts,
                         Int32VectorWriter *alignment_writer,
                         double *like);
  
  void SampleBounds(const Matrix<BaseFloat> &features,
                    const std::vector<int32> &old_bounds, 
                    const LandmarkSeq &landmarks,
                    const AmDiagGmm &am_gmm,
                    const std::vector<int32> &class_counts,
                    std::vector<int32> *new_bounds);
  
  void SampleClusters(const Matrix<BaseFloat> &features,
                      const std::string &utt_id,
                      const std::vector<int32> &new_bounds, 
                      const AmDiagGmm &am_gmm,
                      const std::vector<int32> &class_counts,
                      std::vector<int32> *new_clusters);

  void SampleStateSequence(const Matrix<BaseFloat> &features,
                           const std::vector<int32> &new_bounds,
                           const std::vector<int32> &new_clusters,
                           std::vector<int32> *state_sequence);
  
  void GetBoundsFromLandmarks(const Matrix<BaseFloat> features,
                              const LandmarkSeq &landmarks,
                              const bool major_only,
                              std::vector<int32> *old_bounds);

  void GetBoundsFromAlignment(const std::vector<int32> &alignment,
                              std::vector<int32> *old_bounds,
                              std::vector<int32> *old_clusters);

  BaseFloat ComputeSegmentLikelihood(const SubMatrix<BaseFloat> &features,
                                     const AmDiagGmm &am_gmm,
                                     const int32 pdf_idx);

  BaseFloat ComputeSegmentMarginalLikelihood(const SubMatrix<BaseFloat> &features,
                                             const AmDiagGmm &am_gmm,
                                             const std::vector<int32> &class_counts,
                                             const int32 N);

  void WriteNewAlignment(const std::vector<int32> &state_sequence,
                         const std::string &utt_id,
                         Int32VectorWriter *alignment_writer);
  
  private:

    SegmentGmmBoundarySamplerConfig config_;
    SegmentFeatureExtractor extractor_;
    BaseFloatMatrixWriter posteriors_writer_;

};

}  // end namespace kaldi

#endif