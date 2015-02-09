// baud/hmm-gmm-boundary-sampler.h

// Author: David Harwath

#include <random>
#include <vector>

#include "base/kaldi-common.h"
#include "baud/segment-feature-extractor.h"
#include "baud/segment-gmm-boundary-sampler.h"
#include "gmm/am-diag-gmm.h"
#include "landmarks/landmark-utils.h"
#include "matrix/kaldi-matrix.h"
#include "tree/context-dep.h"
#include "util/common-utils.h"

namespace kaldi {

const BaseFloat kLogZero = -100000;

BaseFloat LogSumExpVector(const std::vector<BaseFloat> &vec) {
	BaseFloat maxval = kLogZero;
	for (int32 i = 0; i < vec.size(); ++i) {
		const BaseFloat val = vec[i];
		if (val > maxval) {
			maxval = val;
		}
	}
	BaseFloat sum = 0.0;
	for (int32 i = 0; i < vec.size(); ++i) {
		sum += std::exp(vec[i] - maxval);
	}
	return std::log(sum) + maxval;
}

void MergeAdjacentDuplicates(const std::vector<int32> &bounds,
														 const std::vector<int32> &clusters,
														 std::vector<int32> *merged_bounds,
														 std::vector<int32> *merged_clusters) {
	KALDI_ASSERT(bounds.size() > 1);
	KALDI_ASSERT(bounds.size() == clusters.size() + 1);
	KALDI_ASSERT(merged_clusters != NULL);
	KALDI_ASSERT(merged_bounds != NULL);
	merged_bounds->clear();
	merged_clusters->clear();
	merged_bounds->push_back(bounds[0]);
	for (int32 i = 0; i < clusters.size(); ++i) {
		if (i == clusters.size() - 1) {
			merged_bounds->push_back(bounds[i + 1]);
			merged_clusters->push_back(clusters[i]);
		} else {
			if (clusters[i] == clusters[i + 1]) {
				continue;
			} else {
				merged_bounds->push_back(bounds[i + 1]);
				merged_clusters->push_back(clusters[i]);
			}
		}
	}
}

bool SegmentGmmBoundarySampler::ResampleAlignment(const Matrix<BaseFloat> &features,
												                          const std::vector<int32> &old_bounds,
												                          const LandmarkSeq &landmarks,
												                          const AmDiagGmm &am_gmm,
												                          const std::string &utt_id,
												                          const std::vector<int32> class_counts,
												                          std::vector<int32> *new_class_counts,
												                          Int32VectorWriter *alignment_writer,
												                          double *like) {
	// Need to come up with a better assertation for feature size matching AM size.
	//KALDI_ASSERT(features.NumCols() == am_gmm.Dim());
	KALDI_ASSERT(new_class_counts != NULL && alignment_writer != NULL);
	KALDI_ASSERT(new_class_counts->size() == class_counts.size());
	// Figure out where the current boundaries are using the alignment, as well as the current cluster ids
	std::vector<int32> new_bounds;
	SampleBounds(features, old_bounds, landmarks, am_gmm, class_counts, &new_bounds);
	std::vector<int32> new_clusters;
	SampleClusters(features, utt_id, new_bounds, am_gmm, class_counts, &new_clusters);
	// Merge any adjacent segments that were assigned to the same cluster
	std::vector<int32> merged_bounds;
	std::vector<int32> merged_clusters;
	MergeAdjacentDuplicates(new_bounds, new_clusters, &merged_bounds, &merged_clusters);
	std::vector<int32> state_sequence;
	SampleStateSequence(features, merged_bounds, merged_clusters, &state_sequence);
	WriteNewAlignment(state_sequence, utt_id, alignment_writer);
	// Update the class counts using old_clusters and new_clusters
	for (int32 i = 0; i < merged_clusters.size(); ++i) {
		(*new_class_counts)[merged_clusters[i]]++;
	}
	return true;
}
  
void SegmentGmmBoundarySampler::SampleBounds(const Matrix<BaseFloat> &features,
												                     const std::vector<int32> &old_bounds, 
												                     const LandmarkSeq &landmarks,
												                     const AmDiagGmm &am_gmm,
												                     const std::vector<int32> &class_counts,
												                     std::vector<int32> *new_bounds) {
	KALDI_ASSERT(new_bounds != NULL);
	new_bounds->clear();
	const int32 num_landmarks = landmarks.landmarks.size();
	// The random number generator we will use to sample a value for each boundary
	std::random_device rd;
	std::default_random_engine generator(rd());
	// The -1th frame is automatically a boundary
	new_bounds->push_back(-1);
	// precompute the class count sum, N
	int32 N = 0;
	for (int32 i = 0; i < class_counts.size(); ++i) {
		N += class_counts[i];
	}
	for (int32 l = 0; l < num_landmarks; ++l) {
		const int32 boundary_frame = landmarks.landmarks[l].first;
		const landmark_type_t landmark_type = landmarks.landmarks[l].second;
		// Only consider landmarks that aren't at the boundary utterance
		if (boundary_frame >= 0 && boundary_frame < features.NumRows() - 1) {
			// These variables will keep track of the left and right closest boundaries to the
			// boundary we are currently examining
			int32 left_bound_idx = 0;
			int32 right_bound_idx = old_bounds.size() - 1;
			// Grab the nearest currently 'on' boundary variable to the left of this position,
			// as well as the nearest one to the right 
			while (left_bound_idx < new_bounds->size() &&
						(*new_bounds)[left_bound_idx] < boundary_frame) { left_bound_idx++; }
			left_bound_idx--;
			while (old_bounds[right_bound_idx] > boundary_frame) { right_bound_idx--; }
			if (right_bound_idx < old_bounds.size() - 1) {
				right_bound_idx++;
			}
			// The segment begins at the frame just after the last boundary
			const int32 start_frame = std::max(0, (*new_bounds)[left_bound_idx] + 1);
			// The segment ends at the next boundary
			const int32 end_frame = std::min(old_bounds[right_bound_idx], features.NumRows() - 1);
			// Now, figure out if this boundary was previously turned 'on' so we can adjust N accordingly
			int32 N_minus = 0;
			if (std::find(old_bounds.begin(), old_bounds.end(), boundary_frame) != old_bounds.end()) {
				N_minus = N - 2;
			} else {
				N_minus = N - 1;
			}
			// There are two possibilities: boundary_frame is off or it is on.
			// If it is off, then this segment spans all the frames from start_frame to end_frame
			// (let's call this 'whole_segment')
			// If it is on, then we have two adjacent segments: [start_frame, boundary_frame] and
			// (boundary_frame, end_frame]
			// (let's call these 'left_segment' and 'right_segment')
			// In order to sample a value for this boundary, we have to compute the relative probabilities
			// of each of these segments.
			const int32 num_cols = features.NumCols();
			KALDI_ASSERT(start_frame < features.NumRows());
			KALDI_ASSERT(0 < features.NumCols());
			KALDI_ASSERT(boundary_frame - start_frame + 1 <= features.NumRows() - start_frame);
			KALDI_ASSERT(num_cols <= features.NumCols());
			const SubMatrix<BaseFloat> left_segment_features =
				SubMatrix<BaseFloat>(features, start_frame, boundary_frame - start_frame + 1, 0, num_cols);
			KALDI_ASSERT(boundary_frame + 1 < features.NumRows());
			KALDI_ASSERT(0 < features.NumCols());
			if (end_frame - boundary_frame > features.NumRows() - (boundary_frame + 1)) {
				KALDI_LOG << "end_frame = " << end_frame;
				KALDI_LOG << "boundary_frame = " << boundary_frame;
				KALDI_LOG << "end_frame - boundary_frame = " << end_frame - boundary_frame;
				KALDI_LOG << "features.NumRows() = " << features.NumRows();
				KALDI_LOG << "features.NumRows() - (boundary_frame + 1) = " << features.NumRows() - (boundary_frame + 1);
				KALDI_LOG << "right_bound_idx = " << right_bound_idx;
				KALDI_LOG << "old_bounds[right_bound_idx] = " << old_bounds[right_bound_idx];
				KALDI_LOG << "old_bounds.size() = " << old_bounds.size();
				KALDI_LOG << "num_landmarks = " << num_landmarks;
			}
			KALDI_ASSERT(boundary_frame + 1 < features.NumRows());
			KALDI_ASSERT(end_frame - boundary_frame <= features.NumRows() - (boundary_frame + 1));
			KALDI_ASSERT(num_cols <= features.NumCols());
			const SubMatrix<BaseFloat> right_segment_features =
				SubMatrix<BaseFloat>(features, boundary_frame + 1, end_frame - boundary_frame, 0, num_cols);
			KALDI_ASSERT(start_frame < features.NumRows());
			KALDI_ASSERT(0 < features.NumCols());
			if (end_frame - start_frame + 1 > features.NumRows() - start_frame) {
				KALDI_LOG << "end_frame = " << end_frame;
				KALDI_LOG << "start_frame = " << start_frame;
				KALDI_LOG << "end_frame - start_frame + 1 = " << end_frame - start_frame + 1;
				KALDI_LOG << "features.NumRows() = " << features.NumRows();
				KALDI_LOG << "features.NumRows() - start_frame = " << features.NumRows() - start_frame;
			}
			KALDI_ASSERT(end_frame - start_frame + 1 <= features.NumRows() - start_frame);
			KALDI_ASSERT(num_cols <= features.NumCols());
			const SubMatrix<BaseFloat> whole_segment_features =
				SubMatrix<BaseFloat>(features, start_frame, end_frame - start_frame + 1, 0, num_cols);
			const BaseFloat left_segment_likelihood = 
				ComputeSegmentMarginalLikelihood(left_segment_features, am_gmm, class_counts, N_minus);
			const BaseFloat right_segment_likelihood = 
				ComputeSegmentMarginalLikelihood(right_segment_features, am_gmm, class_counts, N_minus + 1);
			const BaseFloat whole_segment_likelihood = 
				ComputeSegmentMarginalLikelihood(whole_segment_features, am_gmm, class_counts, N_minus);
			// Now we can compute the normalized posterior for this boundary variable and sample from it.
			std::vector<BaseFloat> log_posteriors;
			log_posteriors.push_back(whole_segment_likelihood);
			log_posteriors.push_back((left_segment_likelihood + right_segment_likelihood)/2);
			//KALDI_LOG << "boundary off likelihood = " << log_posteriors[0];
			//KALDI_LOG << "boundary on likelihood = " << log_posteriors[1];
			const BaseFloat normalizer = LogSumExpVector(log_posteriors);
			// Set the prior based upon the landmark type
			BaseFloat boundary_prior_1;
			if (landmark_type == MAJOR_LANDMARK) {
				boundary_prior_1 = config_.major_boundary_alpha;
			} else {
				boundary_prior_1 = config_.minor_boundary_alpha;
			}
			BaseFloat boundary_prior_0 = 1.0 - boundary_prior_1;
			BaseFloat boundary_posterior_0 = std::exp(log_posteriors[0] - normalizer) * boundary_prior_0;
			BaseFloat boundary_posterior_1 = std::exp(log_posteriors[1] - normalizer) * boundary_prior_1;
			std::vector<BaseFloat> params;
			params.push_back(boundary_posterior_0);
			params.push_back(boundary_posterior_1);
			std::discrete_distribution<int32> dist(params.begin(), params.end());
			const int32 boundary_sample = dist(generator);
			if (boundary_sample || boundary_prior_1 >= 1.0 ||
				end_frame - start_frame + 1 > config_.max_segment_length) {
				new_bounds->push_back(boundary_frame);
			}
		}
	}
	// The very last frame is automatically a boundary
	new_bounds->push_back(features.NumRows() - 1);
}
  
  void SegmentGmmBoundarySampler::SampleClusters(const Matrix<BaseFloat> &features,
													                       const std::string &utt_id,
													                       const std::vector<int32> &new_bounds, 
													                       const AmDiagGmm &am_gmm,
													                       const std::vector<int32> &class_counts,
													                       std::vector<int32> *new_clusters) {
	const int32 num_phones = am_gmm.NumPdfs();
	const int32 num_segs = new_bounds.size() - 1;
	KALDI_ASSERT(new_bounds.size() >= 2 &&
							 new_bounds[0] == -1 &&
							 new_bounds.back() == features.NumRows() - 1);
	KALDI_ASSERT(num_phones == class_counts.size());
	KALDI_ASSERT(new_clusters != NULL);
	new_clusters->clear();
	new_clusters->resize(num_segs);
	int32 N = 0;
	for (int32 c = 0; c < class_counts.size(); ++c) {
		N += class_counts[c];
	}
	// The random number generator we will use to do sampling
	std::random_device rd;
	std::default_random_engine generator(rd());
	// If we want to write out posteriors, size the output matrix
	Matrix<BaseFloat> post_mat; 
	if (posteriors_writer_.IsOpen()) {
		post_mat.Resize(features.NumRows(), class_counts.size());
	}
	// Iterate over each segment and compute likelihoods of all phones
	for (int32 s = 0; s < num_segs; ++s) {
		std::vector<BaseFloat> posteriors;
		// allocate space for all the phones
		posteriors.resize(num_phones);
		// Grab the submatrix of features corresponding to this segment
		const SubMatrix<BaseFloat> segment_features = SubMatrix<BaseFloat>(features, new_bounds[s] + 1,
			new_bounds[s + 1] - new_bounds[s], 0, features.NumCols());
		// Approximate the 'new table' likelihood integral with the average like
		std::vector<BaseFloat> loglikes;
		for (int32 p = 0; p < num_phones; ++p) {
			// only compute the likelihood for classes which are currently occupied. 
			// For empty clusters, assign kLogZero. They will get filled in gradually using
			// the new table likelihoods.
			BaseFloat loglike = kLogZero;
			if (class_counts[p] > 0) {
				loglike = ComputeSegmentLikelihood(segment_features, am_gmm, p);
				BaseFloat prior = 1.0 * class_counts[p] / (N - 1 + config_.cluster_gamma);
				posteriors[p] = loglike + std::log(prior);
				loglikes.push_back(loglike);
			} else {
				posteriors[p] = loglike;
			}
		}

		// Assign the new table posterior to the first empty phone in the list
		for (int32 i = 0; i < class_counts.size(); ++i) {
			if (class_counts[i] == 0) {
				BaseFloat avg_loglike = LogSumExpVector(loglikes) - std::log(loglikes.size());
				BaseFloat new_table_prior = config_.cluster_gamma / (N - 1 + config_.cluster_gamma);
				BaseFloat new_table_posterior = avg_loglike + std::log(new_table_prior);
				posteriors[i] = new_table_posterior;
				break;
			}
		}

		// Now define the categorical distribution to be used to sample a new cluster
		std::vector<BaseFloat> unlogged_posteriors;
		const BaseFloat normalizer = LogSumExpVector(posteriors);
		for (int32 i = 0; i < posteriors.size(); ++i) {
			unlogged_posteriors.push_back(std::exp(posteriors[i] - normalizer));
		}
		std::discrete_distribution<int32> dist(unlogged_posteriors.begin(), unlogged_posteriors.end());
		const int32 sampled_class = dist(generator);
		(*new_clusters)[s] = sampled_class;

		// If we're going to write out the posteriors, copy this current row into the output matrix
		if (posteriors_writer_.IsOpen()) {
			for (int32 post_row = new_bounds[s] + 1; post_row <= new_bounds[s + 1]; ++post_row) {
				BaseFloat *row_data = post_mat.RowData(post_row);
				for (int32 post_idx = 0; post_idx < unlogged_posteriors.size(); ++post_idx) {
					row_data[post_idx] = unlogged_posteriors[post_idx];
				}
			}
		}

	}
	// If we're going to write out the posteriors, do it now
	if (posteriors_writer_.IsOpen()) {
		posteriors_writer_.Write(utt_id, post_mat);
	}
}

// This method is really trivial. We don't even "sample" anything, we just write out
// the pdf index of the cluster each frame is assigned to.
void SegmentGmmBoundarySampler::SampleStateSequence(const Matrix<BaseFloat> &features,
												                            const std::vector<int32> &new_bounds,
												                            const std::vector<int32> &new_clusters,
												                            std::vector<int32> *state_sequence) {
	KALDI_ASSERT(state_sequence != NULL);
	state_sequence->clear();
	state_sequence->reserve(features.NumRows());
	int32 current_seg = 0;
	for (int32 i = 0; i < features.NumRows(); ++i) {
		if (i > new_bounds[current_seg + 1]) {
			current_seg++;
		}
		state_sequence->push_back(new_clusters[current_seg]);
	}
}

void SegmentGmmBoundarySampler::GetBoundsFromLandmarks(const Matrix<BaseFloat> features,
												                               const LandmarkSeq &landmarks,
												                               const bool major_only,
												                               std::vector<int32> *old_bounds) {
	KALDI_ASSERT(old_bounds != NULL);
	old_bounds->clear();
	for (int32 i = 0; i < landmarks.landmarks.size(); ++i) {
		KALDI_ASSERT(landmarks.landmarks[i].first >= -1);
		KALDI_ASSERT(landmarks.landmarks[i].first < features.NumRows());
		const landmark_type_t type = landmarks.landmarks[i].second;
		if (!major_only || type == MAJOR_LANDMARK) {
			(*old_bounds).push_back(landmarks.landmarks[i].first);
		}
	}
}

void SegmentGmmBoundarySampler::GetBoundsFromAlignment(const std::vector<int32> &alignment,
												                               std::vector<int32> *old_bounds,
												                               std::vector<int32> *old_clusters) {
	KALDI_ASSERT(old_bounds != NULL);
	KALDI_ASSERT(old_clusters != NULL);
	old_bounds->clear();
	old_clusters->clear();
	int32 last_unit_id = -1;
	// This is basically just run-length encoding the alignment into old_bounds and old_clusters
	for (int32 t = 0; t < alignment.size(); ++t) {
		const int32 unit_id = alignment[t];
		if (unit_id != last_unit_id) {
			old_clusters->push_back(unit_id);
			old_bounds->push_back(t - 1);
		}
		last_unit_id = unit_id;
	}
	// The last frame is automatically a boundary
	old_bounds->push_back(alignment.size() - 1);
}

BaseFloat SegmentGmmBoundarySampler::ComputeSegmentLikelihood(const SubMatrix<BaseFloat> &features,
												                                      const AmDiagGmm &am_gmm,
												                                      const int32 pdf_idx) {
	Vector<BaseFloat> segment_features = extractor_.AvgThirdsPlusDuration(features);
	KALDI_ASSERT(segment_features.Dim() == am_gmm.Dim());
	return am_gmm.LogLikelihood(pdf_idx, segment_features);
}

BaseFloat SegmentGmmBoundarySampler::ComputeSegmentMarginalLikelihood(const SubMatrix<BaseFloat> &features,
												                                              const AmDiagGmm &am_gmm,
												                                              const std::vector<int32> &class_counts,
												                                              const int32 N) {
	// The way we compute a segment likelihood is to marginalize over all the phones.
	// We first compute the likelihood of the features given each phone, multiply the likelihood
	// by the corresponding Dirichlet prior, then accumulate and normalize.
	const int32 num_phones = am_gmm.NumPdfs();
	std::vector<BaseFloat> posteriors;
	std::vector<BaseFloat> loglikes;
	for (int32 p = 0; p < num_phones; ++p) {
		if (class_counts[p] > 0) {
			const BaseFloat likelihood = ComputeSegmentLikelihood(features, am_gmm, p);
			loglikes.push_back(likelihood);
			const BaseFloat prior = 1.0 * class_counts[p] / (N + config_.cluster_gamma);
			posteriors.push_back(likelihood + std::log(prior));
		}
	}
	// Approximate the integral over all model parameters with the average likelihood
	const BaseFloat new_table_likelihood = LogSumExpVector(loglikes) - std::log(loglikes.size());
	const BaseFloat new_table_prior = config_.cluster_gamma / (N + config_.cluster_gamma);
	posteriors.push_back(new_table_likelihood + std::log(new_table_prior));
	//KALDI_LOG << "new_table_likelihood = " << new_table_likelihood;
	return LogSumExpVector(posteriors);
}

void SegmentGmmBoundarySampler::WriteNewAlignment(const std::vector<int32> &state_sequence,
												                          const std::string &utt_id,
												                          Int32VectorWriter *alignment_writer) {
	KALDI_ASSERT(alignment_writer != NULL);
	alignment_writer->Write(utt_id, state_sequence);
}

  }  // end namespace kaldi