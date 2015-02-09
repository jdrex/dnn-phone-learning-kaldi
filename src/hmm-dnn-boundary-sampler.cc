// baud/hmm-dnn-boundary-sampler.h

// Author: David Harwath

#include <random>
#include <vector>

#include "base/kaldi-common.h"
#include "baud/hmm-dnn-boundary-sampler.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-topology.h"
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

// Probably won't handle nonemitting states well, unless they're the last state.
void GetHmmTransitionMatrixAndPdfIds(const TransitionModel &trans_model,
																		 const ContextDependency &ctx_dep, 
																		 const int32 phone_index, 
																		 std::vector<std::vector<BaseFloat> > *A,
																		 std::vector<int32> *pdf_ids) {
	KALDI_ASSERT(A != NULL && pdf_ids != NULL);
	A->clear();
	pdf_ids->clear();
	// Grab the HMM topology so we can figure out how many states this phone has
	const HmmTopology &topo = trans_model.GetTopo();
	const HmmTopology::TopologyEntry &states = topo.TopologyForPhone(phone_index);
	int32 num_emitting_states = 0;
	for (int32 i = 0; i < states.size(); ++i) {
		if (states[i].pdf_class != kNoPdf) {
			num_emitting_states++;
		}
	}
	// For now, protect our bad assumptions with assertations
	KALDI_ASSERT(num_emitting_states == states.size() - 1);
	KALDI_ASSERT(states.back().pdf_class == kNoPdf);

	const int32 num_states = num_emitting_states;
	A->resize(num_states, std::vector<BaseFloat>(num_states, kLogZero));
	int32 final_emitting_state = num_states - 1;
	std::vector<int32> phoneseq;
	phoneseq.push_back(phone_index);
	for (int32 i = 0; i < num_states; ++i) {
		const int32 pdf_class_i = states[i].pdf_class;
		int32 pdf_id;
		if (!ctx_dep.Compute(phoneseq, pdf_class_i, &pdf_id)) {
			KALDI_ERR << "Error when computing pdf_id for phone " << phone_index << " and pdf_class " << pdf_class_i;
		}
		pdf_ids->push_back(pdf_id);
		const int32 transition_state = trans_model.TripleToTransitionState(phone_index, i, pdf_id);
		const std::vector<std::pair<int32, BaseFloat> > &transitions = states[i].transitions;
		for (int32 transition_index = 0; transition_index < transitions.size(); ++transition_index) {
			const int32 transition_id = trans_model.PairToTransitionId(transition_state, transition_index);
			if (trans_model.IsFinal(transition_id)) {
				final_emitting_state = i;
			}
			const BaseFloat logprob = trans_model.GetTransitionLogProb(transition_id);
			const int32 j = transitions[transition_index].first;
			if (j >= 0 && j < num_states) {
				(*A)[i][j] = logprob;
			} 
		}
	}
	// One last assertation to protect our bad assumptions
	KALDI_ASSERT(final_emitting_state == num_emitting_states - 1);
}

// Returns log likelihoods. Assumes the first state produces the first frame.
void ComputeHmmForwardVariables(const SubMatrix<BaseFloat> &features,
																const AmDiagGmm &am_gmm,
																const std::vector<std::vector<BaseFloat> > &A,
																const std::vector<int32> &pdf_ids,
																std::vector<std::vector<BaseFloat> > *alpha) {
	KALDI_ASSERT(alpha != NULL);
	KALDI_ASSERT(pdf_ids.size() > 0);
	KALDI_ASSERT(features.NumRows() > 0);
	const int32 num_frames = features.NumRows();
	const int32 num_states = pdf_ids.size();
	alpha->clear();
	alpha->resize(num_frames, std::vector<BaseFloat>(num_states, kLogZero));
	// Initialize with the first state and the first frame
	(*alpha)[0][0] = am_gmm.LogLikelihood(pdf_ids[0], features.Row(0));
	//KALDI_LOG << "t = 0";
	//for (int32 i = 0; i < num_states; ++i) {
	//	KALDI_LOG << (*alpha)[0][i];
	//}
	for (int32 t = 1; t < num_frames; ++t) {
		for (int32 j = 0; j < num_states; ++j) {
			const BaseFloat likelihood = am_gmm.LogLikelihood(pdf_ids[j], features.Row(t));
			std::vector<BaseFloat> prev_alphas;
			prev_alphas.resize(num_states);
			for (int32 i = 0; i < num_states; ++i) {
				prev_alphas[i] = (*alpha)[t - 1][i] + A[i][j];
			}
			const BaseFloat alpha_sum = LogSumExpVector(prev_alphas);
			(*alpha)[t][j] = likelihood + alpha_sum;
		}
		//KALDI_LOG << "t = " << t;
		//for (int32 i = 0; i < num_states; ++i) {
		//	KALDI_LOG << (*alpha)[t][i];
		//}
	}
}

// Returns log likelihoods. Constrains the sequence to end in the last emitting state.
void ComputeHmmBackwardVariables(const SubMatrix<BaseFloat> &features,
																 const AmDiagGmm &am_gmm,
																 const std::vector<std::vector<BaseFloat> > &A,
																 const std::vector<int32> &pdf_ids,
																 std::vector<std::vector<BaseFloat> > *beta) {
	KALDI_ASSERT(beta != NULL);
	KALDI_ASSERT(pdf_ids.size() > 0);
	KALDI_ASSERT(features.NumRows() > 0);
	const int32 num_frames = features.NumRows();
	const int32 num_states = pdf_ids.size();
	beta->clear();
	beta->resize(num_frames, std::vector<BaseFloat>(num_states, kLogZero));
	(*beta)[num_frames - 1][num_states - 1] = 0.0;
	for (int32 t = num_frames - 2; t >= 0; --t) {
		// Precompute b_j(O_{t+1})
		std::vector<BaseFloat> b(num_states);
		for (int32 j = 0; j < num_states; ++j) {
			b[j] = am_gmm.LogLikelihood(pdf_ids[j], features.Row(t + 1));
		}
		for (int32 i = 0; i < num_states; ++i) {
			std::vector<BaseFloat> vals(num_states);
			for (int32 j = 0; j < num_states; ++j) {
				vals[j] = b[j] + A[i][j] + (*beta)[t + 1][j];
			}
			(*beta)[t][i] = LogSumExpVector(vals);
		}
	}
}

// Returns raw probabilities
void ComputeFramePosteriors(const std::vector<std::vector<BaseFloat> > &alpha,
														const std::vector<std::vector<BaseFloat> > &beta,
														std::vector<std::vector<BaseFloat> > *frame_posteriors) {
	KALDI_ASSERT(frame_posteriors != NULL);
	KALDI_ASSERT(alpha.size() == beta.size());
	frame_posteriors->clear();
	const int32 num_frames = alpha.size();
	KALDI_ASSERT(num_frames > 0);
	const int32 num_states = alpha[0].size();
	frame_posteriors->resize(num_frames, std::vector<BaseFloat>(num_states, 0.0));
	for (int32 t = 0; t < num_frames; ++t) {
		KALDI_ASSERT(alpha[t].size() == num_states && beta[t].size() == num_states);
		for (int32 s = 0; s < num_states; ++s) {
			(*frame_posteriors)[t][s] = alpha[t][s] + beta[t][s];
		}
		const BaseFloat normalizer = LogSumExpVector((*frame_posteriors)[t]);
		for (int32 s = 0; s < num_states; ++s) {
			(*frame_posteriors)[t][s] = std::exp((*frame_posteriors)[t][s] - normalizer);
		}
	}
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

bool HmmDnnBoundarySampler::ResampleAlignment(const Matrix<BaseFloat> &features,
																						  const std::vector<int32> &old_bounds,
																						  const LandmarkSeq &landmarks,
																						  const AmDiagGmm &am_gmm,
												 											const TransitionModel &trans_model,
												 											const ContextDependency &ctx_dep,
												 											const std::string &utt_id,
												 											const std::vector<int32> class_counts,
																						  std::vector<int32> *new_class_counts,
											                        Int32VectorWriter *alignment_writer,
											                        double *like) {
	KALDI_ASSERT(features.NumCols() == am_gmm.Dim());
	KALDI_ASSERT(new_class_counts != NULL && alignment_writer != NULL);
	KALDI_ASSERT(new_class_counts->size() == class_counts.size());
	// Figure out where the current boundaries are using the alignment, as well as the current cluster ids
	std::vector<int32> new_bounds;
	SampleBounds(features, old_bounds, landmarks, am_gmm, trans_model, ctx_dep, class_counts, &new_bounds);
	std::vector<int32> new_clusters;
	SampleClusters(features, utt_id, new_bounds, am_gmm, trans_model, ctx_dep, class_counts, &new_clusters);
	// Merge any adjacent segments that were assigned to the same cluster
	std::vector<int32> merged_bounds;
	std::vector<int32> merged_clusters;
	MergeAdjacentDuplicates(new_bounds, new_clusters, &merged_bounds, &merged_clusters);
	std::vector<int32> state_sequence;
	SampleStateSequence(features, merged_bounds, merged_clusters, am_gmm, trans_model, ctx_dep, &state_sequence);
	WriteNewAlignment(state_sequence, utt_id, alignment_writer);
	// Update the class counts using old_clusters and new_clusters
	for (int32 i = 0; i < merged_clusters.size(); ++i) {
		(*new_class_counts)[merged_clusters[i]]++;
	}
	return true;
}

void HmmDnnBoundarySampler::GetBoundsFromLandmarks(const Matrix<BaseFloat> features,
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

void HmmDnnBoundarySampler::GetBoundsFromAlignment(const std::vector<int32> &alignment,
																									const TransitionModel &trans_model,
																									std::vector<int32> *old_bounds,
																									std::vector<int32> *old_clusters) {
	KALDI_ASSERT(old_bounds != NULL);
	KALDI_ASSERT(old_clusters != NULL);
	old_bounds->clear();
	old_clusters->clear();
	// The -1th frame is a boundary (since the 0th frame begins the first segment)
	old_bounds->push_back(-1);
	// Examine each transition id to figure out if it corresponds to a final state. If so,
	// mark the boundary, grab the phone id, and store them both.
	for (int32 t = 0; t < alignment.size(); ++t) {
		const int32 trans_id = alignment[t];
		if (trans_model.IsFinal(trans_id)) {
			const int32 phone = trans_model.TransitionIdToPhone(trans_id);
			old_bounds->push_back(t);
			old_clusters->push_back(phone);
		}
	}
}

void HmmDnnBoundarySampler::SampleBounds(const Matrix<BaseFloat> &features,
																				const std::vector<int32> &old_bounds, 
																				const LandmarkSeq &landmarks,
																				const AmDiagGmm &am_gmm,
																				const TransitionModel &trans_model,
																				const ContextDependency &ctx_dep,
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
			/*
			KALDI_LOG << "left_bound_idx = " << left_bound_idx;
			KALDI_LOG << "(*new_bounds).size() = " << (*new_bounds).size();
			KALDI_LOG << "(*new_bounds)[left_bound_index] = " << (*new_bounds)[left_bound_idx];
			KALDI_LOG << "features.NumRows() = " << features.NumRows() << ", features.NumCols() = "
			<< features.NumCols();
			KALDI_LOG << "start_frame = " << start_frame << ", end_frame = " << end_frame;
			KALDI_LOG << "boundary_frame = " << boundary_frame;
			KALDI_LOG << "r = " << (boundary_frame - start_frame + 1);
			*/
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
			/*
			KALDI_LOG << "ro = " << boundary_frame + 1;
			KALDI_LOG << "r = " << end_frame - boundary_frame;
			KALDI_LOG << "end_frame = " << end_frame;
			KALDI_LOG << "co = " << 0;
			KALDI_LOG << "c = " << num_cols;
			KALDI_LOG << "NumRows = " << features.NumRows();
			KALDI_LOG << "NumCols = " << features.NumCols();
			KALDI_LOG << "right_bound_idx = " << right_bound_idx;
			KALDI_LOG << "old_bounds.size() = " << old_bounds.size();
			*/
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
				ComputeSegmentLikelihood(left_segment_features, am_gmm, trans_model, ctx_dep, class_counts, N_minus);
			const BaseFloat right_segment_likelihood = 
				ComputeSegmentLikelihood(right_segment_features, am_gmm, trans_model, ctx_dep, class_counts, N_minus + 1);
			const BaseFloat whole_segment_likelihood = 
				ComputeSegmentLikelihood(whole_segment_features, am_gmm, trans_model, ctx_dep, class_counts, N_minus);
			// Now we can compute the normalized posterior for this boundary variable and sample from it.
			//KALDI_LOG << "left_segment_likelihood = " << left_segment_likelihood;
			//KALDI_LOG << "right_segment_likelihood = " << right_segment_likelihood;
			//KALDI_LOG << "whole_segment_likelihood = " << whole_segment_likelihood;
			std::vector<BaseFloat> log_posteriors;
			log_posteriors.push_back(whole_segment_likelihood);
			log_posteriors.push_back(left_segment_likelihood + right_segment_likelihood);
			const BaseFloat normalizer = LogSumExpVector(log_posteriors);
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
			//KALDI_LOG << "Boundary posteriors: " << "P(0) = " << boundary_posterior_0 << ", P(1) = " << boundary_posterior_1;
			std::discrete_distribution<int32> dist(params.begin(), params.end());
			const int32 boundary_sample = dist(generator);
			if (boundary_sample) {
				new_bounds->push_back(boundary_frame);
			}
		}
	}
	// The very last frame is automatically a boundary
	new_bounds->push_back(features.NumRows() - 1);
}

void HmmDnnBoundarySampler::SampleClusters(const Matrix<BaseFloat> &features,
																					const std::string &utt_id,
																					const std::vector<int32> &new_bounds, 
																					const AmDiagGmm &am_gmm,
																					const TransitionModel &trans_model,
																					const ContextDependency &ctx_dep,
																					const std::vector<int32> &class_counts,
																					std::vector<int32> *new_clusters) {
	const int32 num_phones = trans_model.NumPhones();
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
				loglike = ComputeHmmDnnLikelihood(segment_features, am_gmm, trans_model, ctx_dep, p);
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

void HmmDnnBoundarySampler::SampleStateSequence(const Matrix<BaseFloat> &features,
																								const std::vector<int32> &new_bounds,
																								const std::vector<int32> &new_clusters,
																								const AmDiagGmm &am_gmm,
																								const TransitionModel &trans_model,
																								const ContextDependency &ctx_dep,
																								std::vector<int32> *state_sequence) {
	// The first frame of the features is constrained to map to the first state of the
	// corresponding phone hmm, and similarly the last frame is constrained to map to the
	// last (non-emitting) state. We just need to sample the frames in-between.
	// Except - I'm not clear on how to do this.
	// Ok, there appears to be a method for Gibbs sampling state sequences, but it's a bit complex
	// For now let's just try doing forward-backward to get posteriors for each frame, then sample each
	// frame assignment from its posterior
	KALDI_ASSERT(state_sequence != NULL);
	KALDI_ASSERT(new_bounds.size() == new_clusters.size() + 1)
	state_sequence->clear();
	state_sequence->reserve(features.NumRows());
	std::random_device rd;
	std::default_random_engine generator(rd());
	for (int32 s = 0; s < new_clusters.size(); ++s) {
		// Grab the SubMatrix corresponding to the frames for this segment
		const int32 ro = new_bounds[s] + 1;
		const int32 r = new_bounds[s + 1] - new_bounds[s]; 
		SubMatrix<BaseFloat> seg_features(features, ro, r, 0, features.NumCols());
		// Grab the phone_index for this segment
		const int32 phone_index = new_clusters[s];
		// Get the transition probability matrix
		std::vector<std::vector<BaseFloat> > A;
		std::vector<int32> pdf_ids;
		// trans_model indexes phones starting at 1
		GetHmmTransitionMatrixAndPdfIds(trans_model, ctx_dep, phone_index + 1, &A, &pdf_ids);
		
		// Method 1: sample states from posteriors
		/*
		// Compute the forward variables
		std::vector<std::vector<BaseFloat> > alpha;
		ComputeHmmForwardVariables(seg_features, am_gmm, A, pdf_ids, &alpha);
		// Compute backward variables
		std::vector<std::vector<BaseFloat> > beta;
		ComputeHmmBackwardVariables(seg_features, am_gmm, A, pdf_ids, &beta);
		// Use forward and backward variables to get frame posteriors
		std::vector<std::vector<BaseFloat> > frame_posteriors;
		ComputeFramePosteriors(alpha, beta, &frame_posteriors);
		// Now sample a state for each frame from the posterior
		*/

		// Method 2: Sample states forward in time using backward variables
		// Compute backward variables, then starting at the first state sample
		// a state for each frame conditioned on the last sample
		// i.e. P(state s at time t | last sampled state was i) = a_ij * beta_t(s)
		std::vector<std::vector<BaseFloat> > beta;
		ComputeHmmBackwardVariables(seg_features, am_gmm, A, pdf_ids, &beta);
		std::vector<int32> hmm_state_seq(beta.size());
		hmm_state_seq[0] = 0;
		const int32 num_states = beta[0].size();
		for (int32 t = 1; t < seg_features.NumRows(); ++t) {
			const int32 last_state = hmm_state_seq[t - 1];
			std::vector<BaseFloat> likelihoods(num_states);
			for (int32 state = 0; state < num_states; ++state) {
				likelihoods[state] = A[last_state][state] + beta[t][state];
			}
			const BaseFloat normalizer = LogSumExpVector(likelihoods);
			std::vector<BaseFloat> posteriors(likelihoods.size());
			for (int32 state = 0; state < num_states; ++state) {
				posteriors[state] = std::exp(likelihoods[state] - normalizer);
			}
			std::discrete_distribution<int32> dist(posteriors.begin(), posteriors.end());
			const int32 sampled_state = dist(generator);
			hmm_state_seq[t] = sampled_state;
		}
		KALDI_ASSERT(hmm_state_seq.size() == seg_features.NumRows());

		const HmmTopology &topo = trans_model.GetTopo();
		// trans_model phone indexing is 1-based
		const HmmTopology::TopologyEntry &topo_entry = topo.TopologyForPhone(phone_index + 1);
		std::vector<int32> phoneseq;
		phoneseq.push_back(phone_index + 1);
		for (int32 i = 1; i < hmm_state_seq.size(); ++i) {
			// To compute the transition state, we need the phone_index, pdf_index, and 
			// index of the previous sampled state relative to the phone's topology entry
			const int32 last_state = hmm_state_seq[i - 1];
			const int32 sampled_state = hmm_state_seq[i];
			const int32 pdf_class = topo_entry[last_state].pdf_class;
			int32 pdf_id;
			if (!ctx_dep.Compute(phoneseq, pdf_class, &pdf_id)) {
				KALDI_ERR << "Error when computing pdf_id for phone " << phone_index + 1 << " and pdf_class " << pdf_class;
			}
			const int32 transition_state = trans_model.TripleToTransitionState(phone_index + 1, last_state, pdf_id);
			const std::vector<std::pair<int32, BaseFloat> > &transitions = topo_entry[last_state].transitions;
			// now we have to search this state's transitions for which one corresponds to
			// the state that we just sampled
			int32 transition_index = -1;
			for (int32 trans = 0; trans < transitions.size(); ++trans) {
				if (transitions[trans].first == sampled_state) {
					transition_index = trans;
					break;
				}
			}
			if (transition_index == -1) {
				KALDI_LOG << "Sampled state sequence is";
				for (int32 state = 0; state < hmm_state_seq.size(); ++state) {
					KALDI_LOG << hmm_state_seq[state];
				}
				KALDI_LOG << "Backwards variables are";
				for (int32 state_i = 0; state_i < beta.size(); ++state_i) {
					for (int32 state_j = 0; state_j < beta[state_i].size(); ++state_j) {
						KALDI_LOG << beta[state_i][state_j];
					}
				}
				KALDI_ERR << "Error: no valid transition index exists for sampled state sequence "
				<< last_state << " to " << sampled_state << " in phone " << phone_index + 1;
			}
			const int32 transition_id = trans_model.PairToTransitionId(transition_state, transition_index);
			state_sequence->push_back(transition_id);
		}
		// Now we have to find the transition_index corresponding to the transition to the
		// final state so we can write the last transition_id. The assumption here is that
		// the very last frame in this segment must be assigned to the last emitting state
		// in the HMM. If this isn't the case, this code will throw an error.
		{
			const int32 last_state = hmm_state_seq.back();
			const int32 pdf_class = topo_entry[last_state].pdf_class;
			int32 pdf_id;
			if (!ctx_dep.Compute(phoneseq, pdf_class, &pdf_id)) {
				KALDI_ERR << "Error when computing pdf_id for phone " << phone_index + 1 << " and pdf_class " << pdf_class;
			}
			const int32 transition_state = trans_model.TripleToTransitionState(phone_index + 1, last_state, pdf_id);
			const std::vector<std::pair<int32, BaseFloat> > &transitions = topo_entry[last_state].transitions;
			int32 transition_index = -1;
			for (int32 trans = 0; trans < transitions.size(); ++trans) {
				const int32 transition_id = trans_model.PairToTransitionId(transition_state, trans);
				if (trans_model.IsFinal(transition_id)) {
					transition_index = trans;
					state_sequence->push_back(transition_id);
					break;
				}
			}
			if (transition_index == -1) {
				KALDI_LOG << "State sequence:";
				for (int32 state = 0; state < hmm_state_seq.size(); ++state) {
					KALDI_LOG << hmm_state_seq[state];
				}
				KALDI_ERR << "Error: No transition to final state found when sampling state sequence."
				" It is likely that the final sampled state is not the last emitting state.";
			}
		}
	}
	KALDI_ASSERT(state_sequence->size() == features.NumRows());
}

// Returns log likelihood
BaseFloat HmmDnnBoundarySampler::ComputeSegmentLikelihood(const SubMatrix<BaseFloat> &features,
																													const AmDiagGmm &am_gmm,
																													const TransitionModel &trans_model,
																													const ContextDependency &ctx_dep,
																													const std::vector<int32> &class_counts,
																													const int32 N) {
	// The way we compute a segment likelihood is to marginalize over all the phones.
	// We first compute the likelihood of the features given each phone, multiply the likelihood
	// by the corresponding Dirichlet prior, then accumulate and normalize.
	const int32 num_phones = trans_model.NumPhones();
	std::vector<BaseFloat> posteriors;
	std::vector<BaseFloat> loglikes;
	for (int32 p = 0; p < num_phones; ++p) {
		if (class_counts[p] > 0) {
			const BaseFloat likelihood = ComputeHmmDnnLikelihood(features, am_gmm, trans_model, ctx_dep, p);
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

// This really needs to be tested. Returns log likelihood.
BaseFloat HmmDnnBoundarySampler::ComputeHmmDnnLikelihood(const SubMatrix<BaseFloat> &features,
																												const AmDiagGmm &am_gmm,
																												const TransitionModel &trans_model,
																												const ContextDependency &ctx_dep,
																												const int32 phone_index) {

	// Get the transition probability matrix
	std::vector<std::vector<BaseFloat> > A;
	std::vector<int32> pdf_ids;
	// The trans model indexes phones starting at 1
	GetHmmTransitionMatrixAndPdfIds(trans_model, ctx_dep, phone_index + 1, &A, &pdf_ids);
	// Compute the forward variables
	std::vector<std::vector<BaseFloat> > alpha;
	ComputeHmmForwardVariables(features, am_gmm, A, pdf_ids, &alpha);
	// Lastly, we grab the alpha for the final state at the last frame to get the log likelihood
	// Warning: assumes that the final emitting state is the last state, i.e. pdf_ids.size() - 1
	return (alpha[features.NumRows() - 1][pdf_ids.size() - 1]) / (config_.likelihood_scale);
}

void HmmDnnBoundarySampler::WriteNewAlignment(const std::vector<int32> &state_sequence,
																							const std::string &utt_id,
																						 	Int32VectorWriter *alignment_writer) {
	KALDI_ASSERT(alignment_writer != NULL);
	alignment_writer->Write(utt_id, state_sequence);
}

}  // end namespace kaldi
