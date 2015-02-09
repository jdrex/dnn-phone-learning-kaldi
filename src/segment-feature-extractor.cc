// baud/segment-feature-extractor.cc

// Author: David Harwath

#include <iostream>
#include <vector>

#include "base/kaldi-common.h"
#include "baud/segment-feature-extractor.h"
#include "matrix/kaldi-matrix.h"
#include "util/common-utils.h"

namespace kaldi {

	Vector<BaseFloat> AverageFeatures(const SubMatrix<BaseFloat> &features, const int32 start_idx, const int32 end_idx) {
		if (start_idx < 0 && start_idx >= end_idx && start_idx >= features.NumRows()) {
			KALDI_LOG << "start_idx = " << start_idx << " NumRows = " << features.NumRows();
			KALDI_ASSERT(start_idx >= 0 && start_idx < end_idx && start_idx < features.NumRows());
		}
		
		if (end_idx < 0 && end_idx >= features.NumRows()){
			KALDI_LOG << "end_idx = " << end_idx << " NumRows = " << features.NumRows();
			KALDI_ASSERT(end_idx >= 0 && end_idx < features.NumRows());
		}
		Vector<BaseFloat> avg(features.NumCols(), kSetZero);
		for (int32 i = start_idx; i < end_idx; ++i) {
			avg.AddVec(1.0, features.Row(i));
		}
		avg.Scale(1.0 / (end_idx - start_idx));
		return avg;
	}

	Vector<double> AverageFeatures(const SubMatrix<double> &features, const int32 start_idx, const int32 end_idx) {
		if (start_idx < 0 && start_idx >= end_idx && start_idx >= features.NumRows()) {
			KALDI_LOG << "start_idx = " << start_idx << " NumRows = " << features.NumRows();
			KALDI_ASSERT(start_idx >= 0 && start_idx < end_idx && start_idx < features.NumRows());
		}
		
		if (end_idx < 0 && end_idx >= features.NumRows()){
			KALDI_LOG << "end_idx = " << end_idx << " NumRows = " << features.NumRows();
			KALDI_ASSERT(end_idx >= 0 && end_idx < features.NumRows());
		}
		Vector<double> avg(features.NumCols(), kSetZero);
		for (int32 i = start_idx; i < end_idx; ++i) {
			avg.AddVec(1.0, features.Row(i));
		}
		avg.Scale(1.0 / (end_idx - start_idx));
		return avg;
	}

	Vector<BaseFloat> SegmentFeatureExtractor::AvgThirdsPlusDuration(const SubMatrix<BaseFloat> &features) const {
		const int32 num_seg_feats = 3 * features.NumCols() + 1;
		Vector<BaseFloat> seg_feats(num_seg_feats);
		// Compute the beginning, middle, and end thirds of the segment
		const int32 frames_per_chunk = static_cast<int32>(0.5 + (static_cast<BaseFloat>(features.NumRows()) / 3));
		const int32 beg_offset = 0;
		int32 mid_offset = beg_offset + frames_per_chunk;
		int32 end_offset = mid_offset + frames_per_chunk;
		// If there's less than one frame per chunk, we will just make all the chunks the same.
		if (frames_per_chunk == 0) {
			Vector<BaseFloat> avg = AverageFeatures(features, 0, features.NumRows());
			for (int32 i = 0; i < avg.Dim(); ++i) {
				(seg_feats.Data())[i] = avg(i);
				(seg_feats.Data())[i + features.NumCols()] = avg(i);
				(seg_feats.Data())[i + 2 * features.NumCols()] = avg(i);
			}
		} else {
			Vector<BaseFloat> beg_avg = AverageFeatures(features, beg_offset, mid_offset);
			Vector<BaseFloat> mid_avg = AverageFeatures(features, mid_offset, end_offset);
			if (end_offset == features.NumRows()) {
				end_offset--;
				KALDI_ASSERT(end_offset >= 0);
			}
			Vector<BaseFloat> end_avg = AverageFeatures(features, end_offset, features.NumRows());
			for (int32 i = 0; i < beg_avg.Dim(); ++i) {
				(seg_feats.Data())[i] = beg_avg(i);
			}
			for (int32 i = 0; i < mid_avg.Dim(); ++i) {
				const int32 offset = beg_avg.Dim();
				(seg_feats.Data())[i + offset] = mid_avg(i);
			}
			for (int32 i = 0; i < end_avg.Dim(); ++i) {
				const int32 offset = beg_avg.Dim() + mid_avg.Dim();
				(seg_feats.Data())[i + offset] = end_avg(i);
			}
		}
		(seg_feats.Data())[num_seg_feats - 1] = features.NumRows();
		return seg_feats;
	}

	Vector<double> SegmentFeatureExtractor::AvgThirdsPlusDuration(const SubMatrix<double> &features) const {
		const int32 num_seg_feats = 3 * features.NumCols() + 1;
		Vector<double> seg_feats(num_seg_feats);
		// Compute the beginning, middle, and end thirds of the segment
		const int32 frames_per_chunk = static_cast<int32>(0.5 + (static_cast<BaseFloat>(features.NumRows()) / 3));
		const int32 beg_offset = 0;
		int32 mid_offset = beg_offset + frames_per_chunk;
		int32 end_offset = mid_offset + frames_per_chunk;
		// If there's less than one frame per chunk, we will just make all the chunks the same.
		if (frames_per_chunk == 0) {
			Vector<double> avg = AverageFeatures(features, 0, features.NumRows());
			for (int32 i = 0; i < avg.Dim(); ++i) {
				(seg_feats.Data())[i] = avg(i);
				(seg_feats.Data())[i + features.NumCols()] = avg(i);
				(seg_feats.Data())[i + 2 * features.NumCols()] = avg(i);
			}
		} else {
			Vector<double> beg_avg = AverageFeatures(features, beg_offset, mid_offset);
			Vector<double> mid_avg = AverageFeatures(features, mid_offset, end_offset);
			if (end_offset == features.NumRows()) {
				end_offset--;
				KALDI_ASSERT(end_offset >= 0);
			}
			Vector<double> end_avg = AverageFeatures(features, end_offset, features.NumRows());
			for (int32 i = 0; i < beg_avg.Dim(); ++i) {
				(seg_feats.Data())[i] = beg_avg(i);
			}
			for (int32 i = 0; i < mid_avg.Dim(); ++i) {
				const int32 offset = beg_avg.Dim();
				(seg_feats.Data())[i + offset] = mid_avg(i);
			}
			for (int32 i = 0; i < end_avg.Dim(); ++i) {
				const int32 offset = beg_avg.Dim() + mid_avg.Dim();
				(seg_feats.Data())[i + offset] = end_avg(i);
			}
		}
		(seg_feats.Data())[num_seg_feats - 1] = features.NumRows();
		return seg_feats;
	}

}  // end namespace kaldi