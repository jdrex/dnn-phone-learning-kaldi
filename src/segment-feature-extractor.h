// baud/segment-feature-extractor.h

// Author: David Harwath

#ifndef KALDI_BAUD_SEGMENT_FEATURE_EXTRACTOR_H
#define KALDI_BAUD_SEGMENT_FEATURE_EXTRACTOR_H

#include <iostream>
#include <vector>

#include "base/kaldi-common.h"
#include "matrix/kaldi-matrix.h"
#include "util/common-utils.h"

namespace kaldi {
class SegmentFeatureExtractor {
public:
	SegmentFeatureExtractor() {}

  ~SegmentFeatureExtractor() {}

  // Should really merge these into a templated function
	Vector<BaseFloat> AvgThirdsPlusDuration(const SubMatrix<BaseFloat> &features) const;

	Vector<double> AvgThirdsPlusDuration(const SubMatrix<double> &features) const;

private:
};

}  // end namespace kaldi

#endif