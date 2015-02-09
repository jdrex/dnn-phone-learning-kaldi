// baud/dirichlet-process-utils.h

// Author: David Harwath

#ifndef KALDI_BAUD_DIRICHLET_PROCESS_UTILS_H
#define KALDI_BAUD_DIRICHLET_PROCESS_UTILS_H

#include <iostream>
#include <vector>

#include "base/kaldi-common.h"
#include "matrix/kaldi-matrix.h"
#include "util/common-utils.h"

namespace kaldi {

class DPClusterStats {
public:
	void Read(std::istream &in_stream, bool binary);
	void Write(std::ostream &out_stream, bool binary) const;
};

}  // end namespace kaldi

#endif