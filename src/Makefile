
bin:

EXTRA_CXXFLAGS = -Wno-sign-compare -std=c++11
include ../kaldi.mk

BINFILES = hmm-gmm-sample-ali compute-ali-nmi segment-gmm-sample-ali segment-gmm-acc-stats-ali segment-gmm-init-mono

OBJFILES =

TESTFILES =

ADDLIBS = ../baud/kaldi-baud.a ../landmarks/kaldi-landmarks.a ../hmm/kaldi-hmm.a ../gmm/kaldi-gmm.a \
	../tree/kaldi-tree.a ../thread/kaldi-thread.a ../matrix/kaldi-matrix.a ../util/kaldi-util.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk

lib:

include ../kaldi.mk

EXTRA_CXXFLAGS = -std=c++11

OBJFILES = hmm-gmm-boundary-sampler.o segment-feature-extractor.o segment-gmm-boundary-sampler.o

LIBNAME = kaldi-baud

ADDLIBS = ../landmarks/kaldi-landmarks.a ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../hmm/kaldi-hmm.a \
	../thread/kaldi-thread.a ../matrix/kaldi-matrix.a ../util/kaldi-util.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk
