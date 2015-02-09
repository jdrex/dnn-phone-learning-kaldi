// baudbin/segment-gmm-acc-stats-ali.cc

// Author: David Harwath

#include "base/kaldi-common.h"
#include "baud/segment-feature-extractor.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "gmm/mle-am-diag-gmm.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Accumulate stats for segmental GMM training.\n"
        "Alignments used here are assumed to map frames to pdf-ids, not transition-ids.\n"
        "Contiguous blocks of frames sharing the same pdf-id are treated as a single segment.\n"
        "A single feature vector is extracted for each segment and accumulated for GMM training.\n"
        "Usage:  segment-gmm-acc-stats-ali [options] <model-in> <feature-rspecifier> "
        "<alignments-rspecifier> <stats-out>\n"
        "e.g.:\n segment-gmm-acc-stats-ali 1.mdl scp:train.scp ark:1.ali 1.acc\n";

    ParseOptions po(usage);
    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");
    // add a boolean option to do sampling-based accumulation
    // of sufficient statistics
    bool flag_gibbs = false;
    po.Register("gibbs", &flag_gibbs, "Running sampling-based accumulation");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
        po.PrintUsage();
        exit(1);
      }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        alignments_rspecifier = po.GetArg(3),
        accs_wxfilename = po.GetArg(4);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    Vector<double> transition_accs;
    trans_model.InitStats(&transition_accs);
    AccumAmDiagGmm gmm_accs;
    gmm_accs.Init(am_gmm, kGmmAll);

    double tot_like = 0.0;
    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);
    SegmentFeatureExtractor extractor;

    int32 num_done = 0, num_err = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      if (!alignments_reader.HasKey(key)) {
        KALDI_WARN << "No alignment for utterance " << key;
        num_err++;
      } else {
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const std::vector<int32> &alignment = alignments_reader.Value(key);
        if (alignment.size() != mat.NumRows()) {
          KALDI_WARN << "Alignments has wrong size " << (alignment.size())
                     << " vs. " << (mat.NumRows());
          num_err++;
          continue;
        }
        if (alignment.size() == 0) {
          KALDI_WARN << "Warning: Zero length alignment for utterance " << key;
          num_err++;
          continue;
        }

        num_done++;
        BaseFloat tot_like_this_file = 0.0;
        int32 prev_pdf_id = -1;
        int32 start_frame = -1;
        for (size_t i = 0; i < alignment.size(); i++) {
          int32 pdf_id = alignment[i];
          if (pdf_id != prev_pdf_id) {
            if (prev_pdf_id != -1) {
              SubMatrix<BaseFloat> seg_feats = SubMatrix<BaseFloat>(mat, start_frame, i - start_frame, 0, mat.NumCols());
              Vector<BaseFloat> feats = extractor.AvgThirdsPlusDuration(seg_feats);
              if (flag_gibbs) {
                tot_like_this_file
                    += gmm_accs.GibbsAccumulateForGmm(am_gmm, feats,
                                                      prev_pdf_id, 1.0);
              } else {
                tot_like_this_file += gmm_accs.AccumulateForGmm(am_gmm, feats,
                                                                prev_pdf_id, 1.0);
              }
            }
            start_frame = i;
          }
          prev_pdf_id = pdf_id;
        }
        // Account for the very last segment
        SubMatrix<BaseFloat> seg_feats = SubMatrix<BaseFloat>(mat, start_frame, mat.NumRows() - start_frame, 0, mat.NumCols());
        Vector<BaseFloat> feats = extractor.AvgThirdsPlusDuration(seg_feats);
        if (flag_gibbs) {
          tot_like_this_file
              += gmm_accs.GibbsAccumulateForGmm(am_gmm, feats,
                                                prev_pdf_id, 1.0);
        } else {
          tot_like_this_file += gmm_accs.AccumulateForGmm(am_gmm, feats,
                                                          prev_pdf_id, 1.0);
        }


        tot_like += tot_like_this_file;
        tot_t += alignment.size();
        if (num_done % 50 == 0) {
          KALDI_LOG << "Processed " << num_done << " utterances; for utterance "
                    << key << " avg. like is "
                    << (tot_like_this_file/alignment.size())
                    << " over " << alignment.size() <<" frames.";
        }
      }
    }
    KALDI_LOG << "Done " << num_done << " files, " << num_err
              << " with errors.";

    KALDI_LOG << "Overall avg like per frame (Gaussian only) = "
              << (tot_like/tot_t) << " over " << tot_t << " frames.";

    {
      Output ko(accs_wxfilename, binary);
      transition_accs.Write(ko.Stream(), binary);
      gmm_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written accs.";
    if (num_done != 0)
      return 0;
    else
      return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


