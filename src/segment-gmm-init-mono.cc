// baudbin/segment-gmm-init-mono.cc

// Author: David Harwath

#include "base/kaldi-common.h"
#include "baud/segment-feature-extractor.h"
#include "landmarks/landmark-utils.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/hmm-topology.h"
#include "hmm/transition-model.h"

namespace kaldi {
// This function reads a file like:
// 1 2 3
// 4 5
// 6 7 8
// where each line is a list of integer id's of phones (that should have their pdfs shared).
void ReadSharedPhonesList(std::string rxfilename, std::vector<std::vector<int32> > *list_out) {
  list_out->clear();
  Input input(rxfilename);
  std::istream &is = input.Stream();
  std::string line;
  while (std::getline(is, line)) {
    list_out->push_back(std::vector<int32>());
    if (!SplitStringToIntegers(line, " \t\r", true, &(list_out->back())))
      KALDI_ERR << "Bad line in shared phones list: " << line << " (reading "
                << PrintableRxfilename(rxfilename) << ")";
    std::sort(list_out->rbegin()->begin(), list_out->rbegin()->end());
    if (!IsSortedAndUniq(*(list_out->rbegin())))
      KALDI_ERR << "Bad line in shared phones list (repeated phone): " << line
                << " (reading " << PrintableRxfilename(rxfilename) << ")";
  }
}

} // end namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;
    using kaldi::MAJOR_LANDMARK;
    using kaldi::MINOR_LANDMARK;

    const char *usage =
        "Initialize monophone GMM.\n"
        "Usage:  gmm-init-mono <topology-in> <dim> <model-out> <tree-out> \n"
        "e.g.: \n"
        " gmm-init-mono topo 39 mono.mdl mono.tree\n";

    bool binary = true;
    std::string train_feats;
    std::string train_landmarks;
    std::string shared_phones_rxfilename;
    bool resolve_landmarks = true;
    bool major_only = true;
    BaseFloat perturb_factor = 0.0;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("train-feats", &train_feats,
                "rspecifier for training features [used to set mean and variance]");
    po.Register("train-landmarks", &train_landmarks,
                "rspecifier for landmarks [used to extract segments from train feats");
    po.Register("shared-phones", &shared_phones_rxfilename,
                "rxfilename containing, on each line, a list of phones whose pdfs should be shared.");
    po.Register("perturb-factor", &perturb_factor,
                "Perturb the means using this fraction of standard deviation.");
    po.Register("resolve-landmarks", &resolve_landmarks, "Resolve landmarks to feature frames if they are at a different framerate");
    po.Register("major-only", &major_only, "Compute segment stats based on segments derived from major landmarks only");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }


    std::string topo_filename = po.GetArg(1);
    int dim = atoi(po.GetArg(2).c_str());
    KALDI_ASSERT(dim> 0 && dim < 10000);
    std::string model_filename = po.GetArg(3);
    std::string tree_filename = po.GetArg(4);
    Vector<BaseFloat> glob_inv_var(dim);
    glob_inv_var.Set(1.0);
    Vector<BaseFloat> glob_mean(dim);
    glob_mean.Set(1.0);
    if (train_feats != "" && train_landmarks == "") {
      KALDI_LOG << "Not using supplied training features; landmarks must also be supplied.";
    }
    if (train_feats == "" && train_landmarks != "") {
      KALDI_LOG << "Not using supplied training landmarks; features must also be supplied.";
    }
    if (train_feats != "" && train_landmarks != "") {
      SegmentFeatureExtractor extractor;
      double count = 0.0;
      Vector<double> var_stats(dim);
      Vector<double> mean_stats(dim);
      SequentialDoubleMatrixReader feat_reader(train_feats);
      RandomAccessLandmarkSeqReader landmark_reader(train_landmarks);
      for (; !feat_reader.Done(); feat_reader.Next()) {
        const std::string &key = feat_reader.Key();
        const Matrix<double> &mat = feat_reader.Value();
        if (!landmark_reader.HasKey(key)) {
          KALDI_WARN << "No landmarks found for utterance " << key;
          continue;
        }
        LandmarkSeq landmarks_temp = landmark_reader.Value(key);
        LandmarkSeq landmarks;
        landmarks.utt_id = landmarks_temp.utt_id;
        landmarks.num_frames = landmarks_temp.num_frames;
        for (int32 i = 0; i < landmarks_temp.landmarks.size(); ++i) {
          if (major_only) {
            if (landmarks_temp.landmarks[i].second == MAJOR_LANDMARK) {
              landmarks.landmarks.push_back(landmarks_temp.landmarks[i]);
            }
          } else {
            landmarks.landmarks.push_back(landmarks_temp.landmarks[i]);
          }
        }
        // Hack: back up all the by 1 frame. So 0 becomes -1, 10 becomes 9, etc.
        // Do this for all frames except the last one.
        // I do this because the major landmark algorithm tends to place major landmarks
        // slightly ahead of where they probably should be. Maybe I can fix this in the future.
        for (int32 l = 0; l < landmarks.landmarks.size() - 1; ++l) {
          const int32 loc = landmarks.landmarks[l].first;
          const landmark_type_t type = landmarks.landmarks[l].second;
          landmarks.landmarks[l] = std::make_pair(loc - 1, type);
        }
        if (resolve_landmarks) {
          // Figure out the factor by which we need to stretch the landmarks
          // The first landmark stays at -1. The last landmark must go at features.NumRows() - 1.
          landmarks.num_frames = mat.NumRows();
          BaseFloat landmark_rate_scale = 
            static_cast<BaseFloat>(landmarks.num_frames) / static_cast<BaseFloat>(mat.NumRows());
          for (int32 l = 1; l < landmarks.landmarks.size() - 1; ++l) {
            const landmark_type_t type = landmarks.landmarks[l].second;
            const BaseFloat loc = static_cast<BaseFloat>(landmarks.landmarks[l].first);
            const BaseFloat new_loc = loc / landmark_rate_scale;
            const int32 prev_loc = landmarks.landmarks[l - 1].first;
            const int32 new_loc_rounded = std::min(std::max(prev_loc, static_cast<int32>(new_loc + 0.5)), mat.NumRows() - 1);
            landmarks.landmarks[l] = std::make_pair(new_loc_rounded, type);
          }
          landmarks.landmarks[landmarks.landmarks.size() - 1] = std::make_pair(mat.NumRows() - 1, MAJOR_LANDMARK);
        }
        for (int32 i = 1; i < landmarks.landmarks.size(); ++i) {
          int32 start_frame = landmarks.landmarks[i-1].first + 1;
          int32 end_frame = landmarks.landmarks[i].first;
          if (start_frame == end_frame + 1) {
            start_frame = end_frame;
          }
          if (start_frame < 0 || start_frame > mat.NumRows()) {
            KALDI_ERR << "start_frame is " << start_frame << ", NumRows is " << mat.NumRows();
          }
          if (end_frame < 0 || end_frame > mat.NumRows()) {
            KALDI_ERR << "end_frame is " << end_frame << ", NumRows is " << mat.NumRows();
          }
          if (start_frame > end_frame) {
            KALDI_LOG << "utt_id = " << key;
            KALDI_LOG << "original landmarks num_frames = " << landmarks_temp.num_frames;
            KALDI_LOG << "resolved landmarks num_frames = " << landmarks.num_frames;
            KALDI_LOG << "start_frame = " << start_frame;
            KALDI_LOG << "end_frame = " << end_frame;
            KALDI_LOG << "end_frame - start_frame + 1 = " << end_frame - start_frame + 1;
          }
          SubMatrix<double> seg_features =
            SubMatrix<double>(mat, start_frame, end_frame - start_frame + 1, 0, mat.NumCols());
          Vector<double> feats = extractor.AvgThirdsPlusDuration(seg_features);
          count += 1.0;
          var_stats.AddVec2(1.0, feats);
          mean_stats.AddVec(1.0, feats);
        }
      }
      if (count == 0) { KALDI_ERR << "no features were seen."; }
      var_stats.Scale(1.0/count);
      mean_stats.Scale(1.0/count);
      var_stats.AddVec2(-1.0, mean_stats);
      if (var_stats.Min() <= 0.0)
        KALDI_ERR << "bad variance";
      var_stats.InvertElements();
      glob_inv_var.CopyFromVec(var_stats);
      glob_mean.CopyFromVec(mean_stats);
      KALDI_LOG << "Computed global mean and variance from " << count << " segments.";
      KALDI_LOG << "Global mean is " << glob_mean;
      KALDI_LOG << "Global inverse variance is " << glob_inv_var;
    }

    HmmTopology topo;
    bool binary_in;
    Input ki(topo_filename, &binary_in);
    topo.Read(ki.Stream(), binary_in);

    const std::vector<int32> &phones = topo.GetPhones();

    std::vector<int32> phone2num_pdf_classes (1+phones.back());
    for (size_t i = 0; i < phones.size(); i++)
      phone2num_pdf_classes[phones[i]] = topo.NumPdfClasses(phones[i]);

    // Now the tree [not really a tree at this point]:
    ContextDependency *ctx_dep = NULL;
    if (shared_phones_rxfilename == "") {  // No sharing of phones: standard approach.
      ctx_dep = MonophoneContextDependency(phones, phone2num_pdf_classes);
    } else {
      std::vector<std::vector<int32> > shared_phones;
      ReadSharedPhonesList(shared_phones_rxfilename, &shared_phones);
      // ReadSharedPhonesList crashes on error.
      ctx_dep = MonophoneContextDependencyShared(shared_phones, phone2num_pdf_classes);
    }

    int32 num_pdfs = ctx_dep->NumPdfs();

    AmDiagGmm am_gmm;
    DiagGmm gmm;
    gmm.Resize(1, dim);
    {  // Initialize the gmm.
      Matrix<BaseFloat> inv_var(1, dim);
      inv_var.Row(0).CopyFromVec(glob_inv_var);
      Matrix<BaseFloat> mu(1, dim);
      mu.Row(0).CopyFromVec(glob_mean);
      Vector<BaseFloat> weights(1);
      weights.Set(1.0);
      gmm.SetInvVarsAndMeans(inv_var, mu);
      gmm.SetWeights(weights);
      gmm.ComputeGconsts();
    }

    for (int i = 0; i < num_pdfs; i++)
      am_gmm.AddPdf(gmm);

    if (perturb_factor != 0.0) {
      for (int i = 0; i < num_pdfs; i++)
        am_gmm.GetPdf(i).Perturb(perturb_factor);
    }

    // Now the transition model:
    TransitionModel trans_model(*ctx_dep, topo);

    {
      Output ko(model_filename, binary);
      trans_model.Write(ko.Stream(), binary);
      am_gmm.Write(ko.Stream(), binary);
    }

    // Now write the tree.
    ctx_dep->Write(Output(tree_filename, binary).Stream(),
                   binary);

    delete ctx_dep;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

