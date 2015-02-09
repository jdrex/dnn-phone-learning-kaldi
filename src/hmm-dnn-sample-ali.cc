// baudbin/hmm-dnn-sample-ali.cc

// Author: David Harwath (2014)

#include "base/kaldi-common.h"
#include "baud/hmm-dnn-boundary-sampler.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "landmarks/landmark-utils.h"
#include "tree/context-dep.h"
#include "util/common-utils.h"
#include "util/timer.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Gibbs re-sample alignments using HMM-DNN-based models.\n"
        "Usage: hmm-dnn-sample-ali [options] tree-in model-in features-rspecifier"
        " landmarks-rspecifier alignments-wspecifier stats-wspecifier\n";

    ParseOptions po(usage);
    Timer timer;
    HmmDnnBoundarySamplerConfig config;

    std::string stats_rspecifier = "none";
    std::string alignments_rspecifier = "none";
    bool major_only = true;
    bool resolve_landmarks = true;
    po.Register("stats-rspecifier", &stats_rspecifier, "Table containing counts of how many "
                "segments have been assigned to each acoustic model. If none is "
                "specified, initializes all counts to one.");
    po.Register("ali-in", &alignments_rspecifier, "rspecifier for reading alignments. "
      "These alignments are used to glean the previous values of boundaries. If none "
      "is specified, landmarks are assumed to be \"on\" boundaries.");
    po.Register("major-only", &major_only, "When the ali-in option is not specified and "
      "landmarks are used to initialize boundaries, setting this option to true will "
      "cause only major landmarks to be turned \"on\" (minor landmarks will be off)");
    po.Register("resolve-landmarks", &resolve_landmarks, "When the frame resolution of "
      "the landmarks does not match that of the features, attempt to map the landmarks "
      "onto the appropriate feature frames.");

    config.Register(&po);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 6) {

      po.PrintUsage();
      exit(1);
    }

    std::string tree_in_filename = po.GetArg(1),
                model_in_filename = po.GetArg(2),
                features_rspecifier = po.GetArg(3),
                landmarks_rspecifier = po.GetArg(4),
                alignments_wspecifier = po.GetArg(5),
                stats_wspecifier = po.GetArg(6);

    const std::string stats_key = "baud_stats";
    ContextDependency ctx_dep;
    ReadKaldiObject(tree_in_filename, &ctx_dep);
    
    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    KALDI_LOG << "Number of phones in trans_model is " << trans_model.NumPhones();
    
    std::vector<int32> class_counts;
    if (stats_rspecifier != "none") {
      RandomAccessBaseFloatVectorReader vector_reader(stats_rspecifier);
      if (vector_reader.HasKey(stats_key)) {
        const Vector<BaseFloat> &class_counts_vector = vector_reader.Value(stats_key);
        for (int32 i = 0; i < class_counts_vector.Dim(); ++i) {
          class_counts.push_back(std::round(class_counts_vector(i)));
        }
      } else {
        KALDI_WARN << "Incorrect stats vector specified. Initializing all class counts to 1.";
        for (int32 i = 0; i < trans_model.NumPhones(); ++i) {
          class_counts.push_back(1);
        }
      }
    } else {
      for (int32 i = 0; i < trans_model.NumPhones(); ++i) {
        class_counts.push_back(1);
      }
    }

    RandomAccessInt32VectorReader alignments_reader;
    if (alignments_rspecifier != "none" ) {
      alignments_reader.Open(alignments_rspecifier);
    }

    SequentialBaseFloatMatrixReader features_reader(features_rspecifier);
    RandomAccessLandmarkSeqReader landmarks_reader(landmarks_rspecifier);
    Int32VectorWriter alignments_writer(alignments_wspecifier);

    HmmDnnBoundarySampler sampler(config);
    
    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_done = 0, num_err = 0;

    // The new class counts which we will write out
    std::vector<int32> new_class_counts(class_counts.size());

    for (; !features_reader.Done(); features_reader.Next()) {
      std::string utt = features_reader.Key();
      if (!landmarks_reader.HasKey(utt)) {
        KALDI_WARN << "No landmarks for utterance " << utt;
        num_err++;
        continue;
      }
      Matrix<BaseFloat> features(features_reader.Value());
      features_reader.FreeCurrent();
      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << utt;
        num_err++;
        continue;
      }
      // Need to read in landmarks, or set to NULL
      LandmarkSeq landmarks = landmarks_reader.Value(utt);
      if (landmarks.num_frames != features.NumRows() && !resolve_landmarks) {
        KALDI_WARN << "Landmarks have wrong size " << landmarks.num_frames
                    << " vs. " << (features.NumRows());
        num_err++;
        continue;
      }
      // Hack: back up all the by 1 frame. So 0 becomes -1, 10 becomes 9, etc.
      // Do this for all frames except the last one.
      // I do this because the major landmark algorithm tends to place major landmarks
      // slightly ahead of where they probably should be. Maybe I can fix this in the future.
      /*for (int32 l = 0; l < landmarks.landmarks.size() - 1; ++l) {
        const int32 loc = landmarks.landmarks[l].first;
        const landmark_type_t type = landmarks.landmarks[l].second;
        landmarks.landmarks[l] = std::make_pair(loc - 1, type);
      }*/
      // Set the first landmark at -1
      landmarks.landmarks[0] = std::make_pair(-1, MAJOR_LANDMARK);
      if (resolve_landmarks) {
        // Figure out the factor by which we need to stretch the landmarks
        // The first landmark stays at -1. The last landmark must go at features.NumRows() - 1.
        //KALDI_LOG << "Landmarks think there are " << landmarks.num_frames << " frames";
        //KALDI_LOG << features.NumRows() << " frames exist";
        BaseFloat landmark_rate_scale = 
          static_cast<BaseFloat>(landmarks.num_frames) / static_cast<BaseFloat>(features.NumRows());
        //KALDI_LOG << "Dividing landmark locations by " << landmark_rate_scale;
        for (int32 l = 1; l < landmarks.landmarks.size() - 1; ++l) {
          const landmark_type_t type = landmarks.landmarks[l].second;
          const BaseFloat loc = static_cast<BaseFloat>(landmarks.landmarks[l].first);
          const BaseFloat new_loc = loc / landmark_rate_scale;
          const int32 prev_loc = landmarks.landmarks[l - 1].first;
          const int32 new_loc_rounded = std::min(std::max(prev_loc + 1, static_cast<int32>(new_loc + 0.5)), features.NumRows() - 1);
          landmarks.landmarks[l] = std::make_pair(new_loc_rounded, type);
          //KALDI_LOG << "Moved landmark from frame " << loc << " to frame " << new_loc_rounded;
        }
        landmarks.landmarks[landmarks.landmarks.size() - 1] = std::make_pair(features.NumRows() - 1, MAJOR_LANDMARK);
        landmarks.num_frames = features.NumRows();
      }
      double like;
      std::vector<int32> old_bounds;
      // if previously aligned, just read the bounds in
      if (alignments_reader.IsOpen()) {
        if (!alignments_reader.HasKey(utt)) {
        KALDI_WARN << "No alignment for utterance " << utt;
        num_err++;
        continue;
        }
        const std::vector<int32> &alignment = alignments_reader.Value(utt);
        if (alignment.size() != features.NumRows()) {
          KALDI_WARN << "Alignments has wrong size " << (alignment.size())
                     << " vs. " << (features.NumRows());
          num_err++;
          continue;
        }
        std::vector<int32> old_clusters;
        sampler.GetBoundsFromAlignment(alignment, trans_model, &old_bounds, &old_clusters);
      } 
      else { // otherwise sample the initial bounds
        sampler.GetBoundsFromLandmarks(features, landmarks, major_only, &old_bounds);
      }
      
      // resample boundaries
      if (sampler.ResampleAlignment(features, old_bounds, landmarks, am_gmm, trans_model,
        ctx_dep, utt, class_counts, &new_class_counts, &alignments_writer, &like)) {
        tot_like += like;
        frame_count += features.NumRows();
        num_done++;
        if (num_done % 10 == 0) {
          double elapsed = timer.Elapsed();
          KALDI_LOG << "Finished " << num_done << " utterances with avg RT factor "
          << (elapsed * 100.0 / frame_count) << " assuming 100 frames/sec";
        }
      } else {
          num_err++;
      }
    }

    // Write out updated counts based on this block
    {
      Vector<BaseFloat> class_counts_vector(new_class_counts.size());
      for (int32 i = 0; i < new_class_counts.size(); ++i){
        class_counts_vector(i) = static_cast<BaseFloat>(new_class_counts[i]);
      }
      BaseFloatVectorWriter stats_writer(stats_wspecifier);
      stats_writer.Write(stats_key, class_counts_vector);
    }
      
    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_done << " utterances, failed for "
              << num_err;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count) << " over "
              << frame_count << " frames.";

    if (num_done != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
