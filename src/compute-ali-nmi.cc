// baudbin/compute-ali-nmi.cc

// Author: David Harwath (2014)

#include <map>

#include "base/kaldi-common.h"
#include "hmm/transition-model.h"
#include "util/common-utils.h"
#include "util/timer.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Compute the normalized mutual information between alignments.\n"
        "If you have state level alignments, you should first convert them to\n"
        "phone-level alignments using ali-to-phones with the --per-frame=true option.\n"
        "Usage: compute-ali-nmi [options] ali-1-rspecifier ali-2-rspecifier\n";

    ParseOptions po(usage);
    Timer timer;

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string ali_1_rspecifier = po.GetArg(1),
                ali_2_rspecifier = po.GetArg(2);

    SequentialInt32VectorReader ali_1_reader(ali_1_rspecifier);
    RandomAccessInt32VectorReader ali_2_reader(ali_2_rspecifier);

    int32 num_err = 0;
    int32 utt_count = 0;
    // Map to store the cooccurance counts indexed by the units from ali_1
    std::map<std::pair<int32, int32>, BaseFloat> counts_1_2;
    // The inverse map
    std::map<std::pair<int32, int32>, BaseFloat> counts_2_1;
    // The map to store marginal class counts for ali_1
    std::map<int32, BaseFloat> counts_1;
    // Ditto for ali_2
    std::map<int32, BaseFloat> counts_2;

    // Read in all the data
    for (; !ali_1_reader.Done(); ali_1_reader.Next()) {
      std::string utt_id = ali_1_reader.Key();
      const std::vector<int32> &ali_1 = ali_1_reader.Value();
      if (!ali_2_reader.HasKey(utt_id)) {
        KALDI_WARN << "Warning: No alignment pair found for utterance " << utt_id;
        num_err++;
        continue;
      }
      const std::vector<int32> &ali_2 = ali_2_reader.Value(utt_id);
      if (ali_1.size() != ali_2.size()) {
        KALDI_WARN << "Warning: Differing alignment sizes " << ali_1.size() <<
        " and " << ali_2.size() << " found for utterance " << utt_id;
        num_err++;
        continue;
      }
      for (int32 i = 0; i < ali_1.size(); ++i) {
        const int32 &class_1 = ali_1[i];
        const int32 &class_2 = ali_2[i];
        std::pair<int32, int32> pair_1_2 = std::make_pair(class_1, class_2);
        std::pair<int32, int32> pair_2_1 = std::make_pair(class_2, class_1);
        if (counts_1_2.find(pair_1_2) != counts_1_2.end()) {
          counts_1_2[pair_1_2] = counts_1_2[pair_1_2] + 1.0;
          counts_2_1[pair_2_1] = counts_2_1[pair_2_1] + 1.0;
        } else {
          counts_1_2[pair_1_2] = 1.0;
          counts_2_1[pair_2_1] = 1.0;
        }
        if (counts_1.find(class_1) != counts_1.end()) {
          counts_1[class_1] = counts_1[class_1] + 1.0;
        } else {
          counts_1[class_1] = 1.0;
        }
        if (counts_2.find(class_2) != counts_2.end()) {
          counts_2[class_2] = counts_2[class_2] + 1.0;
        } else {
          counts_2[class_2] = 1.0;
        }
      }
      utt_count++;
    }
    // Compute the probability distributions
    std::map<std::pair<int32, int32>, BaseFloat> P_1_2;
    std::map<int32, BaseFloat> P_1;
    std::map<int32, BaseFloat> P_2;
    BaseFloat sum_1 = 0.0;
    BaseFloat sum_2 = 0.0;
    // Sum up the total counts
    for (std::map<int32, BaseFloat>::iterator iter = counts_1.begin(); iter != counts_1.end(); ++iter) {
      const BaseFloat &value = iter->second;
      sum_1 += value;
    }
    for (std::map<int32, BaseFloat>::iterator iter = counts_2.begin(); iter != counts_2.end(); ++iter) {
      const BaseFloat &value = iter->second;
      sum_2 += value;
    }
    KALDI_ASSERT(sum_1 > 0);
    KALDI_ASSERT(sum_2 > 0);
    // Compute the marginal distributions
    for (std::map<int32, BaseFloat>::iterator iter = counts_1.begin(); iter != counts_1.end(); ++iter) {
      const int32 &key = iter->first;
      const BaseFloat &value = iter->second;
      P_1[key] = value / sum_1;
    }
    for (std::map<int32, BaseFloat>::iterator iter = counts_2.begin(); iter != counts_2.end(); ++iter) {
      const int32 &key = iter->first;
      const BaseFloat &value = iter->second;
      P_2[key] = value / sum_2;
    }
    // Compute the joint distribution
    for (std::map<std::pair<int32, int32>, BaseFloat>::iterator iter = counts_1_2.begin(); iter != counts_1_2.end(); ++iter) {
      const std::pair<int32, int32> &key = iter->first;
      const BaseFloat &value = iter->second;
      P_1_2[key] = value / sum_1;
    }
    // Now compute I_1_2, H_1, and H_2
    BaseFloat I_1_2 = 0.0;
    BaseFloat H_1 = 0.0;
    BaseFloat H_2 = 0.0;
    for (std::map<std::pair<int32, int32>, BaseFloat>::iterator iter = P_1_2.begin(); iter != P_1_2.end(); ++iter) {
      const std::pair<int32, int32> &key = iter->first;
      const int32 &key_1 = key.first;
      const int32 &key_2 = key.second;
      const BaseFloat &value = iter->second;
      I_1_2 += value * (std::log(value) - std::log(P_1[key_1]) - std::log(P_2[key_2]));
    }
    for (std::map<int32, BaseFloat>::iterator iter = P_1.begin(); iter != P_1.end(); ++iter) {
      const BaseFloat &value = iter->second;
      H_1 -= value * std::log(value);
    }
    for (std::map<int32, BaseFloat>::iterator iter = P_2.begin(); iter != P_2.end(); ++iter) {
      const BaseFloat &value = iter->second;
      H_2 -= value * std::log(value);
    }
    // Now compute NMI
    BaseFloat NMI = 2 * I_1_2 / (H_1 + H_2);
    KALDI_LOG << "NMI = " << NMI;

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed;
    KALDI_LOG << "Done " << utt_count << " utterances, failed for "
              << num_err;


    if (utt_count != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
