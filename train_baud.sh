#!/bin/bash

# To be run from ..
# Flat start and monophone training, with delta-delta features.
# This script applies cepstral mean normalization (per speaker).

# Begin configuration section.
nj=4
cmd=run.pl
num_iters=20
#num_gibbs_iters=15    # Number of iterations of training
#num_viterbi_iters=15
totgauss=1500 # Target #Gaussians.  
max_iter_inc=15 # Last iter to increase #Gauss on.
power=0.25 # exponent to determine number of gaussians from occurrence counts
boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment
config= # name of config file.
stage=-5
norm_vars=false # false : cmn, true : cmvn
sample_freq=16000
ref_ali_dir=none
anneal_scale=1
anneal_decay=1.0
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: steps/train_baud.sh [options] <data-dir> <lang-dir> <exp-dir>"
  echo " e.g.: steps/train_baud.sh data/train.1k data/lang exp/mono"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --feat_dim <dim>                                 # This option is ignored now but"
  echo "                                                   # retained for back-compatibility."
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --ref_ali_dir <ali-dir>                          # Where to find reference alignments for NMI"
  exit 1;
fi

data=$1
lang=$2
dir=$3

ref_ali_file=$dir/ref_ali
ref_phones_file=$dir/ref_phones
hyp_ali_file=$dir/hyp_ali
hyp_phones_file=$dir/hyp_phones
nmi_file=$dir/nmi

mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

echo $norm_vars > $dir/norm_vars # keep track of feature normalization type for decoding, alignment
#feats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |"
feats="ark,s,cs:compute-cmvn-stats scp:$sdata/JOB/feats.scp ark:- | apply-cmvn --norm-vars=$norm_vars ark:- scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |"
example_feats="`echo $feats | sed s/JOB/1/g`";
echo $example_feats

# get reference alignments
if [ -f $ref_ali_file ]; then
  rm -f $ref_ali_file
fi
if [ -f $ref_ali_dir/final.mdl ]; then
  find $ref_ali_dir/ -name 'ali*.gz' | xargs -I{} gunzip -c {} >> $ref_ali_file
  ali-to-phones --per-frame=true $ref_ali_dir/final.mdl ark:$ref_ali_file ark,t:$ref_phones_file
fi

# compute landmarks
if [ $stage -le -4 ]; then
  if [ -f $data/segments ]; then
      echo "$0 [info]: segments file exists: using that. Assuming segments are already split."
      $cmd JOB=1:$nj $dir/log/landmarks.JOB.log \
      	extract-segments scp,p:$sdata/JOB/wav.scp $sdata/JOB/segments ark:- \| \
      	compute-spectrogram-feats --window-type=hamming --sample-frequency=$sample_freq --round-to-power-of-two=false \
          --dft-length=1024 --frame-length=10 --frame-shift=5 ark:- ark:- \| \
          compute-landmarks --sigma=1.5 --major-only=false --frame-change-increment=2 --span=8 --min-segment-length=4 \
          --boundary-threshold=15 --cluster-threshold=25 ark:- ark,t:$sdata/JOB/landmarks.ark || exit 1;
  else
      echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
      $cmd JOB=1:$nj $dir/log/landmarks.JOB.log \
          compute-spectrogram-feats --window-type=hamming --sample-frequency=$sample_freq --round-to-power-of-two=false \
          --dft-length=1024 --frame-length=10 --frame-shift=5 scp:$sdata/JOB/wav.scp ark:- \| \
          compute-landmarks --sigma=1.5 --major-only=false --frame-change-increment=2 --span=8 --min-segment-length=4 \
          --boundary-threshold=15 --cluster-threshold=20 ark:- ark,t:$sdata/JOB/landmarks.ark || exit 1;
  fi
fi
echo "$0: Initializing GMM-HMM acoustic model."

if [ $stage -le -3 ]; then
# Note: JOB=1 just uses the 1st part of the features-- we only need a subset anyway.
  feat_dim=`feat-to-dim "$example_feats" - 2>/dev/null`
  [ -z "$feat_dim" ] && echo "error getting feature dimension" && exit 1;
  $cmd JOB=1 $dir/log/init.log \
    gmm-init-mono "--train-feats=$feats subset-feats --n=10 ark:- ark:-|" $lang/topo $feat_dim \
    $dir/0.mdl $dir/tree || exit 1;
fi

numgauss=`gmm-info --print-args=false $dir/0.mdl | grep gaussians | awk '{print $NF}'`
incgauss=$[($totgauss-$numgauss)/$max_iter_inc] # per-iter increment for #Gauss

if [ $stage -le -1 ]; then
  echo "$0: Performing initial alignment"
  $cmd JOB=1:$nj $dir/log/align.0.JOB.log \
      hmm-gmm-sample-ali --major-only=true --major-boundary-alpha=1.0 $dir/tree $dir/0.mdl "$feats" ark:$sdata/JOB/landmarks.ark \
      "ark,t:|gzip -c >$dir/ali.JOB.gz" ark,t:$dir/JOB.stats || exit 1;
  $cmd JOB=1:$nj $dir/log/acc.0.JOB.log \
      gmm-acc-stats-ali --binary=true $dir/0.mdl "$feats" "ark:gunzip -c $dir/ali.JOB.gz|" $dir/0.JOB.acc || exit 1;
  stats=`ls $dir/*.stats | sed -e 's/^/ark:/g' -e 's/\n/ /g'`
  vector-sum $stats ark,t:$dir/class_counts >/dev/null || exit 1;
  rm $dir/*.stats 2>/dev/null
  # Get hyp alignments and compute initial NMI
  if [ -f $nmi_file ]; then
    rm -f $nmi_file
  fi
  if [ -f $ref_phones_file ]; then
    find $dir/ -name 'ali*.gz' | xargs -I{} gunzip -c {} >> $hyp_ali_file
    ali-to-phones --per-frame=true $dir/0.mdl ark:$hyp_ali_file ark,t:$hyp_phones_file
    compute-ali-nmi ark:$ref_phones_file ark:$hyp_phones_file 2> $dir/nmi_temp
    this_nmi=`grep 'NMI' $dir/nmi_temp | cut -d' ' -f3-`
    echo "Iter 0 " $this_nmi >> $nmi_file
  fi
fi

# In the following steps, the --min-gaussian-occupancy=3 option is important, otherwise
# we fail to est "rare" phones and later on, they never align properly.

if [ $stage -le 0 ]; then
  gmm-est --min-gaussian-occupancy=3  --mix-up=$numgauss --power=$power \
    $dir/0.mdl "gmm-sum-accs - $dir/0.*.acc|" $dir/1.mdl 2> $dir/log/update.0.log || exit 1;
  rm $dir/0.*.acc
fi

x=1
scale=$anneal_scale
decay=$anneal_decay
while [ $x -lt $num_iters ]; do
  echo "$0: Pass $x"
  if [ $stage -le $x ]; then
    echo "$0: Aligning data"
    $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
      hmm-gmm-sample-ali --major-boundary-alpha=0.9 --minor-boundary-alpha=0.2 --likelihood-scale=$scale --stats-rspecifier=ark:$dir/class_counts $dir/tree $dir/$x.mdl \
      "$feats" ark:$sdata/JOB/landmarks.ark "ark,t:|gzip -c >$dir/ali.JOB.gz" \
      ark,t:$dir/JOB.stats || exit 1;
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-acc-stats-ali $dir/$x.mdl "$feats" "ark:gunzip -c $dir/ali.JOB.gz|" \
      $dir/$x.JOB.acc || exit 1;
    $cmd $dir/log/update.$x.log \
      gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss --power=$power $dir/$x.mdl \
      "gmm-sum-accs - $dir/$x.*.acc|" $dir/$[$x+1].mdl || exit 1;
    stats=`ls $dir/*.stats | sed -e 's/^/ark:/g' -e 's/\n/ /g'`
    vector-sum $stats ark,t:$dir/class_counts > /dev/null
    rm $dir/*.stats 2>/dev/null
    rm $dir/$x.mdl $dir/$x.*.acc $dir/$x.occs 2>/dev/null
  fi
  if [ $x -le $max_iter_inc ]; then
     numgauss=$[$numgauss+$incgauss];
  fi
  if [ -f $ref_phones_file ]; then
    rm -f $hyp_ali_file
    find $dir/ -name 'ali*.gz' | xargs -I{} gunzip -c {} >> $hyp_ali_file
    ali-to-phones --per-frame=true $dir/$[$x+1].mdl ark:$hyp_ali_file ark,t:$hyp_phones_file
    compute-ali-nmi ark:$ref_phones_file ark:$hyp_phones_file 2> $dir/nmi_temp
    this_nmi=`grep 'NMI' $dir/nmi_temp | cut -d' ' -f3-`
    echo "Iter $x " $this_nmi >> $nmi_file
  fi
  x=$[$x+1]
  scale=$(echo "$scale*$decay" | bc -l)
done

( cd $dir; rm final.{mdl,occs} 2>/dev/null; ln -s $x.mdl final.mdl; ln -s $x.occs final.occs )

utils/summarize_warnings.pl $dir/log

echo Done
