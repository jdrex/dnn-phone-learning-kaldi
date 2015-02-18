#!/bin/bash

# To be run from ..
# Flat start and monophone training, with delta-delta features.
# This script applies cepstral mean normalization (per speaker).

# Begin configuration section.
nj=4
cmd=hostd2.pl
cuda_cmd=run.pl
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
ref_landmarks_file=$dir/ref_landmarks
hyp_ali_file=$dir/hyp_ali
hyp_phones_file=$dir/hyp_phones
hyp_landmarks_file=$dir/hyp_landmarks
nmi_file=$dir/nmi

mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

echo $norm_vars > $dir/norm_vars # keep track of feature normalization type for decoding, alignment
#feats="ark,s,cs:compute-cmvn-stats scp:$sdata/JOB/feats.scp ark:- | apply-cmvn --norm-vars=$norm_vars ark:- scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |"

# get reference alignments
if [ -f $ref_ali_file ]; then
  rm -f $ref_ali_file
fi
if [ -f $ref_ali_dir/final.mdl ]; then
  find $ref_ali_dir/ -name 'ali*.gz' | xargs -I{} gunzip -c {} >> $ref_ali_file
  ali-to-phones --per-frame=true $ref_ali_dir/final.mdl ark:$ref_ali_file ark,t:$ref_phones_file
  ali-to-landmarks ark:$ref_phones_file ark,t:$ref_landmarks_file
fi

# compute landmarks
if [ $stage -le 0 ]; then
    $cmd JOB=1:$nj $dir/log/spectrograms.JOB.log \
        compute-spectrogram-feats --window-type=hamming --sample-frequency=$sample_freq --round-to-power-of-two=false \
        --dft-length=1024 --frame-length=10 --frame-shift=5 scp:$sdata/JOB/wav.scp ark,scp:$sdata/JOB/spec_feats.ark,$sdata/JOB/spec_feats.scp
    
    $cmd JOB=1:$nj $dir/log/landmarks.JOB.log \
        compute-landmarks --sigma=1.5 --major-only=false --frame-change-increment=2 --span=8 --min-segment-length=4 \
        --boundary-threshold=15 --cluster-threshold=20 scp:$sdata/JOB/spec_feats.scp ark,t:$sdata/JOB/landmarks.ark || exit 1;

    # create one scp file for spectrograms
    cat $sdata/[0-9]*/spec_feats.scp > $data/spec_feats.scp
fi

if [ $stage -le 1 ]; then
    echo "$0: Pre-training DBN."
    # Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)
    # CHANGE THIS BACK TO 20
    $cuda_cmd $dir/log/pretrain_dbn.log steps/nnet/pretrain_dbn.sh --hid-dim 1024 --rbm-iter 2 $data $dir/dbn || exit 1;
  
    echo "$0: Initializing DNN from DBN"
    # make network prototype for softmax layer
    mlp_proto=$dir/dnn/nnet.proto
    utils/nnet/make_nnet_proto.py 64 300 0 1024 > $mlp_proto || exit 1

    # initialize softmax layer
    mlp_init=$dir/dnn/nnet.init; log=$dir/log/nnet_initialize.log
    nnet-initialize $mlp_proto $mlp_init 2>$log || { cat $log; exit 1; }

    # concat dbn and softmax layer
    mlp_init_old=$mlp_init; mlp_init=$dir/dnn/nnet_dbn_dnn.init
    nnet-concat $dir/dbn/final.nnet $mlp_init_old $mlp_init || exit 1 
fi


if [ $stage -le 2 ]; then
    feature_transform=$dir/dbn/final.feature_transform
    #Run the forward pass to generate BN features
    $cmd JOB=1:$nj $dir/log/make_bnfeats.JOB.log nnet-forward --feature-transform=$feature_transform $dir/dbn/final.nnet \
	scp:$sdata/JOB/spec_feats.scp ark,scp:$sdata/JOB/bn_feats.ark,scp:$sdata/JOB/bn_feats.scp || exit 1;

    echo "$0: Performing initial alignment"
    $cmd JOB=1:$nj $dir/log/align.0.JOB.log \
	hmm-dnn-sample-ali --major-only=true --major-boundary-alpha=1.0 $dir/tree $dir/0.mdl ark:$sdata/JOB/bn_feats.ark ark:$sdata/JOB/landmarks.ark \
	"ark,t:|gzip -c >$dir/ali.JOB.gz" ark,t:$dir/JOB.stats || exit 1;
    $cmd JOB=1:$nj $dir/log/acc.0.JOB.log \
	hmm-acc-stats-ali --binary=true $dir/0.mdl "$feats" "ark:gunzip -c $dir/ali.JOB.gz|" $dir/0.JOB.acc || exit 1;
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
	ali-to-landmarks ark:$hyp_phones_file ark,t:$hyp_landmarks_file
	score-landmarks ark:$hyp_landmarks_file ark:$ref_landmarks_file
	this_nmi=`grep 'NMI' $dir/nmi_temp | cut -d' ' -f3-`
	echo "Iter 0 " $this_nmi >> $nmi_file
    fi
fi

exit 0
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
	feature_transform=$dir/dbn/final.feature_transform
	nnet=$dir/dnn/feature_extractor.nnet
	nnet-copy --remove-last-layers=2 --binary=false $dir/dnn/final.nnet $nnet 2> $dir/log/feature_extractor.log

	rm $sdata/*/bn_feats*

	#Run the forward pass to generate BN features
	$cmd JOB=1:$nj $dir/log/make_bnfeats.JOB.log nnet-forward --feature-transform=$feature_transform $nnet \
	    scp:$sdata/JOB/spec_feats.scp ark,scp:$sdata/JOB/bn_feats.ark,scp:$sdata/JOB/bn_feats.scp || exit 1;

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
	ali-to-landmarks ark:$hyp_phones_file ark,t:$hyp_landmarks_file
	score-landmarks ark:$hyp_landmarks_file ark:$ref_landmarks_file
	this_nmi=`grep 'NMI' $dir/nmi_temp | cut -d' ' -f3-`
	echo "Iter $x " $this_nmi >> $nmi_file
    fi
    x=$[$x+1]
    scale=$(echo "$scale*$decay" | bc -l)
done

( cd $dir; rm final.{mdl,occs} 2>/dev/null; ln -s $x.mdl final.mdl; ln -s $x.occs final.occs )

utils/summarize_warnings.pl $dir/log

echo Done
