./scrpts/basis.sh
cc="--test-loc ../CodeContest_data_split/test.json" # specify the location of CodeContests test set
neo="-l ../models/2.7B"

extra='--alg bs'

# beam search is much faster than the other algorithms, so we can call synthesis_exp.py once to run all problems in a dataset

# run on APPS
start=0
end=5000
# use GPT-2
$submit python synthesis_exp.py -s $start -e $end $extra
# use GPT-Neo
$submit python synthesis_exp.py -s $start -e $end --prefix "neo-" $extra $neo

# run on CodeContests
start=0
end=165
# use GPT-2
$submit python synthesis_exp.py -s $start -e $end --save "cc_results" $extra $cc
# use GPT-Neo
$submit python synthesis_exp.py -s $start -e $end --save "cc_results" --prefix "neo-" $extra $cc