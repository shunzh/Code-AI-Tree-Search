cc="--test-loc ../CodeContest_data_split/test.json" # specify the location of CodeContests test set
neo="-l ../models/2.7B"

rollout=5000 # maximum number of rollouts of tree search
extra="--rollout $rollout"

# run on APPS
start=0
end=5000
# use GPT-2
./scripts/run.sh $start $end "t-" "results" $extra
# use GPT-Neo
./scripts/run.sh $start $end "t-neo-" "results" "$extra $neo"

# run on CodeContests
start=0
end=165
# use GPT-2
./scripts/run.sh $start $end "t-" "cc_results" "$extra $cc"
# use GPT-Neo
./scripts/run.sh $start $end "t-neo-" "cc_results" "$extra $cc $neo"