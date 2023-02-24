cc="--test-loc ../CodeContest_data_split/test.json"
neo="-l ../models/2.7B"

extra="--num-samples 512 --alg sample"

# run on APPS
start=0
end=5000
# use GPT-2
./scripts/run.sh $start $end "s-" "results" $extra
# use GPT-Neo
./scripts/run.sh $start $end "s-neo-" "results" "$extra $neo"

# run on CodeContests
start=0
end=165
# use GPT-2
./scripts/run.sh $start $end "s-" "cc_results" "$extra $cc"
# use GPT-Neo
./scripts/run.sh $start $end "s-neo-" "cc_results" "$extra $cc $neo"