cc="--test-loc ../CodeContest_data_split/test.json"
neo="-l ../models/2.7B"

extra="--pop-size 200"

# run on APPS
start=0
end=5000
# use GPT-2
./scripts/run_smc.sh $start $end "smc-200-" "results" $extra
# use GPT-Neo
./scripts/run_smc.sh $start $end "smc-200-neo-" "results" "$extra $neo"

# run on CodeContests
start=0
end=165
# use GPT-2
./scripts/run_smc.sh $start $end "smc-200-" "cc_results" "$extra $cc"
# use GPT-Neo
./scripts/run_smc.sh $start $end "smc-200-neo-" "cc_results" "$extra $cc $neo"