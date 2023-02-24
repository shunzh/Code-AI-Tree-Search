start=4000
end=4500
rollout=5000
results="results"

for ucb in 4
do
  for ucb_base in 10
  do
    extra="--rollout $rollout --ucb-constant $ucb --ucb-base $ucb_base"
    # use sampling instead of beam search in PG-TD
    ./scripts/run.sh $start $end "ts-ucb$ucb-$ucb_base-" $results "$extra --ts-mode sample"

    for j in 1 3 5
    do
      # use different number of test cases
      ./scripts/run.sh $start $end "t-pc$j-ucb$ucb-$ucb_base-" $results "$extra --public-cases $j"
    done
  done
done