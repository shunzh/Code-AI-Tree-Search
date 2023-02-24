start=4000
end=5000
rollout=5000
results="results"

for ucb in 2 4 6
do
  for ucb_base in 10
  do
    extra="--rollout $rollout --ucb-constant $ucb --ucb-base $ucb_base --max-sample-times 512"

    ./scripts/run.sh $start $end "t-ucb$ucb-$ucb_base-" $results "$extra"
    ./scripts/run.sh $start $end "t-ucb$ucb-$ucb_base-b3-" $results "$extra --num-beams 3"
    ./scripts/run.sh $start $end "t-ucb$ucb-$ucb_base-b5-" $results "$extra --num-beams 5"

    ./scripts/run.sh $start $end "t2-ucb$ucb-$ucb_base-" $results "$extra --width 2"
    ./scripts/run.sh $start $end "t4-ucb$ucb-$ucb_base-" $results "$extra --width 4"
  done
done