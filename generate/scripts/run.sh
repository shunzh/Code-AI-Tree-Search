source scripts/basis.sh
start=$1
end=$2
prefix=$3
save=$4
extra=$5

for (( i = $start; i < $end; i++ ))
do
  if [ ! -f "$save/$prefix$i.json" ] && [ ! -f "$save/$prefix$i.log" ]
  then
    if [ "$flag" == "aimos" ]
    then
      # cannot submit all jobs all at once on aimos
      eval $submit python synthesis_exp.py -i $i --prefix $prefix --save $save $extra &
      sleep 5
    else
      eval $submit python synthesis_exp.py -i $i --prefix $prefix --save $save $extra
    fi
  else
    echo "Skipping $prefix$i"
  fi
done