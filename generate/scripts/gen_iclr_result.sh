for karg in 'k128-' 'k256-' 'k512-' 'k1024-' ''
do
  python test_solution.py -s 4000 -e 5000 --prefix t3-3-samples-${karg} --save iclr_results --backup ../eval/results/all_results.json
  python test_solution.py -s 4000 -e 5000 --prefix t2-2-samples-${karg} --save iclr_results --backup ../eval/results/all_results.json
  python test_solution.py -s 4000 -e 5000 --prefix t4-4-samples-${karg} --save iclr_results --backup iclr_results/t3-3-samples-${karg}all_results.json

  python test_solution.py -s 4000 -e 5000 --prefix t3-3-b3-samples-${karg} --save iclr_results --backup iclr_results/t3-3-samples-${karg}all_results.json
  python test_solution.py -s 4000 -e 5000 --prefix t3-3-b5-samples-${karg} --save iclr_results --backup iclr_results/t3-3-samples-${karg}all_results.json

  python test_solution.py -s 4000 -e 5000 --prefix t3-6-samples-${karg} --save iclr_results --backup iclr_results/t3-3-samples-${karg}all_results.json
  python test_solution.py -s 4000 -e 5000 --prefix t3-6-b3-samples-${karg} --save iclr_results --backup iclr_results/t3-3-samples-${karg}all_results.json
  python test_solution.py -s 4000 -e 5000 --prefix t3-6-b5-samples-${karg} --save iclr_results --backup iclr_results/t3-6-samples-${karg}all_results.json

  python test_solution.py -s 4000 -e 5000 --prefix t3-9-samples-${karg} --save iclr_results --backup iclr_results/t3-6-samples-${karg}all_results.json
  python test_solution.py -s 4000 -e 5000 --prefix t3-9-b3-samples-${karg} --save iclr_results --backup iclr_results/t3-9-samples-${karg}all_results.json
  python test_solution.py -s 4000 -e 5000 --prefix t3-9-b5-samples-${karg} --save iclr_results --backup iclr_results/t3-9-samples-${karg}all_results.json
done


for karg in 'k128-' 'k256-' 'k512-' 'k1024-' ''
do
  python test_solution.py -s 4000 -e 5000 --prefix smc-samples-${karg} --save iclr_results --backup ../eval/results/all_results.json
  python test_solution.py -s 4000 -e 5000 --prefix smc-20-samples-${karg} --save iclr_results --backup ../eval/results/all_results.json
  python test_solution.py -s 4000 -e 5000 --prefix smc-50-samples-${karg} --save iclr_results --backup ../eval/results/all_results.json
  python test_solution.py -s 4000 -e 5000 --prefix smc-100-samples-${karg} --save iclr_results --backup ../eval/results/all_results.json
  python test_solution.py -s 4000 -e 5000 --prefix smc-200-samples-${karg} --save iclr_results --backup ../eval/results/all_results.json
done