# We only need to send requests to OpenAI server, no GPU is needed to run this script.
python synthesis_exp.py -s 4000 -e 4100 -l code-davinci-002 --save codex_results