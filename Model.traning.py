from transformers import pipeline
sarcasm_pipe=pipeline('text-classification',model='finitautomata/bertweet-base-sarcasm-detectiom')

