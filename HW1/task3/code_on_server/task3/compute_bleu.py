import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import json
with open('output_eval.json') as f:
    output = json.load(f)
ground_truths = []
smoothing_function = SmoothingFunction().method1
predicts = []
for i, item in enumerate(output):
    ground_truth = item['en_sentence']
    predict = item['output_sentence']
    predict.insert(0, ground_truth[0])
    ground_truths.append(ground_truth)
    predicts.append(predict)
bleu_score = corpus_bleu(ground_truths, predicts, weights=(1.0,0,0,0), smoothing_function=smoothing_function)
print(bleu_score)
    