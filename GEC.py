from happytransformer import HappyTextToText
from datasets import load_dataset
import csv
import torch
import os
from happytransformer import TTTrainArgs



happy_tt = HappyTextToText("T5", "t5-base",from_tf=True)

train_dataset = load_dataset("jfleg", split='validation[:]')
eval_dataset = load_dataset("jfleg", split='test[:]')

def generate_csv(csv_path, dataset):
    with open(csv_path, 'w', newline='') as csvfile:
        writter = csv.writer(csvfile)
        writter.writerow(["input", "target"])
        for case in dataset:
            input_text = case["sentence"]
            for correction in case["corrections"]:
                # a few of the cases are blank strings. So we'll skip them
                if input_text and correction:
                    writter.writerow([input_text, correction])

generate_csv("train.csv", train_dataset)
generate_csv("eval.csv", eval_dataset)

args = TTTrainArgs(batch_size=8)
happy_tt.train("train.csv", args=args)
happy_tt.save('Model_gec')