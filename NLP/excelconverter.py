import pandas as pd

text_file_path = "./NLP/val.txt"

with open(text_file_path, "r") as file:
    lines = file.readlines()

data = [line.strip().split(";") for line in lines]
df = pd.DataFrame(data, columns=["Text", "Emotion"])
df.to_csv("./NLP/val.csv", index=False)

# print("Data saved to emotions.csv")
