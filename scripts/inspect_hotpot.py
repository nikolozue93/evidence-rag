from datasets import load_dataset

dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", streaming=True,)

example = next(iter(dataset))

print("Question:", example["question"])
print("Answer:", example["answer"])
print("Question type:", example["type"])
print("Context fields:", example["context"].keys())

for title, sentences in zip(example["context"]["title"], example["context"]["sentences"],):
    print("\nDocument:", title)
    print("Text:", " ".join(sentences)[:300])