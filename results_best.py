with open("test_urbansound32k_fold0.log", "r") as f:
    text = f.read()

text_accs = text.split("{'acc':")[1:]
text_accs = [float(t.split('}')[0].strip()) for t in text_accs]

print(max(text_accs))