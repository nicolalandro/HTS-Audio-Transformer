with open("logs/test_urbansound32k_fold4_from_scratch.log", "r") as f:
    text = f.read()

text_accs = text.split("{'acc':")[1:]
text_accs = [
    ( 
        int(t.split('}')[1].split('Epoch')[1].split(':')[0].strip()) , 
        float(t.split('}')[0].strip()) 
    )
    for t in text_accs
]

print(sorted(text_accs, key=lambda x: x[1])[-1])