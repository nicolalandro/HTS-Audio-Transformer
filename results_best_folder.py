from audioop import avg
import os

folder = "logs"
best_accs = []
for f in os.listdir(folder):
    with open(os.path.join(folder, f), "r") as f:
        text = f.read()

    text_accs = text.split("{'acc':")[1:]
    text_accs = [
        ( 
            int(t.split('}')[1].split('Epoch')[1].split(':')[0].strip()) , 
            float(t.split('}')[0].strip()) 
        )
        for t in text_accs
    ]

    best_acc = sorted(text_accs, key=lambda x: x[1])[-1]
    best_accs.append(best_acc[1])
    print(best_acc)

print('avg:', sum(best_accs)/len(best_accs))