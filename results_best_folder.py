from audioop import avg
import os

folder = "logs/urbansound_scratch"
best_accs = []
for f in os.listdir(folder):
    file_path = os.path.join(folder, f)
    with open(file_path, "r") as f:
        text = f.read()

    text_accs = text.split("{'acc':")[1:]
    text_accs = [
        ( 
            int(t.split('}')[1].split('Epoch')[1].split(':')[0].strip()) , 
            float(t.split('}')[0].strip()) 
        )
        for t in text_accs
    ]
    try:
        best_acc = sorted(text_accs, key=lambda x: x[1])[-1]
        best_accs.append(best_acc[1])
        print(file_path, best_acc)
    except Exception as e:
        print(file_path, e)
        

print('avg:', sum(best_accs)/len(best_accs))