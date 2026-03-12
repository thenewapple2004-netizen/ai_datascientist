with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(852, 934): # Indices 852 (Line 853) to 933 (Line 934)
    if lines[i].strip():
        lines[i] = "    " + lines[i]

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
