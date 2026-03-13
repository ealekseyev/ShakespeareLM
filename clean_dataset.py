import re

path = "kafka_dostoyevsky.txt"

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Collapse 3+ consecutive newlines down to two (one blank line)
text = re.sub(r"\n{2,}", "\n", text)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Done.")
