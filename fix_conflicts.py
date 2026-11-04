import re

path = r"g:\Мой диск\bond-portfolio-analyzer\app.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Удаляем все маркеры конфликта Git
content = re.sub(r"^<<<<<<< HEAD\s*\n?", "", content, flags=re.MULTILINE)
content = re.sub(r"^=======\s*\n?", "", content, flags=re.MULTILINE)
content = re.sub(r"^>>>>>>> .*\s*\n?", "", content, flags=re.MULTILINE)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Маркеры конфликта удалены из app.py")