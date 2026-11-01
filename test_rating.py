"""
Тест парсера рейтингов smart-lab.ru для нескольких ISIN.
"""
import requests, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Список тестовых ISIN: 5 "плохих" которые вернули "Нет данных"
test_isins = [
    "RU000A0JT6B2",   # 1-й "плохой"
    "RU000A0JX5W4",   # 10-й 
    "RU000A1017J5",   # Хороший (проверенный)
    "RU000A0JQAM6",   # 1-й из списка
]

for isin in test_isins:
    url = f"https://smart-lab.ru/q/bonds/{isin}/"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        print(f"\n{'='*60}")
        print(f"ISIN: {isin} | HTTP {resp.status_code} | len={len(resp.text)}")
        
        # Ищем "облигация" в тексте
        if "облигаци" in resp.text.lower():
            print("→ Страница облигации найдена")
        else:
            print("→ СТРАНИЦА НЕ ОБЛИГАЦИЯ (редирект?)")
            # Проверяем первые 500 символов
            print(f"  Первые 500 символов: {resp.text[:500]}")
            continue
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, 'lxml')
        
        # Ищем прогресс-бар
        bar = soup.find('div', class_='linear-progress-bar')
        if bar:
            text = bar.get_text(strip=True)
            print(f"→ Прогресс-бар найден, текст: '{text}'")
            filled = bar.find('div', class_=lambda x: x and ('fill' in str(x).lower() or 'filed' in str(x).lower()))
            if filled:
                style = filled.get('style', '')
                m = re.search(r'width\s*:\s*([\d.]+)\s*%', style)
                print(f"→ Ширина: {m.group(1)}%" if m else f"→ Стиль: {style[:100]}")
        else:
            print("→ Прогресс-бар НЕ НАЙДЕН")
            # Ищем текст "Кредитный рейтинг"
            txt = soup.find(string=lambda t: t and 'рейтинг' in t.lower())
            if txt:
                print(f"→ Найдено 'рейтинг': {txt[:100]}")
            else:
                print("→ 'рейтинг' не найден на странице")
        
    except Exception as e:
        print(f"\nISIN: {isin} | ОШИБКА: {e}")