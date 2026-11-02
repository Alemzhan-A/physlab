"""Генерация QR-кода для ссылки на приложение: python make_qr.py https://<имя>.streamlit.app"""
import sys

import segno

url = sys.argv[1] if len(sys.argv) > 1 else "https://physaudit-lab.streamlit.app"
qr = segno.make(url, error="h")
qr.save("qr_code.png", scale=20, border=2)  # ~ 700 px, для печати
qr.save("qr_code.svg", scale=10, border=2)  # вектор, любой размер
print(f"QR для {url} → qr_code.png, qr_code.svg")
