# 🤖 Telegram Bot Setup Guide

## Quick Start (Български)

### Стъпка 1: Инсталирай зависимостите

```bash
pip install -r requirements.txt
```

Това ще инсталира:
- Flask (web framework)
- python-telegram-bot (Telegram API)
- Pillow (image processing)
- requests (HTTP client)

### Стъпка 2: Създай Telegram Bot

1. Отвори Telegram и намери **@BotFather**
2. Изпрати `/newbot`
3. Въведи име за бота (напр. "My Gallery Bot")
4. Въведи username (трябва да завършва на "bot", напр. "mygallery_bot")
5. **Копирай bot token** - изглежда така: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`

### Стъпка 3: Тествай дали всичко е OK

Стартирай диагностичния скрипт:

```bash
python test_bot.py
```

Това ще провери:
- ✓ Дали са инсталирани всички библиотеки
- ✓ Дали директориите съществуват
- ✓ Дали emoji работят на твоята система
- ✓ Дали telegram_bot.py може да се импортира

### Стъпка 4: Стартирай Flask приложението

```bash
python app.py
```

Трябва да видиш:
```
 * Running on http://127.0.0.1:5000
```

### Стъпка 5: Конфигурирай бота от Web UI

1. Отвори браузър: `http://localhost:5000`
2. Кликни на **⚙️ Settings** (горе вдясно)
3. В секцията "📱 Telegram Bot":
   - Залепи bot token-а от BotFather
   - Избери дали искаш auto-analyze
   - Избери AI style (Classic, Detailed, Tags)
4. Кликни **💾 Save Configuration**

### Стъпка 6: Стартирай бота

1. Кликни **▶️ Start** бутона
2. Изчакай 2-3 секунди
3. Трябва да видиш: **🟢 Bot Running (PID: XXXX)**

### Стъпка 7: Провери логовете

1. Кликни **📄 View Logs**
2. Трябва да видиш:
```
[2025-11-15 XX:XX:XX] [STDOUT] Starting Telegram Gallery Bot...
[2025-11-15 XX:XX:XX] [STDOUT] 🤖 Telegram Gallery Bot is running...
[2025-11-15 XX:XX:XX] [STDOUT] 📁 Photos directory: photos
```

### Стъпка 8: Тествай в Telegram

1. Намери своя бот в Telegram (username от стъпка 2)
2. Изпрати `/start` → трябва да получиш welcome съобщение
3. Изпрати снимка
4. Ботът трябва да отговори:
   - "✅ Photo saved to gallery!"
   - "🤖 Analyzing with AI..." (ако е включено)
   - "✨ Analysis Complete!" (с description и tags)

### Стъпка 9: Провери в галерията

1. Върни се в браузъра
2. Кликни "All Images" в sidebar-а
3. Трябва да видиш снимката от Telegram!

---

## Troubleshooting (Решения на проблеми)

### ❌ "ModuleNotFoundError: No module named 'telegram'"

**Решение:**
```bash
pip install python-telegram-bot==20.7
```

### ❌ "Bot exited immediately"

**Причини:**
1. Невалиден bot token
2. Липсващи библиотеки

**Решение:**
1. Кликни "📄 View Logs" в Settings
2. Потърси грешки в логовете
3. Провери дали token-ът е правилен (без интервали, цялата стойност)

### ❌ "UnicodeEncodeError" на Windows

**Решение:** Това вече е оправено в кода! Просто рестартирай app.py.

### ⚠️ Bot статусът е "🟡 Bot Configured (Offline)"

**Причина:** Ботът е конфигуриран но не е стартиран

**Решение:** Кликни "▶️ Start"

### ⚠️ Bot статусът е "🔴 Bot Not Configured"

**Причина:** Липсва bot token

**Решение:** Добави bot token в Settings и кликни Save

### ⚠️ Ботът не получава снимки

**Причини:**
1. Не си изпратил `/start` команда на бота
2. Ботът не е стартиран
3. Изпращаш в group където ботът няма permissions

**Решение:**
1. Изпрати `/start` на бота
2. Провери Status: трябва да е 🟢 Bot Running
3. Ако е в group, направи бота admin или използвай Privacy Mode OFF

### ⚠️ AI анализът не работи

**Причина:** LM Studio не е стартирано или не е на правилния адрес

**Решение:**
1. Стартирай LM Studio
2. Провери дали е на `http://localhost:1234`
3. Провери AI Status в header-а: трябва да е "🟢 AI Connected"

---

## Advanced Configuration (Напреднали настройки)

### Използване на .env файл

Вместо да конфигурираш от UI, можеш да създадеш `.env` файл:

```bash
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
AUTO_ANALYZE=true
AI_STYLE=classic

# Gallery Settings
PHOTOS_DIR=photos
DATA_DIR=data
GALLERY_API_URL=http://localhost:5000

# LM Studio
LM_STUDIO_URL=http://localhost:1234

# Optional: Restrict to specific chats (comma-separated chat IDs)
# ALLOWED_CHATS=123456,789012
```

### Стартиране в background (Linux/Mac)

```bash
nohup python app.py > app.log 2>&1 &
```

### Стартиране с systemd (Linux)

Създай `/etc/systemd/system/ai-gallery.service`:

```ini
[Unit]
Description=AI Gallery Web App
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/ai-gallery1986
Environment="PATH=/usr/bin:/usr/local/bin"
ExecStart=/usr/bin/python3 app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

После:
```bash
sudo systemctl enable ai-gallery
sudo systemctl start ai-gallery
sudo systemctl status ai-gallery
```

---

## Bot Commands (Команди на бота)

След като бота е стартиран, можеш да изпращаш тези команди в Telegram:

- `/start` - Welcome съобщение и информация
- `/status` - Статистики за галерията
- `/help` - Помощ и списък с функции

---

## Security Tips (Съвети за сигурност)

1. **Никога не споделяй bot token-а публично!**
2. Не commit-вай `.env` файла в git (вече е в `.gitignore`)
3. Ако компрометираш token-а, ревокирай го от @BotFather с `/revoke`
4. За production, използвай ALLOWED_CHATS за да ограничиш достъпа

---

## Performance Tips

- За повече от 1000 снимки, разгледай използването на Redis за кеширане
- За production deployment, използвай gunicorn вместо Flask development server
- Размести Telegram bot и Flask app на отделни процеси за по-добра стабилност

---

Ако имаш проблеми, провери:
1. `python test_bot.py` - диагностика
2. Settings > View Logs - bot логове
3. Flask console output - app логове
