# Chinese Font Guide for StatsPAI (中文字体指南)

## Quick Start

```python
import statspai as sp
sp.use_chinese()  # One-line fix, auto-detects the best Chinese font on your system
```

---

## 1. Cross-Platform Chinese Font Family

All platforms have "宋体" (Song/Serif CJK), but they are designed by different companies:

```
宋体风格 (Serif CJK)
├── Apple:    Songti SC / TC          macOS 预装
├── Microsoft: SimSun / NSimSun       Windows 预装
├── Google:   Noto Serif CJK SC      开源，全平台可安装
├── Adobe:    Source Han Serif        与 Noto 同源
└── SinoType: STSong (华文宋体)       macOS 预装，商业授权
```

### Platform Availability

| Font | macOS | Windows | Linux | License |
|------|:-----:|:-------:|:-----:|---------|
| **Songti SC** (宋体-简) | Pre-installed | - | - | Apple proprietary |
| **SimSun** (宋体) | - | Pre-installed | Manual install | Microsoft proprietary |
| **NSimSun** (新宋体) | - | Pre-installed | Manual install | Microsoft proprietary |
| **Noto Serif CJK SC** (思源宋体) | Installable | Installable | `apt install` | **Open source (OFL)** |
| **STSong** (华文宋体) | Pre-installed | With Office | - | SinoType proprietary |
| **AR PL UMing** | - | - | `apt install` | Open source (Arphic) |

### Visual Similarity to SimSun (微软宋体)

| Font | Similarity | Notes |
|------|:----------:|-------|
| **NSimSun** | ~99% | Monospace variant of SimSun, same designer |
| **STSong** | ~85% | Slightly thicker strokes, more rounded |
| **AR PL UMing** | ~75% | Open source imitation, rougher details |
| **Noto Serif CJK SC** | ~65% | Modern redesign, more elegant, better typography |
| **Songti SC** | ~60% | Apple design, thicker strokes |

### Key Differences: Songti SC vs SimSun vs Noto Serif CJK

| Feature | Songti SC (Apple) | SimSun (Microsoft) | Noto Serif CJK (Google) |
|---------|:-----------------:|:------------------:|:-----------------------:|
| Designer | Apple | Zhongyi (中易) | Adobe (西塚涼子) |
| Design era | 2000s | 1990s | 2017 |
| Strokes | Thicker, rounded | Thin, sharp | Balanced, modern |
| Character count | ~30,000 | ~28,000 | ~65,000+ |
| Font weights | 1 | 1 | **7** (ExtraLight to Black) |
| Screen optimization | Retina displays | Low-res bitmap hints | High-res screens |
| Best for | macOS screen | Windows/print legacy | Cross-platform, modern |

---

## 2. StatsPAI Chinese Font Usage

### Auto Mode (Recommended)

```python
import statspai as sp

sp.use_chinese()           # Auto-detect best available font
sp.use_chinese('serif')    # Prefer 宋体 (Songti SC / SimSun / Noto Serif CJK)
sp.use_chinese('sans')     # Prefer 黑体 (PingFang / SimHei / Noto Sans CJK)
sp.use_chinese('SimSun')   # Use a specific font
```

### Font Presets for Academic Publishing

```python
from statspai.plots.interactive import FigureEditor

editor = FigureEditor(fig=fig)

# Chinese thesis standard (中文学位论文)
# Auto-detects: Songti SC (Mac) / SimSun (Win) / Noto Serif CJK (Linux)
editor.apply_font_preset('Chinese Thesis (中文学位论文)')

# Chinese journal (中文期刊)
editor.apply_font_preset('Chinese Journal (中文期刊)')

# Chinese slides (中文PPT) — uses sans-serif (黑体/苹方)
editor.apply_font_preset('Chinese Slide (中文PPT)')
```

### What StatsPAI Does Automatically

When you select a Chinese font, StatsPAI:
1. Sets `font.family` to the correct family (serif/sans-serif)
2. Puts the Chinese font first in the preference list
3. Sets `axes.unicode_minus = False` (fixes the minus sign "−" display issue)
4. Generates reproducible code that includes all these settings

---

## 3. Installing Chinese Fonts

### macOS

Chinese fonts are **pre-installed**. No action needed:
- Songti SC (宋体), PingFang SC (苹方), Heiti TC (黑体), Kaiti SC (楷体), etc.

Just call `sp.use_chinese()`.

### Windows

SimSun (宋体) and SimHei (黑体) are **pre-installed**. No action needed.

Just call `sp.use_chinese()`.

### Linux (Ubuntu / Debian)

```bash
# Option 1: Open source Noto CJK fonts (RECOMMENDED)
sudo apt install fonts-noto-cjk fonts-noto-cjk-extra

# Option 2: WenQuanYi (文泉驿)
sudo apt install fonts-wqy-microhei fonts-wqy-zenhei

# Option 3: Arphic (文鼎)
sudo apt install fonts-arphic-uming fonts-arphic-ukai

# After installing, refresh font cache
fc-cache -fv

# Clear matplotlib cache (IMPORTANT!)
python3 -c "
import matplotlib, shutil, os
c = matplotlib.get_cachedir()
if os.path.exists(c): shutil.rmtree(c)
print(f'Cleared: {c}')
"
```

### Linux (Fedora / RHEL)

```bash
sudo dnf install google-noto-serif-cjk-sc-fonts google-noto-sans-cjk-sc-fonts
fc-cache -fv
```

### Linux (Arch)

```bash
sudo pacman -S noto-fonts-cjk
fc-cache -fv
```

---

## 4. Installing SimSun (微软宋体) on Linux

If your journal **strictly requires SimSun**, you need to install it manually.

### Method 1: Copy from Windows

```bash
# 1. From a Windows machine, copy: C:\Windows\Fonts\simsun.ttc (17MB)
# 2. Transfer to Linux server

# 3. Install to user fonts directory (no sudo needed)
mkdir -p ~/.local/share/fonts/
cp simsun.ttc ~/.local/share/fonts/

# 4. Refresh caches
fc-cache -fv

# 5. Clear matplotlib cache
python3 -c "
import matplotlib, shutil, os
c = matplotlib.get_cachedir()
if os.path.exists(c): shutil.rmtree(c)
print(f'Cleared: {c}')
"

# 6. Verify
fc-list | grep -i simsun
# Expected output: /home/user/.local/share/fonts/simsun.ttc: SimSun,宋体:style=Regular
```

### Method 2: Extract via Wine (no Windows machine needed)

```bash
sudo apt install winetricks
winetricks cjkfonts

# Copy the extracted font
cp ~/.wine/drive_c/windows/Fonts/simsun.ttc ~/.local/share/fonts/
fc-cache -fv
```

### Method 3: Download from online sources

Some open font archives host SimSun. Search for "simsun.ttc download" — but verify the license terms for your use case.

---

## 5. Deploying on Render.com (Linux Server)

Render servers have no `sudo` access. Three approaches:

### Approach A: Bundle font file in your repo (SimSun)

Project structure:
```
your-project/
├── fonts/
│   └── simsun.ttc          # Copy from Windows
├── render_build.sh
├── requirements.txt
└── app.py
```

`render_build.sh`:
```bash
#!/bin/bash
set -e

pip install -r requirements.txt

# Install fonts to user directory (no sudo needed)
mkdir -p ~/.local/share/fonts/
cp fonts/*.ttc ~/.local/share/fonts/ 2>/dev/null || true
cp fonts/*.ttf ~/.local/share/fonts/ 2>/dev/null || true
fc-cache -fv 2>/dev/null || true

# Clear matplotlib font cache (CRITICAL — matplotlib won't find new fonts without this)
python -c "
import matplotlib, shutil, os
c = matplotlib.get_cachedir()
if os.path.exists(c):
    shutil.rmtree(c)
    print(f'Cleared matplotlib cache: {c}')
"
```

Render Dashboard → Settings → **Build Command**: `bash render_build.sh`

### Approach B: Download Noto CJK at build time (no font file in repo)

`render_build.sh`:
```bash
#!/bin/bash
set -e

pip install -r requirements.txt

# Download open-source Noto Serif CJK (思源宋体)
mkdir -p ~/.local/share/fonts/
FONT_URL="https://github.com/notofonts/noto-cjk/releases/download/Serif2.003/08_NotoSerifCJKsc.zip"
curl -sL "$FONT_URL" -o /tmp/noto-serif.zip
unzip -o /tmp/noto-serif.zip -d /tmp/noto-serif/
cp /tmp/noto-serif/*.otf ~/.local/share/fonts/ 2>/dev/null || true

# Also download Noto Sans CJK (思源黑体) for sans-serif
SANS_URL="https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/09_NotoSansCJKsc.zip"
curl -sL "$SANS_URL" -o /tmp/noto-sans.zip
unzip -o /tmp/noto-sans.zip -d /tmp/noto-sans/
cp /tmp/noto-sans/*.otf ~/.local/share/fonts/ 2>/dev/null || true

fc-cache -fv 2>/dev/null || true

# Clear matplotlib cache
python -c "
import matplotlib, shutil, os
c = matplotlib.get_cachedir()
if os.path.exists(c):
    shutil.rmtree(c)
    print(f'Cleared matplotlib cache: {c}')
"

echo "Chinese fonts installed successfully"
```

### Approach C: Dockerfile (if using Docker on Render)

```dockerfile
FROM python:3.11-slim

# Install font utilities + open-source Chinese fonts
RUN apt-get update && \
    apt-get install -y fontconfig fonts-noto-cjk && \
    rm -rf /var/lib/apt/lists/* && \
    fc-cache -fv

# Or use bundled SimSun:
# COPY fonts/simsun.ttc /usr/share/fonts/truetype/
# RUN fc-cache -fv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clear matplotlib cache
RUN python -c "import matplotlib, shutil, os; c=matplotlib.get_cachedir(); shutil.rmtree(c, True)"

COPY . /app
WORKDIR /app
CMD ["python", "app.py"]
```

### After deployment, in your app code:

```python
import statspai as sp

# Auto-detect the installed Chinese font
font = sp.use_chinese()
print(f"Using Chinese font: {font}")

# Now all plots will render Chinese correctly
fig, ax = plt.subplots()
ax.set_title('因果推断效果分析')
ax.set_xlabel('相对处理时间（年）')
ax.set_ylabel('平均处理效应')
```

---

## 6. Troubleshooting

### Problem: Chinese text shows as boxes (□□□□)

```python
# Step 1: Check what fonts are available
import matplotlib.font_manager as fm
cn_fonts = [f.name for f in fm.fontManager.ttflist
            if any(kw in f.name for kw in ['Song', 'Hei', 'PingFang', 'Noto', 'Hiragino', 'SimSun'])]
print(f"Chinese fonts found: {cn_fonts}")

# Step 2: If empty, install fonts (see Section 3)

# Step 3: If fonts are installed but still not working, clear cache
import matplotlib, shutil, os
cache = matplotlib.get_cachedir()
shutil.rmtree(cache, ignore_errors=True)
print(f"Cleared cache: {cache}")
# Then restart Python/Jupyter kernel
```

### Problem: Minus sign shows as "−" (square) with Chinese fonts

```python
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False  # Use hyphen-minus instead
# sp.use_chinese() does this automatically
```

### Problem: Font installed but matplotlib doesn't find it

```bash
# Rebuild matplotlib font cache
python3 -c "
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)
print('Font manager rebuilt')
print(f'Total fonts: {len(fm.fontManager.ttflist)}')
"
```

---

## 7. Recommendation Summary

| Scenario | Recommended Font | Why |
|----------|-----------------|-----|
| Personal use on Mac | `sp.use_chinese()` (auto) | Songti SC pre-installed |
| Personal use on Windows | `sp.use_chinese()` (auto) | SimSun pre-installed |
| Linux desktop | `Noto Serif CJK SC` | Best open-source quality |
| Cloud/server deployment | `Noto Serif CJK SC` | Open source, no license issues |
| Journal requires "宋体" | Install `SimSun` | Copy from Windows |
| Cross-platform reproducible | `Noto Serif CJK SC` | Same font everywhere |
| Presentations/slides | `sp.use_chinese('sans')` | 黑体/苹方 more readable |
