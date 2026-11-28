#OCR using tesseract OCR on CPU


!apt-get -qq install -y tesseract-ocr    #for google colab env
!pip install -q pytesseract

from google.colab import files
uploaded = files.upload()
img_path = list(uploaded.keys())[0]
print("Using:", img_path)

import os, json
import cv2
import pandas as pd
import pytesseract
from pytesseract import Output
from IPython.display import Image, display

# ---- Run OCR ----
image = cv2.imread(img_path)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

data = pytesseract.image_to_data(
    rgb,
    lang="eng",
    config="--oem 3 --psm 3",   # psm 3 = fully automatic page segmentation
    output_type=Output.DICT,
)

# ---- Word-level table, dropping empty/non-text entries ----
words = pd.DataFrame(data)
words = words[(words["conf"].astype(float) >= 0) & (words["text"].str.strip() != "")]

# ---- Group words into lines (same granularity as PaddleOCR) ----
rows = []
for _, g in words.groupby(["block_num", "par_num", "line_num"], sort=False):
    rows.append({
        "page": 1,
        "text": " ".join(g["text"]),
        "confidence": round(g["conf"].astype(float).mean() / 100, 3),  # 0-1 scale like Paddle
        "x1": int(g["left"].min()),
        "y1": int(g["top"].min()),
        "x2": int((g["left"] + g["width"]).max()),
        "y2": int((g["top"] + g["height"]).max()),
    })

df = pd.DataFrame(rows)

# ---- Print text with confidence ----
for _, r in df.iterrows():
    print(f"{r['text']}  ({r['confidence']:.2f})")

# ---- Annotated image + JSON ----
os.makedirs("output", exist_ok=True)
annotated = image.copy()
for _, r in df.iterrows():
    cv2.rectangle(annotated, (r["x1"], r["y1"]), (r["x2"], r["y2"]), (0, 200, 0), 2)
cv2.imwrite("output/annotated.png", annotated)
df.to_json("output/ocr_raw.json", orient="records", indent=2)
display(Image("output/annotated.png"))

# ---- Full text as one string ----
full_text = "\n".join(df["text"])
print(full_text)

# ---- DataFrame with positions and confidence ----
df

# ---- Filter low-confidence results ----
df_clean = df[df["confidence"] >= 0.80]
print("\n".join(df_clean["text"]))

# ---- Save and download ----
with open("ocr_text.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

df.to_csv("ocr_results.csv", index=False)

files.download("ocr_text.txt")
files.download("ocr_results.csv")
