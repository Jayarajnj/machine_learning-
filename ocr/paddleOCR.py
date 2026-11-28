#OCR using paddle OCR on CPU


!pip install -q paddlepaddle paddleocr

from google.colab import files
uploaded = files.upload()
img_path = list(uploaded.keys())[0]
print("Using:", img_path)

from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    enable_mkldnn=False,   # avoids the oneDNN PIR error
)

result = ocr.predict(img_path)

for res in result:
    for text, score in zip(res["rec_texts"], res["rec_scores"]):
        print(f"{text}  ({score:.2f})")

import glob
from IPython.display import Image, display

for res in result:
    res.save_to_img("output")    # image with boxes and recognized text
    res.save_to_json("output")   # raw results as JSON

for f in sorted(glob.glob("output/*.png")) + sorted(glob.glob("output/*.jpg")):
    display(Image(f))
    

full_text = "\n".join(t for res in result for t in res["rec_texts"])
print(full_text)


import pandas as pd

rows = []
for page, res in enumerate(result, start=1):
    for text, score, box in zip(res["rec_texts"], res["rec_scores"], res["rec_boxes"]):
        x1, y1, x2, y2 = [int(v) for v in box]
        rows.append({"page": page, "text": text, "confidence": round(float(score), 3),
                     "x1": x1, "y1": y1, "x2": x2, "y2": y2})

df = pd.DataFrame(rows)
df


df_clean = df[df["confidence"] >= 0.80]
print("\n".join(df_clean["text"]))


from google.colab import files

with open("ocr_text.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

df.to_csv("ocr_results.csv", index=False)

files.download("ocr_text.txt")
files.download("ocr_results.csv")
