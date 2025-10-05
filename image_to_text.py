# import easyocr
# import csv
# import os
# import numpy as np
# import pandas as pd
# import cv2
# import re


# def predict_number_plate(img, ocr):
#     result = ocr.ocr(img, cls=True)
#     result = result[0]
#     texts = [line[1][0] for line in result]
#     scores = [line[1][1] for line in result]
#     if (scores[0]*100) >= 90:
#         return re.sub(r'[^a-zA-Z0-9]', '', texts[0]), scores[0]
#     else:
#         return None, None






import easyocr
import csv
import os
import numpy as np
import pandas as pd
import cv2
import re

def predict_number_plate(img, ocr):
    """Extracts text from a number plate image using OCR."""
    result = ocr.ocr(img, cls=True)

    # Check if OCR returned valid results
    if not result or not isinstance(result, list) or len(result) == 0:
        print("OCR did not detect any text.")
        return None, None

    result = result[0]  # Get first detected text block
    if not result:  # Ensure it's not empty
        print("OCR result is empty.")
        return None, None

    texts = [line[1][0] for line in result if line and len(line) > 1]
    scores = [line[1][1] for line in result if line and len(line) > 1]

    if texts and scores and (scores[0] * 100) >= 90:
        return re.sub(r'[^a-zA-Z0-9]', '', texts[0]), scores[0]
    else:
        return None, None


