# General Libraries
import numpy as np
import pyautogui
import time
import mss
import cv2
import easyocr
import pygetwindow as gw

# ====== NLTK word list initialization =====
wordlist_path = "popular.txt"

with open(wordlist_path, "r") as f:
    word_list = [line.strip() for line in f if line.strip()]
used_words = set()

print(word_list[:10])  # Print first 10 words for verification

# ====== Global variables =====
template_img = cv2.imread("turn_indicator.png")
is_typing = False
is_turn = False

# ===== Parameters =====
word_speed = 0.05  # seconds per letter

# ===== EasyOCR initialization =====
reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader once at startup

# ===== Helper functions =====
def imageProcessing(img, isolation):
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    if isolation:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img

def processSyllable(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rgb_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    results = reader.readtext(rgb_img, detail=0)
    return ''.join(results) if results else ''

def is_turn_active(screen_img, template, threshold=0.8):
    global is_turn
    screen_gray = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    print(f"Template match confidence: {max_val:.2f}")
    
    if (max_val >= threshold):
        is_turn = True
    else:
        is_turn = False

def enterWord(syllable):
    global used_words
    syllable = syllable.lower()
    
    if (not is_turn_active):
        return

    for w in word_list:
        lw = w.lower()
        if syllable in lw and lw not in used_words:
            print(f"Typing new unique word: {w}")
            pyautogui.typewrite(w, word_speed)
            pyautogui.press('enter')  # Optional: press Enter after typing
            used_words.add(lw)
            time.sleep(2)  # Wait for typing to finish
            return

    print(f"No unused words containing '{syllable}'")

# ===== Get Chrome window =====
chrome_windows = [w for w in gw.getAllWindows() if "chrome" in w.title.lower()]
if not chrome_windows:
    print("Chrome window not found!")
    exit(1)

chrome = chrome_windows[0]

chrome_left = chrome.left
chrome_top = chrome.top
chrome_width = chrome.width
chrome_height = chrome.height

print(f"Found Chrome window at ({chrome_left}, {chrome_top}), size {chrome_width}x{chrome_height}")
syllable_width = 75
syllable_height = 75

syllableBounds = {
    "left": chrome_left + int((chrome_width - syllable_width) / 2),
    "top": chrome_top + int((chrome_height - syllable_height) / 1.69),
    "width": syllable_width,
    "height": syllable_height
}

chromeBounds = {
    "left": chrome_left,
    "top": chrome_top,
    "width": chrome_width,
    "height": chrome_height
}

# ===== Main loop =====
with mss.mss() as sct:
    while True:

        # Capture syllable region
        syllableCapture = np.array(sct.grab(syllableBounds))

        # Display preview
        cv2.imshow("Syllable Capture", syllableCapture)

        # Check turn indicator
        chromeCapture = np.array(sct.grab(chromeBounds))
        is_turn_active(chromeCapture, template_img)

        if is_turn:
            # OCR processing
            ocr_result = processSyllable(syllableCapture)
            
            if (ocr_result):
                print(f"OCR result: {ocr_result}")
                print("Turn detected: entering word...")
                enterWord(ocr_result)

        key = cv2.waitKey(25) & 0xFF
        if key == ord("q"):
            print("Quitting.")
            break

cv2.destroyAllWindows()
