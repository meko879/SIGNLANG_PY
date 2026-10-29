# collect_data.py
# Records CLIPS (30 frames each) for ONE word, with an automatic countdown.
# Run it once per word per session:
#     python collect_data.py --word hello --session s1 --clips 25
#     python collect_data.py --word "name is" --session s1 --clips 25
#     python collect_data.py --word neutral --session s1 --clips 25
#
# Saves to dataset_v2/<word>/<session>_<number>.npz
# Safe to re-run: it continues after the last clip of that session (never overwrites).
#
# SESSION RULES (this is what makes the data good):
#   s1 = today.  s2 = another day / different light / different clothes.
#   s3 = ideally ANOTHER PERSON. Keep s3 only for testing (train_model.py --test-session s3).
#   Use the SAME number of clips for every word (including neutral).
#   Between clips, change your distance and hand position a little. Keep BOTH hands inside the view.

import argparse
import os
import time
import cv2
import numpy as np
import mediapipe as mp
from features import ACTIONS, SEQ_LEN, extract_features, extract_raw

parser = argparse.ArgumentParser()
parser.add_argument('--word', choices=ACTIONS)
parser.add_argument('--session')
parser.add_argument('--clips', type=int, default=25)
args = parser.parse_args()

if args.word is None:
    print("Which word do you want to record?")
    for i, a in enumerate(ACTIONS, 1):
        print(f"  {i}. {a}")
    while True:
        choice = input("Type the number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(ACTIONS):
            args.word = ACTIONS[int(choice) - 1]
            break
        print("Please type a number from the list.")
if args.session is None:
    args.session = input("Session name (s1, s2, s3 ...) [s1]: ").strip() or "s1"
assert '_' not in args.session, "session name must not contain '_'"
DATA_DIR = 'dataset collect'
MAX_MISSING = 0.4    # reject a clip if no hand is seen in more than 40% of its frames
COUNTDOWN = 2.0      # seconds to get ready before each clip
PAUSE = 1.0          # seconds of rest after each clip

out_dir = os.path.join(DATA_DIR, args.word)
os.makedirs(out_dir, exist_ok=True)

nums = []
for f in os.listdir(out_dir):
    if f.startswith(args.session + '_') and f.endswith('.npz'):
        try:
            nums.append(int(f[len(args.session) + 1:-4]))
        except ValueError:
            pass
saved = len(nums)
next_num = max(nums) + 1 if nums else 1

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)


def read():
    ok, frame = cap.read()
    if not ok:
        return None, None
    image = cv2.flip(frame, 1)                      # flip BEFORE MediaPipe (same as the apps)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hl, mp_hands.HAND_CONNECTIONS)
    return image, results


def put_text(image, lines, color=(255, 255, 255)):
    for i, line in enumerate(lines):
        cv2.putText(image, line, (20, 40 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)


def quit_all():
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nStopped. '{args.word}' now has {saved} clips in session {args.session}.")
    raise SystemExit


def show_for(seconds, lines, color):
    end = time.time() + seconds
    while time.time() < end:
        image, _ = read()
        if image is None:
            quit_all()
        put_text(image, lines, color)
        cv2.imshow('Collect', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            quit_all()


print(f"[INFO] word='{args.word}' session={args.session} | already saved: {saved} | target: {args.clips}")

# wait for SPACE
while True:
    image, _ = read()
    if image is None:
        quit_all()
    put_text(image, [f"WORD: {args.word.upper()}   session {args.session}",
                     f"clips: {saved}/{args.clips}",
                     "Vary distance/position a little between clips",
                     "Keep BOTH hands inside the view",
                     "SPACE = start   |   q = quit"], (0, 255, 255))
    cv2.imshow('Collect', image)
    key = cv2.waitKey(10) & 0xFF
    if key == 32:
        break
    if key == ord('q'):
        quit_all()

while saved < args.clips:
    show_for(COUNTDOWN, [f"GET READY: {args.word.upper()}", f"clip {saved + 1}/{args.clips}"], (0, 255, 255))

    feats, raws = [], []
    while len(feats) < SEQ_LEN:
        image, results = read()
        if image is None:
            quit_all()
        feats.append(extract_features(results))
        raws.append(extract_raw(results))
        put_text(image, [f"RECORDING: {args.word.upper()}",
                         f"clip {saved + 1}/{args.clips}   frame {len(feats)}/{SEQ_LEN}"], (0, 255, 0))
        cv2.imshow('Collect', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            quit_all()

    feats, raws = np.stack(feats), np.stack(raws)
    missing = float(np.mean([not f.any() for f in feats]))
    if missing > MAX_MISSING:
        show_for(1.5, ["REJECTED: hands not visible", "this clip is repeated"], (0, 0, 255))
        continue

    path = os.path.join(out_dir, f"{args.session}_{next_num:03d}.npz")
    np.savez_compressed(path, features=feats, raw=raws)
    saved += 1
    next_num += 1
    show_for(PAUSE, [f"saved {saved}/{args.clips}"], (255, 255, 255))

cap.release()
cv2.destroyAllWindows()
print(f"\nDone. '{args.word}' has {saved} clips in session {args.session}.")