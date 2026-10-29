# train_model.py
# Trains on the clips from collect_data.py with an HONEST test:
#   - frames are never split from their own clip (no leakage between neighbours)
#   - best option: hold out a whole session (another day / another person)
#
# Usage:
#     python train_model.py --test-session s3          (recommended)
#     python train_model.py                            (no test session: last 20% of clips per word)
#     python train_model.py --model sequence           (LSTM over whole clips, for moving signs)
#
# Saves: action_model_v2.keras  +  actions_v2.json (the word order, so it can never get mixed up)

import argparse
import json
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input
from keras.callbacks import EarlyStopping
from features import ACTIONS, SEQ_LEN, TOTAL_FEATURES, augment

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='dataset collect')
parser.add_argument('--model', choices=['frame', 'sequence'], default='sequence')#sequence= lstm whole clip , frame=frame by frame 
parser.add_argument('--test-session', default=None, help='session to use ONLY for testing, e.g. s3')
parser.add_argument('--epochs', type=int, default=80)
args = parser.parse_args()

N = len(ACTIONS)


def sort_key(name):
    session, num = name[:-4].rsplit('_', 1)
    return (session, int(num))


# ---------- load clips ----------
clips, labels, sessions = [], [], []
per_word = {i: [] for i in range(N)}          # word index -> clip indices in recording order
for wi, action in enumerate(ACTIONS):
    d = os.path.join(args.data, action)
    if not os.path.isdir(d):
        print(f"[WARN] missing folder for '{action}': {d}")
        continue
    for f in sorted([f for f in os.listdir(d) if f.endswith('.npz')], key=sort_key):
        clips.append(np.load(os.path.join(d, f))['features'])
        labels.append(wi)
        sessions.append(f.split('_')[0])
        per_word[wi].append(len(clips) - 1)
labels = np.array(labels)
print("Clips per word:", {ACTIONS[i]: len(v) for i, v in per_word.items()})
counts = [len(v) for v in per_word.values() if len(v) > 0]
if counts and max(counts) > 1.3 * min(counts):
    print("[WARN] words have unequal numbers of clips. Record the same number for every word.")

# ---------- split by CLIP / SESSION ----------
test_idx, pool_idx = [], []
for wi, idxs in per_word.items():
    if args.test_session:
        test_idx += [i for i in idxs if sessions[i] == args.test_session]
        pool_idx += [i for i in idxs if sessions[i] != args.test_session]
    else:
        n_test = max(1, int(round(len(idxs) * 0.2))) if idxs else 0
        test_idx += idxs[len(idxs) - n_test:]
        pool_idx += idxs[:len(idxs) - n_test]
if not test_idx:
    raise SystemExit("No test clips found. Check --test-session or record more data.")

train_idx, val_idx = [], []                      # every 6th clip of each word -> validation
for wi in range(N):
    mine = [i for i in pool_idx if labels[i] == wi]
    for k, i in enumerate(mine):
        (val_idx if k % 6 == 5 else train_idx).append(i)
print(f"Clips -> train: {len(train_idx)} | validation: {len(val_idx)} | test: {len(test_idx)}")
if not val_idx or not train_idx:
    raise SystemExit("Too few clips per word. Record at least ~20 clips per word per session first.")


def to_frames(idx_list):
    X, y, cid = [], [], []
    for i in idx_list:
        keep = clips[i].any(axis=1)                # drop frames with no hand (the app never predicts those)
        X.append(clips[i][keep])
        y += [labels[i]] * int(keep.sum())
        cid += [i] * int(keep.sum())
    return np.concatenate(X).astype(np.float32), np.array(y), np.array(cid)


def to_clips(idx_list):
    return np.stack([clips[i] for i in idx_list]).astype(np.float32), labels[idx_list]


if args.model == 'frame':
    X_tr, y_tr, _ = to_frames(train_idx)
    X_va, y_va, _ = to_frames(val_idx)
    X_te, y_te, cid_te = to_frames(test_idx)
else:
    X_tr, y_tr = to_clips(train_idx)
    X_va, y_va = to_clips(val_idx)
    X_te, y_te = to_clips(test_idx)

X_tr_aug = augment(X_tr, copies=3)
y_tr_aug = np.tile(y_tr, 4)                        # original + 3 copies

present = np.unique(y_tr_aug)
w = compute_class_weight('balanced', classes=present, y=y_tr_aug)
class_weight = {int(c): float(v) for c, v in zip(present, w)}

# ---------- model ----------
if args.model == 'frame':
    model = Sequential([
        Input(shape=(TOTAL_FEATURES,)),
        Dense(128, activation='relu'), Dropout(0.3),
        Dense(64, activation='relu'), Dropout(0.3),
        Dense(N, activation='softmax')])
else:
    model = Sequential([
        Input(shape=(SEQ_LEN, TOTAL_FEATURES)),
        LSTM(64, return_sequences=True), Dropout(0.3),
        LSTM(64), Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(N, activation='softmax')])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_tr_aug, to_categorical(y_tr_aug, N),
          validation_data=(X_va, to_categorical(y_va, N)),
          epochs=args.epochs, batch_size=64, class_weight=class_weight,
          callbacks=[EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)],
          verbose=2)

# ---------- honest test ----------
labels_list = list(range(N))
probs = model.predict(X_te, verbose=0)
if args.model == 'frame':
    print(f"\nFrame-level test accuracy: {(np.argmax(probs, 1) == y_te).mean() * 100:.1f}%")
    clip_ids = np.unique(cid_te)
    y_true = np.array([labels[c] for c in clip_ids])
    y_pred = np.array([np.argmax(probs[cid_te == c].mean(axis=0)) for c in clip_ids])
else:
    y_true, y_pred = y_te, np.argmax(probs, 1)

print(f"CLIP-level test accuracy: {(y_true == y_pred).mean() * 100:.1f}%   ({len(y_true)} test clips)")
print("\nConfusion matrix (rows = real word, columns = predicted). Order:", ACTIONS)
print(confusion_matrix(y_true, y_pred, labels=labels_list))
print(classification_report(y_true, y_pred, labels=labels_list, target_names=ACTIONS, zero_division=0))

model.save('action_model_v2.keras')
with open('actions_v2.json', 'w', encoding='utf-8') as f:
    json.dump({'actions': ACTIONS, 'model_type': args.model,
               'seq_len': SEQ_LEN, 'features': TOTAL_FEATURES}, f, ensure_ascii=False, indent=2)
print("\nSaved: action_model_v2.keras and actions_v2.json (your old action_model.keras is untouched)")