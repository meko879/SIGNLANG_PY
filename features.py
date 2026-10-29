# features.py
# ONE place that turns MediaPipe results into the numbers the model sees.
# Used by: collect_data.py, train_model.py, and (next step) the apps.
# If you change anything in this file you MUST re-collect data and re-train.
#
# IMPORTANT: the camera frame must be flipped (cv2.flip(frame, 1)) BEFORE it is
# given to MediaPipe, exactly like your current code. Then MediaPipe's
# "Left"/"Right" labels match your real left/right hand.

import numpy as np

# Words the system knows. The ORDER matters (it is also saved next to the model).
# "neutral" = hands visible but NOT signing (resting, adjusting, moving into position).
ACTIONS = ["hello", "mk", "my", "name is", "nice", "to meet", "you", "thank you","Goodbye"]

SEQ_LEN = 30                          # frames per clip
HAND_FEATURES = 66                    # per hand: 63 shape numbers + 3 wrist position
TOTAL_FEATURES = 2 * HAND_FEATURES    # left hand + right hand = 132
RAW_FEATURES = 126                    # raw x,y,z of 21 points x 2 hands (for the avatar later)


def assign_hands(results):
    """Return (left, right) landmark objects (or None) in a FIXED order,
    using MediaPipe's Left/Right label instead of detection order."""
    left = right = None
    if results.multi_hand_landmarks and results.multi_handedness:
        for lms, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handed.classification[0].label   # "Left" or "Right"
            if label == "Left" and left is None:
                left = lms
            elif label == "Right" and right is None:
                right = lms
            elif left is None:        # both hands got the same label: use the free slot
                left = lms
            elif right is None:
                right = lms
    return left, right


def _hand_vector(lms):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark], dtype=np.float32)  # 21 x 3
    wrist = pts[0].copy()
    rel = pts - wrist                       # hand SHAPE, relative to the wrist
    scale = np.linalg.norm(rel[9])          # wrist -> base of middle finger
    if scale > 1e-6:
        rel = rel / scale                   # same numbers whether the hand is near or far
    return np.concatenate([rel.flatten(), wrist]).astype(np.float32)   # 63 + 3 = 66


def extract_features(results):
    """132 numbers = [left hand 66 | right hand 66]. A missing hand = zeros."""
    left, right = assign_hands(results)
    zeros = np.zeros(HAND_FEATURES, dtype=np.float32)
    lv = _hand_vector(left) if left is not None else zeros
    rv = _hand_vector(right) if right is not None else zeros
    return np.concatenate([lv, rv]).astype(np.float32)


def extract_raw(results):
    """126 raw numbers (no normalisation), same left/right order. Used for the avatar."""
    left, right = assign_hands(results)
    out = []
    for lms in (left, right):
        if lms is None:
            out.extend([0.0] * 63)
        else:
            for lm in lms.landmark:
                out.extend([lm.x, lm.y, lm.z])
    return np.array(out, dtype=np.float32)


def augment(X, copies=3, seed=0):
    """Make extra training copies with small random changes so the model does not
    memorise exact positions. Works for frames (n, 132) and clips (n, T, 132).
    A hand that is absent (all zeros) is left untouched."""
    rng = np.random.default_rng(seed)
    out = [X]
    extra = (1,) * (X.ndim - 2)             # () for frames, (1,) for clips
    for _ in range(copies):
        Xa = X.copy()
        for h in range(2):
            s = h * HAND_FEATURES
            present = np.any(X[..., s:s + HAND_FEATURES] != 0, axis=-1, keepdims=True)
            shape_noise = rng.normal(0, 0.02, size=X[..., s:s + 63].shape)
            shift = rng.normal(0, 0.05, size=(X.shape[0],) + extra + (2,))  # same shift for a whole clip
            Xa[..., s:s + 63] += shape_noise * present
            Xa[..., s + 63:s + 65] += shift * present
        out.append(Xa.astype(np.float32))
    return np.concatenate(out)