import os
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.signal import butter, filtfilt, welch, find_peaks, detrend, resample
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 유틸: 신호처리 필터
def butter_bandpass(low, high, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return b, a

def bandpass_filter(x, fs, low, high, order=3):
    if len(x) < max(16, 3*order):
        return x
    b, a = butter_bandpass(low, high, fs, order=order)
    return filtfilt(b, a, x)

def softmax(x, tau=1.0):
    z = (x - np.max(x)) / max(1e-8, tau)
    e = np.exp(z)
    return e / np.sum(e)


# 0단계: 비디오 로딩 & 얼굴 검출
def read_video(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fs = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        n += 1
        if max_frames and n >= max_frames: break
    cap.release()
    return np.array(frames), float(fs)

def detect_face_bbox(frame_rgb):
    # 간단한 Haar cascade 기반 얼굴 탐지
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        # 실패 시 프레임 중앙 60% 영역을 얼굴로 가정 (강건성 확보)
        h, w = gray.shape
        return int(0.2*w), int(0.2*h), int(0.6*w), int(0.6*h)
    # 가장 큰 얼굴 선택
    x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
    return int(x), int(y), int(w), int(h)

# 1단계: GMM 기반 피부 확률 + 상위 32 ROI + CHROM rPPG
def skin_probability_gmm(face_rgb, n_components=2):
    # YCrCb에서 Cr, Cb만 사용해 GMM 학습
    ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
    Cr = ycrcb[...,1].reshape(-1,1).astype(np.float32)
    Cb = ycrcb[...,2].reshape(-1,1).astype(np.float32)
    X = np.hstack([Cr, Cb])

    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gmm.fit(X)

    # 피부 색 대략적 중심(문헌상): Cr~150, Cb~120 근방
    skin_center = np.array([150, 120], dtype=np.float32)
    means = gmm.means_
    skin_idx = np.argmin(np.linalg.norm(means - skin_center, axis=1))

    # 피부 성분의 posterior ^ (정규화)
    posterior = gmm.predict_proba(X)[:, skin_idx]
    prob_map = posterior.reshape(face_rgb.shape[:2])
    return prob_map

def select_top_rois(prob_map, roi_size=(16,16), top_k=32):
    H, W = prob_map.shape
    rh, rw = roi_size
    rois = []
    for y in range(0, max(1, H-rh+1), rh):
        for x in range(0, max(1, W-rw+1), rw):
            p = prob_map[y:y+rh, x:x+rw].mean()
            rois.append((p, x, y, rw, rh))
    rois.sort(key=lambda t: t[0], reverse=True)
    return rois[:top_k]

def extract_chrom_signal(frames_rgb, roi):
    # CHROM: X = 3R - 2G, Y = 1.5R + G - 1.5B, alpha = std(X)/std(Y), S = X - alpha*Y
    x, y, w, h = roi
    R = frames_rgb[:, y:y+h, x:x+w, 0].mean(axis=(1,2))
    G = frames_rgb[:, y:y+h, x:x+w, 1].mean(axis=(1,2))
    B = frames_rgb[:, y:y+h, x:x+w, 2].mean(axis=(1,2))
    X = 3*R - 2*G
    Y = 1.5*R + G - 1.5*B
    sX, sY = np.std(X)+1e-8, np.std(Y)+1e-8
    alpha = sX / sY
    S = X - alpha * Y
    # 기본 detrend (선형)
    S = detrend(S, type='linear')
    return S

def step1_rppg(frames_rgb, fs, top_k=32, roi_size=(16,16)):
    # 1) 첫 프레임에서 얼굴 검출 → 얼굴 박스 고정 (간단화)
    x, y, w, h = detect_face_bbox(frames_rgb[0])
    face_crops = frames_rgb[:, y:y+h, x:x+w, :]

    # 2) 첫 프레임 얼굴에서 GMM 피부 확률맵
    prob_map = skin_probability_gmm(frames_rgb[0, y:y+h, x:x+w, :])

    # 3) 상위 ROI 선택
    top_rois = select_top_rois(prob_map, roi_size=roi_size, top_k=top_k)

    # 4) 각 ROI에서 CHROM rPPG 추출
    roi_signals = []
    for _, rx, ry, rw, rh in top_rois:
        sig = extract_chrom_signal(face_crops, (rx, ry, rw, rh))
        roi_signals.append(sig)
    roi_signals = np.array(roi_signals)  # [K, T]
    return roi_signals, (x,y,w,h), prob_map, top_rois


# 2단계: 잡음 제거 & SNR 가중 평균
def compute_snr(sig, fs, hr_band=(0.7, 3.0), noise_band=(0.2, 0.6)):
    # Welch PSD에서 HR 대역 피크 전력 / 노이즈대역 평균전력
    f, Pxx = welch(sig, fs=fs, nperseg=min(256, len(sig)))
    def band_power(fmin, fmax):
        idx = (f >= fmin) & (f <= fmax)
        if not np.any(idx): return 1e-8
        return np.max(Pxx[idx])  # HR 대역은 피크 전력
    def band_mean(fmin, fmax):
        idx = (f >= fmin) & (f <= fmax)
        if not np.any(idx): return 1e-8
        return np.mean(Pxx[idx])
    signal_p = band_power(*hr_band)
    noise_p = band_mean(*noise_band)
    snr = 10*np.log10(signal_p / (noise_p + 1e-8))
    return snr

def step2_denoise_and_fuse(roi_signals, fs):
    # HR 대역 대역통과
    roi_bp = np.array([bandpass_filter(s, fs, 0.7, 3.0, order=3) for s in roi_signals])

    # 각 ROI SNR 계산
    snrs = np.array([compute_snr(s, fs) for s in roi_bp])
    weights = softmax(snrs, tau=1.0)  # 합=1

    # 가중 평균으로 융합
    fused = np.average(roi_bp, axis=0, weights=weights)
    return fused, snrs, weights

# -----------------------------
# 3단계: IBI 추출 (피크 간 간격)
# -----------------------------
def step3_ibi_from_peaks(rppg_fused, fs, hr_min=40, hr_max=180):
    # 피크 간 최소 거리 제약 (최소 HR 기준)
    min_dist = int(fs * 60.0 / hr_max)  # 샘플
    peaks, _ = find_peaks(rppg_fused, distance=max(1, min_dist), prominence=np.std(rppg_fused)*0.2)
    if len(peaks) < 2:
        return np.array([]), np.array([])  # beat_times, IBI_ms
    beat_times = peaks / fs  # sec
    ibi = np.diff(beat_times) * 1000.0  # ms
    return beat_times, ibi

# -----------------------------
# 4단계: 후처리 (선형 추세 제거, 균일 보간 등)
# -----------------------------
def step4_postprocess_ibi_to_resp(beat_times, ibi_ms, target_len, target_fs):
    if len(ibi_ms) == 0:
        return np.zeros(target_len)

    # IBI는 박동 중간시간에 정의 (두 피크 중앙 시각)
    mid_times = (beat_times[:-1] + beat_times[1:]) / 2.0  # sec

    # 균일 시계열로 보간(목표: 비디오 프레임 타임스탬프)
    t_uniform = np.arange(target_len) / target_fs
    # 경계 보간을 위해 양 끝 extrapolate 허용
    f = interp1d(mid_times, ibi_ms, kind='linear', fill_value='extrapolate', assume_sorted=True)
    ibi_uniform = f(t_uniform)

    # 저주파 추세 제거(선형 detrend)
    resp_like = detrend(ibi_uniform, type='linear')

    # 호흡 대역(0.1–0.4Hz)로 부드럽게 (선택)
    resp_filt = bandpass_filter(resp_like, target_fs, 0.08, 0.5, order=2)
    return resp_filt

# -----------------------------
# 5단계: 파이프라인 메인
# -----------------------------
def process_video_to_resp_signal(video_path, max_frames=None, visualize=True):
    frames_rgb, fs = read_video(video_path, max_frames=max_frames)

    # 1단계: rPPG 추출 (GMM 피부 + 상위 32 ROI + CHROM)
    roi_signals, face_bbox, prob_map, top_rois = step1_rppg(frames_rgb, fs, top_k=32, roi_size=(16,16))

    # 2단계: 잡음 제거 & SNR 가중 평균
    rppg_fused, snrs, weights = step2_denoise_and_fuse(roi_signals, fs)

    # 3단계: IBI 추출
    beat_times, ibi_ms = step3_ibi_from_peaks(rppg_fused, fs)

    # 4단계: 후처리(보간 + 선형 detrend)
    resp_signal = step4_postprocess_ibi_to_resp(
        beat_times, ibi_ms, target_len=len(frames_rgb), target_fs=fs
    )

    out = {
        "fps": fs,
        "frames": frames_rgb.shape[0],
        "face_bbox": face_bbox,                # (x,y,w,h)
        "skin_prob_map": prob_map,             # 첫 프레임 얼굴에서의 피부 확률
        "top_rois": top_rois,                  # 선택된 ROI 목록
        "roi_signals": roi_signals,            # [K, T]
        "rppg_fused": rppg_fused,              # [T]
        "snrs": snrs,                          # [K]
        "weights": weights,                    # [K], 합=1
        "beat_times_sec": beat_times,          # [N]
        "ibi_ms": ibi_ms,                      # [N-1]
        "resp_signal": resp_signal,            # [T] 최종 호흡 신호(정규화X)
    }

    if visualize:
        T = np.arange(len(rppg_fused))/fs
        fig, axs = plt.subplots(4,1, figsize=(12,10), sharex=False)
        axs[0].plot(T, rppg_fused)
        axs[0].set_title("2단계 결과: 융합된 rPPG (0.7–3.0Hz)")
        axs[0].set_xlabel("Time (s)")

        if len(ibi_ms) > 0:
            axs[1].plot(beat_times[1:], ibi_ms, marker='o')
            axs[1].set_title("3단계: IBI (ms)")
            axs[1].set_xlabel("Time (s)")
            axs[1].set_ylabel("IBI (ms)")
        else:
            axs[1].text(0.5, 0.5, "IBI 추출 실패(피크 부족)", ha='center', va='center')
            axs[1].set_axis_off()

        axs[2].plot(T, detrend(rppg_fused, type='linear'))
        axs[2].set_title("참고: rPPG 선형 detrend")
        axs[2].set_xlabel("Time (s)")

        axs[3].plot(T, resp_signal)
        axs[3].set_title("5단계: 최종 호흡 신호 (IBI 기반 RSA)")
        axs[3].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

    return out

# -----------------------------
# 실행 예시
# -----------------------------
if __name__ == "__main__":
    VIDEO_PATH = "/home/subi/PycharmProjects/data.avi"  # COHFACE의 RGB 비디오 경로
    result = process_video_to_resp_signal(VIDEO_PATH, max_frames=None, visualize=True)

    # 간단 로그
    print(f"fps={result['fps']:.2f}, frames={result['frames']}")
    print(f"IBI 개수: {len(result['ibi_ms'])}, 평균 IBI: {np.mean(result['ibi_ms']) if len(result['ibi_ms']) else np.nan:.1f} ms")
    print(f"ROI SNR 평균: {np.mean(result['snrs']):.2f} dB")