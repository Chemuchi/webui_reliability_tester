# Dataset 기반 신뢰성 평가 도구 (Reliability Test WebUI)

YOLO 모델의 디텍션 성능 및 신뢰성 항목(mAP, FPR, Safety 등)을 평가하는 Streamlit 기반 웹 UI입니다.  
영상(`.mp4`) 1개와 라벨(`.json`) 1개를 업로드하면, 내부적으로 노트북(`reliability_test.ipynb`)을 실행하여 결과 리포트와 함께 `reports-날짜-시간.zip` 파일을 다운로드받을 수 있습니다.

---

## 📋 평가 시나리오

| 시나리오 | 설명 | Safety | XAI | Robust | Transparency |
|---|---|---|---|---|---|
| TC0 | Baseline (원본) | △ | − | △ | ● |
| TC1 | 저조도 (조명 L2) | ● | − | ● | ● |
| TC2 | 극저조도 + 노이즈 (L3) | ● | △ | ● | ● |
| TC3 | 우천 합성 (W2) | ● | △ | ● | ● |
| TC4 | 안개 합성 (W3) | ● | △ | ● | ● |
| TC5 | 저해상도 (S3) | △ | − | ● | ● |
| TC6 | 복합 (L3+W3+S3) | ● | △ | ● | ● |

---

## 🚀 실행 방법

### 🍎 macOS (Apple Silicon / Intel) — 가상환경(venv) 방식 권장

Mac에서는 Apple GPU(MPS)를 네이티브로 활용할 수 있어, Docker보다 **venv 방식이 더 빠릅니다.**

```bash
# 1. 저장소 클론
git clone <repo-url>
cd webui_reliability_tester

# 2. Python 가상환경 생성 (Python 3.10 이상 권장)
python3 -m venv .venv
source .venv/bin/activate

# 3. 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt

# 4. Streamlit 앱 실행
streamlit run webui_streamlit_app.py
```

브라우저에서 자동으로 `http://localhost:8501` 이 열립니다.

> **Tip**: 빠른 응답을 위해 watchdog 설치를 권장합니다.
> ```bash
> pip install watchdog
> ```

---

### 🪟 Windows (NVIDIA GPU) — Docker 방식 권장

Windows에서는 NVIDIA GPU를 Docker 컨테이너 안에서 그대로 사용할 수 있습니다.

#### 사전 준비

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) 설치 (WSL2 백엔드 활성화)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 설치

#### 실행

```bash
# 1. 저장소 클론
git clone <repo-url>
cd webui_reliability_tester

# 2. Docker 이미지 빌드
docker build -t reliability-webui .

# 3. GPU와 공유 메모리를 지정하여 컨테이너 실행
docker run --gpus all --shm-size=8g -p 8501:8501 reliability-webui
```

브라우저에서 `http://localhost:8501` 에 접속합니다.

> **⚠️ `--shm-size=8g` 필수**: Docker 컨테이너의 기본 공유 메모리(shm)는 64MB로, PyTorch DataLoader가 멀티워커로 텐서를 공유할 때 부족합니다. 이 옵션 없이 실행하면 `RuntimeError: unable to allocate shared memory` 오류가 발생합니다.

---

## 📖 사용법

### 1단계 — 시나리오 선택

웹 UI 상단의 체크리스트에서 평가할 시나리오를 선택합니다.  
기본값은 TC0~TC6 **전체**이며, 특정 시나리오만 선택하면 해당 항목만 평가됩니다.

### 2단계 — 파일 업로드

| 구분 | 형식 | 설명 |
|---|---|---|
| 영상 파일 | `.mp4` | 평가 대상 영상 (1개) |
| 라벨 파일 | `.json` | 해당 영상의 정답 라벨 (1개) |

> 영상과 라벨 파일의 **파일명(stem)이 일치**하는 것을 권장합니다.

### 3단계 — 평가 실행

`[평가 실행]` 버튼을 클릭합니다.

- 실행 중에는 버튼이 비활성화되며 **실시간 로그**와 **진행률 바**가 표시됩니다.
- `[🚫 평가 중단 (Stop)]` 버튼으로 언제든 평가를 중단할 수 있습니다.

### 4단계 — 결과 다운로드

평가가 완료되면 결과물 압축 파일(`reports-YYYYMMDD-HHMMSS.zip`)을 다운로드할 수 있습니다.

#### 압축 파일 내 주요 항목

```
reports/
├── report.md                          # 전체 평가 요약 리포트
├── scenario_reliability_mapping.csv   # 시나리오별 신뢰성 매핑표
├── summary_df.csv                     # 밝기/엣지/블러 등 이미지 품질 지표
├── safety_df.csv                      # TP/TN/FP/FN 기반 Safety 평가
├── eval_df.csv                        # mAP50, mAP50-95, Precision, Recall
└── runs/                              # YOLO 추론 결과 이미지
```

---

## 🗂️ 프로젝트 구조

```
webui_reliability_tester/
├── webui_streamlit_app.py    # Streamlit WebUI 메인 앱
├── notebook_runner.py        # 노트북 실행 및 로그 스트리밍 모듈
├── reliability_test.ipynb    # 신뢰성 평가 로직 노트북
├── yolo26n.pt                # 평가에 사용할 YOLO 모델 가중치
├── requirements.txt          # Python 의존성 목록
└── README.md
```

---

## ⚙️ 의존성

| 패키지 | 용도 |
|---|---|
| `streamlit >= 1.33` | WebUI 프레임워크 (`@st.fragment` 지원 버전) |
| `ultralytics >= 8.1` | YOLO 모델 추론 |
| `torch` | PyTorch (MPS/CUDA 자동 감지) |
| `nbformat`, `nbclient` | Jupyter 노트북 파싱 |
| `opencv-python-headless` | 영상 프레임 처리 |
| `pandas`, `numpy` | 데이터 처리 |
| `tqdm` | 진행률 표시 |
