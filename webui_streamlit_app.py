
import streamlit as st
import tempfile
import re
import html as _html
from pathlib import Path

from notebook_runner import run_reliability_notebook, zip_dir

st.set_page_config(page_title="Dataset Reliability Evaluator", layout="wide")

st.title("Dataset 기반 평가 도구 (Reliability Test)")

st.markdown("""
이 도구는 **1개의 영상(mp4)과 1개의 라벨(json)** 파일을 업로드하여, 
모델의 디텍션 성능 및 신뢰성 항목(mAP, FPR 등)을 평가하고 최종 결과물 리포트를 생성합니다.

사용 방법:
1. 평가하고자 하는 **1개의 .mp4** 동영상 파일과 매칭되는 **1개의 .json** 라벨 파일을 하단 영역에 올려줍니다.
2. `[평가 실행]` 버튼을 누르면 내부적으로 신뢰성 테스트용 노트북(`reliability_test.ipynb`)을 실행합니다.
3. 모델 추론 과정과 결과 생성이 완료되면, 탐지 결과 이미지가 포함된 폴더와 함께 전체 결과가 요약된 **reports.zip** 파일을 다운로드할 수 있습니다.

> 주의: 원활한 매칭을 위해 업로드하는 영상과 라벨 파일의 베이스 이름(stem)이 일치하는 것이 좋습니다.
""")

import pandas as pd

st.subheader("📊 Test Suite 커버리지 (매핑표)")
DOT, TRI, NONE = "●", "△", "−"
mapping_data = [
    ("TC0", "Baseline(원본)",            TRI, NONE, TRI, NONE, DOT),
    ("TC1", "저조도(조명 L2)",            DOT, NONE, DOT, NONE, DOT),
    ("TC2", "극저조도+노이즈(L3)",        DOT, TRI, DOT, NONE, DOT),
    ("TC3", "우천 합성(W2)",              DOT, TRI, DOT, NONE, DOT),
    ("TC4", "안개 합성(W3)",              DOT, TRI, DOT, NONE, DOT),
    ("TC5", "저해상도(S3)",               TRI, NONE, DOT, NONE, DOT),
    ("TC6", "복합(L3+W3+S3)",             DOT, TRI, DOT, NONE, DOT),
]
scenario_map_df = pd.DataFrame(mapping_data, columns=[
    "시나리오", "설명", "Safety", "XAI", "Robust", "Fairness", "Transparency"
])
st.table(scenario_map_df)

st.subheader("2) 실행할 시나리오 선택")
all_scenarios = ["TC0", "TC1", "TC2", "TC3", "TC4", "TC5", "TC6"]
selected_scenarios = st.multiselect(
    "실행할 시나리오를 선택하세요 (기본: 전체)",
    options=all_scenarios,
    default=all_scenarios,
    help="선택된 시나리오만 평가됩니다. 비워두면 전체가 실행됩니다."
)
if not selected_scenarios:
    selected_scenarios = all_scenarios
    st.info("선택된 시나리오가 없어 전체 시나리오로 실행됩니다.")

st.subheader("3) 데이터 업로드 (드래그 앤 드롭)")

video = st.file_uploader(
    "영상 파일 (mp4) - 1개 선택",
    type=["mp4"],
    accept_multiple_files=False
)

label = st.file_uploader(
    "라벨 파일 (json) - 1개 선택",
    type=["json"],
    accept_multiple_files=False
)

if (not video) or (not label):
    st.info("영상(mp4)과 라벨(json)을 각각 1개씩 업로드하면 실행 버튼이 활성화됩니다.")

notebook_path = Path(__file__).resolve().parent / "reliability_test.ipynb"

if "is_running" not in st.session_state:
    st.session_state.is_running = False

def start_run():
    st.session_state.is_running = True

run_btn = st.button(
    "4) 평가 실행",
    type="primary",
    disabled=(not video or not label or st.session_state.is_running),
    on_click=start_run
)

if st.session_state.is_running:
    import shutil

    if "tmp_dir" not in st.session_state:
        st.session_state.tmp_dir = tempfile.mkdtemp()
        st.session_state.process = None
        st.session_state.log_path = None
        st.session_state.reports_dir = None
        st.session_state.is_done = False

    tmp = Path(st.session_state.tmp_dir)
    base = tmp / "base"
    videos_dir = base / "aihub" / "raw" / "videos"
    labels_dir = base / "aihub" / "raw" / "labels"

    # 최초 파일 저장 및 백그라운드 프로세스 실행
    if st.session_state.process is None and not st.session_state.get("is_done", False):
        videos_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        st.info("업로드 파일 저장 중...")
        (videos_dir / video.name).write_bytes(video.getvalue())
        (labels_dir / label.name).write_bytes(label.getvalue())

        try:
            proc, log_p, rep_dir = run_reliability_notebook(
                notebook_path=Path(notebook_path),
                base_dir=base,
                skip_existing=True,
                selected_scenarios=selected_scenarios,
            )
            st.session_state.process = proc
            st.session_state.log_path = log_p
            st.session_state.reports_dir = rep_dir
        except Exception as e:
            st.exception(e)
            st.session_state.is_running = False
            st.stop()

    # ── 중단 버튼 (fragment 밖 → 클릭 시 전체 rerun으로 상태 초기화) ──
    st.info("노트북 실행 중... (환경에 따라 시간이 걸릴 수 있음)")

    if st.button("🚫 평가 중단 (Stop)", type="secondary"):
        if st.session_state.process:
            st.session_state.process.kill()
        st.session_state.is_running = False
        st.session_state.process = None
        st.session_state.is_done = False
        st.warning("작업이 중단되었습니다.")
        st.rerun()

    tqdm_pattern = re.compile(r"(\d+)%\|.*\|\s*(\d+)/(\d+)")

    # ── @st.fragment: 로그 패널만 1초마다 독립 갱신 (페이지 스크롤 위치 유지) ──
    @st.fragment(run_every=1)
    def _log_panel():
        prog = st.progress(st.session_state.get("_last_pct", 0.0))
        prog_txt = st.empty()

        log_path = st.session_state.get("log_path")
        if log_path and Path(log_path).exists():
            with open(log_path, "r", encoding="utf-8") as f:
                logs = f.read()
            lines = logs.replace("\r", "\n").split("\n")
            lines = [ln for ln in lines if ln.strip()]

            escaped = _html.escape("\n".join(lines[-200:]))
            st.markdown(
                f'<div style="max-height:420px;overflow-y:auto;'
                f'background:#1E1E1E;color:#D4D4D4;padding:10px;border-radius:5px;'
                f'font-family:monospace;font-size:13px;white-space:pre-wrap;">'
                f'{escaped}</div>',
                unsafe_allow_html=True
            )

            for line in reversed(lines):
                m = tqdm_pattern.search(line)
                if m and m.group(1):
                    try:
                        pct = int(m.group(1)) / 100.0
                        st.session_state["_last_pct"] = pct
                        prog.progress(max(0.0, min(pct, 1.0)))
                        prog_txt.text(f"진행 상황: {m.group(1)}%")
                        break
                    except ValueError:
                        pass

        # 프로세스 완료 감지 (fragment 내부에서 처리)
        proc = st.session_state.get("process")
        if proc is not None:
            retcode = proc.poll()
            if retcode is not None:
                if retcode == 0:
                    st.session_state.is_done = True
                else:
                    st.error(f"실행 중 오류 발생 (Exit code: {retcode})")
                    st.session_state.is_running = False
                st.session_state.process = None
                st.rerun()  # 완료 후 전체 페이지 rerun → 결과물 표시

    with st.expander("실시간 로그 보기", expanded=True):
        _log_panel()

    # ── 완료 후 결과물 출력 ──
    if st.session_state.get("is_done", False):
        reports_dir = st.session_state.reports_dir
        report_md = Path(reports_dir) / "report.md"
        if report_md.exists():
            st.subheader("요약 리포트 (report.md)")
            st.markdown(report_md.read_text(encoding="utf-8", errors="replace"))

        runs1 = Path.cwd() / "runs"
        runs2 = tmp / "base" / "runs"
        runs_out = Path(reports_dir) / "runs"
        if runs1.exists():
            shutil.move(str(runs1), str(runs_out))
        elif runs2.exists():
            shutil.move(str(runs2), str(runs_out))

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        zip_filename = f"reports-{timestamp}.zip"

        out_zip = tmp / zip_filename
        zip_dir(Path(reports_dir), out_zip)

        st.download_button(
            f"{zip_filename} 다운로드",
            data=out_zip.read_bytes(),
            file_name=zip_filename,
            mime="application/zip",
            type="primary"
        )

        st.subheader("생성된 파일 목록")
        if reports_dir and Path(reports_dir).exists():
            files = sorted([
                p.relative_to(reports_dir).as_posix()
                for p in Path(reports_dir).rglob("*") if p.is_file()
            ])
            st.code("\n".join(files), language="text")

        if st.button("🔄 새로운 평가 시작"):
            st.session_state.is_running = False
            st.session_state.is_done = False
            del st.session_state.tmp_dir
            st.rerun()
