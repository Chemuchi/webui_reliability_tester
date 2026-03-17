
from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional
import re
import io
import contextlib
import os


def _replace_assignment(source: str, var_name: str, value_expr: str) -> tuple[str, bool]:
    """
    코드 셀의 `VAR = ...` 대입문을 찾아 `VAR = value_expr` 형태로 교체합니다.
    """
    pattern = re.compile(rf"^(\s*){re.escape(var_name)}\s*=.*$", flags=re.M)
    replaced = bool(pattern.search(source))
    if not replaced:
        return source, False
    return pattern.sub(rf"\1{var_name} = {value_expr}", source), True


def _patch_notebook_cells(
    nb,
    base_dir: Path,
    max_videos: Optional[int],
    max_images_total: Optional[int],
    skip_existing: bool,
):
    """
    실행 전에 노트북 코드셀 내부 상수(BASE, MAX_VIDEOS, MAX_IMAGES_TOTAL, SKIP_EXISTING)를 교체합니다.
    """
    base_dir = Path(base_dir).resolve()
    base_expr = f'Path(r"{base_dir.as_posix()}").resolve()'
    max_videos_expr = "None" if max_videos is None else str(int(max_videos))
    max_images_total_expr = "None" if max_images_total is None else str(int(max_images_total))
    skip_existing_expr = str(bool(skip_existing))

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue

        src = cell.get("source", "")
        src, _ = _replace_assignment(src, "BASE", base_expr)
        src, _ = _replace_assignment(src, "MAX_VIDEOS", max_videos_expr)
        src, _ = _replace_assignment(src, "MAX_IMAGES_TOTAL", max_images_total_expr)
        src, _ = _replace_assignment(src, "SKIP_EXISTING", skip_existing_expr)
        cell["source"] = src

    return nb


@contextlib.contextmanager
def _temporary_env(updates: dict[str, str]):
    """
    노트북 실행 중에만 환경변수를 일시 적용합니다.
    """
    old_values = {k: os.environ.get(k) for k in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for k, old in old_values.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


@contextlib.contextmanager
def _temporary_cwd(path: Path):
    """
    노트북 실행 중 상대경로 산출물이 작업 폴더 밖으로 새지 않도록 cwd를 고정합니다.
    """
    path = Path(path).resolve()
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(path)
        yield
    finally:
        try:
            os.chdir(previous)
        except Exception:
            pass


def _format_reports_error(base_dir: Path) -> str:
    base_dir = Path(base_dir).resolve()
    expected = base_dir / "reports"
    candidates = [p for p in base_dir.rglob("reports") if p.is_dir()]
    if candidates:
        found = ", ".join(str(p) for p in sorted(candidates))
        return (
            "reports 디렉토리가 예상 경로에 생성되지 않았습니다. "
            f"예상 경로: {expected}. 발견된 후보: {found}"
        )
    return f"reports 디렉토리가 생성되지 않았습니다. 예상 경로: {expected}"

import subprocess
from typing import Optional, Callable, Tuple

def run_reliability_notebook(
    notebook_path: Path,
    base_dir: Path,
    max_videos: Optional[int] = None,
    max_images_total: Optional[int] = None,
    skip_existing: bool = True,
    selected_scenarios: Optional[list] = None,
) -> Tuple[subprocess.Popen, Path, Path]:
    """
    reliability_test.ipynb 를 실행해서 base_dir/reports 를 생성하고, 그 경로를 반환합니다.
    - notebook_path: 팀원이 만든 ipynb
    - base_dir: 데이터셋이 풀려있는 루트 (aihub/raw/... 가 이 아래 있어야 함)
    """
    notebook_path = Path(notebook_path)
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    try:
        import nbformat
        from nbclient import NotebookClient
    except Exception as e:
        raise RuntimeError("nbformat/nbclient 가 필요합니다. 설치: pip install nbformat nbclient") from e

    nb = nbformat.read(notebook_path, as_version=4)
    nb = _patch_notebook_cells(
        nb=nb,
        base_dir=base_dir,
        max_videos=max_videos,
        max_images_total=max_images_total,
        skip_existing=skip_existing,
    )
    # nbclient 대신 직접 파이썬 스크립트로 변환 후 subprocess 실행
    source_lines = []
    for cell in nb.cells:
        if cell.cell_type == 'code':
            lines = cell.source.splitlines()
            # 매직 커맨드 제거
            lines = [line for line in lines if not line.strip().startswith(('!', '%'))]
            source_lines.append("\n".join(lines))
    script_source = "\n\n".join(source_lines)

    base_dir = Path(base_dir).resolve()
    
    # 선택된 시나리오를 SCENARIOS 정의 직후에 필터링 코드를 삽입
    if selected_scenarios:
        names_repr = repr(selected_scenarios)
        filter_snippet = f"\nSCENARIOS = [s for s in SCENARIOS if s.name in {names_repr}]\n"
        # SCENARIOS = [ 정의 바로 다음 줄 이후에 삽입
        # 가장 마지막 ']' 닫는 지점 다음에 붙이기
        import re as _re
        script_source = _re.sub(
            r'(SCENARIOS\s*=\s*\[.*?\])',
            r'\1' + filter_snippet.replace('\\', '\\\\'),
            script_source,
            count=1,
            flags=_re.DOTALL
        )

    script_path = base_dir / "run_script.py"
    script_path.write_text(script_source, encoding="utf-8")
    
    log_path = base_dir / "kernel_log.txt"
    log_file = open(log_path, "w", encoding="utf-8")

    env_updates = os.environ.copy()
    env_updates["BASE_DIR"] = str(base_dir)
    env_updates["YOLO_CONFIG_DIR"] = str(base_dir / ".ultralytics")
    env_updates["PYTHONUNBUFFERED"] = "1"

    import subprocess
    import sys
    
    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(base_dir),
        env=env_updates,
        text=True
    )

    return process, log_path, base_dir / "reports"
    if not reports_dir.exists():
        raise RuntimeError(_format_reports_error(base_dir))
    return reports_dir

def zip_dir(src_dir: Path, out_zip: Path) -> Path:
    import zipfile
    src_dir = Path(src_dir)
    out_zip = Path(out_zip)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(src_dir).as_posix())
    return out_zip
