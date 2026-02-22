from flask import Flask, render_template, request, flash, redirect, url_for, session, send_from_directory, jsonify
import numpy as np
import itertools
import pandas as pd
from datetime import datetime
import os
from fractions import Fraction
from functools import reduce
import operator
from threading import Lock
import uuid
import logging
from logging.handlers import RotatingFileHandler
import re


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SURVEY_XLSX = os.path.join(DATA_DIR, "survey_results.xlsx")
log_file = os.path.join(LOG_DIR, "ahp_app.log")

excel_write_lock = Lock()
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

app = Flask(__name__)
# 환경 변수에서 Secret Key를 불러옴
app.secret_key = os.environ.get('FLASK_SECRET_KEY')

# 환경변수 비어 있을 경우 대비 (선택)
if not app.secret_key:
    raise RuntimeError("ERROR: FLASK_SECRET_KEY environment variable is not set.")


handler = RotatingFileHandler(
    log_file,
    maxBytes=5 * 1024 * 1024,   # 5MB
    backupCount=5,
    encoding="utf-8"
)

formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(message)s"
)
handler.setFormatter(formatter)

handler.setLevel(logging.INFO)

app.logger.setLevel(logging.INFO)
app.logger.addHandler(handler)


# AHP 계층 구조 정의
AHP_HIERARCHY = {
    "goal": "애로요인 우선순위 분석",
    "criteria": ["창업자금", "비즈니스모델", "경영관리"],
    "sub_criteria": {
        "창업자금": ["운영자금", "시설자금", "초기 자기자본", "투자유치"],
        "비즈니스모델": ["사업타당성(BM)", "특허분쟁", "기술개발", "시장정보획득", "기술사업화"],
        "경영관리": ["네트워크활용", "전문인력 확보", "판로개척", "해외시장개척"]
    }
}

# Random Index (RI) for consistency check
RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
CR_THRESHOLD_INDIVIDUAL = 0.20
CR_THRESHOLD_GROUP = 0.15

def get_pairwise_comparisons(items):
    """항목 리스트로부터 모든 쌍대비교 조합을 생성합니다."""
    return list(itertools.combinations(items, 2))

@app.route('/ahp/')
def index():
    """설문 시작 페이지. 설문 목적을 안내하고 응답자 정보를 입력받습니다."""
    # 응답자 정보 선택 옵션 정의
    respondent_options = {
        "age_groups": ["20대", "30대", "40대", "50대", "60대 이상"],
        "regions": ["경북(영천)", "경북(포항)", "경북(구미)", "경북(경산)", "경북(경주)", "경북(안동)", "경북(기타)", "대구"],
        "fields": ["정보통신", "전기전자", "식품/바이오", "기계/제조", "디자인", "도소매", "서비스", "교육", "기타"],
        "experience_levels": ["예비창업자", "1년 미만", "1~3년 미만", "3~7년 미만", "7년 이상"]
    }
    return render_template('index.html', options=respondent_options)


@app.route('/ahp/api/cr/step1', methods=['POST'])
def api_cr_step1():
    try:
        data = request.get_json(force=True) or {}
        criteria = AHP_HIERARCHY["criteria"]

        comparisons = {}
        for a, b in get_pairwise_comparisons(criteria):
            key = f"criteria_{a}_{b}"
            form = data.get("form") or data  # ✅ {form:{...}} 이든 {...} 이든 다 받기
            raw = form.get(key)
            if raw is None:
                return jsonify({"ready": False, "message": "모든 문항을 선택하면 CR을 계산합니다."})
            comparisons[(a, b)] = parse_ahp_value(raw)

        _, _, _, _, cr, _ = calculate_ahp_weights(criteria, comparisons, cr_threshold=CR_THRESHOLD_INDIVIDUAL)

        cr_f = float(cr)  # ✅ numpy float -> python float
        ok_b = bool(cr_f <= float(CR_THRESHOLD_INDIVIDUAL))
        return jsonify({
            "ready": True,                    # python bool
            "cr": cr_f,                       # python float
            "ok": ok_b,                       # python bool
            "threshold": float(CR_THRESHOLD_INDIVIDUAL)
        })
    except Exception as e:
        app.logger.exception(f"[API CR STEP1 FAIL] {type(e).__name__}: {e}")
        # 입력이 이상하면 그냥 안내만
        return jsonify({"ready": False, "message": "CR 계산 중 오류가 발생했습니다."}), 200

@app.route('/ahp/api/cr/step2', methods=['POST'])
def api_cr_step2():
    try:
        data = request.get_json(force=True) or {}
        results = []
        all_ready = True
        all_ok = True

        for crit, sub_crits in AHP_HIERARCHY["sub_criteria"].items():
            if len(sub_crits) <= 1:
                results.append({"crit": crit, "ready": True, "cr": 0.0, "ok": True, "threshold": CR_THRESHOLD_INDIVIDUAL})
                continue

            comparisons = {}
            for a, b in get_pairwise_comparisons(sub_crits):
                key = f"sub_{crit}_{a}_{b}"
                form = data.get("form") or data  # ✅ {form:{...}} 이든 {...} 이든 다 받기
                raw = form.get(key)
                if raw is None:
                    all_ready = False
                    results.append({"crit": crit, "ready": False, "message": "모든 문항 선택 후 계산"})
                    break
                comparisons[(a, b)] = parse_ahp_value(raw)

            # 해당 crit이 ready=False로 들어갔으면 다음 그룹
            if results and results[-1].get("crit") == crit and results[-1].get("ready") is False:
                continue

            _, _, _, _, cr, _ = calculate_ahp_weights(sub_crits, comparisons, cr_threshold=CR_THRESHOLD_INDIVIDUAL)

            cr_f = float(cr)
            ok_b = bool(cr_f <= float(CR_THRESHOLD_INDIVIDUAL))

            results.append({
                "crit": crit,
                "ready": True,
                "cr": cr_f,
                "ok": ok_b,
                "threshold": float(CR_THRESHOLD_INDIVIDUAL)
            })

        return jsonify({
                    "ready": bool(all_ready),
                    "ok": bool(all_ok),
                    "threshold": float(CR_THRESHOLD_INDIVIDUAL),
                    "results": results
                })

    except Exception as e:
        # ✅ 여기 추가(스택트레이스 포함)
        app.logger.exception(f"[API CR STEP2 FAIL] {type(e).__name__}: {e}")
        return jsonify({"ready": False, "message": "CR 계산 중 오류가 발생했습니다."}), 200



@app.route('/ahp/start_survey', methods=['POST'])
def start_survey():
    """응답자 정보를 세션에 저장하고 설문 페이지로 리디렉션합니다."""
    try:
        # 기본 값 수집
        age = request.form['age']
        region = request.form['region']
        field = request.form['field']
        experience = request.form['experience']

        email = request.form.get('email', '').strip()
        if email and not EMAIL_RE.match(email):
            flash("이메일 형식이 올바르지 않습니다. 예: name@example.com", "error")
            return redirect('/ahp/')

        # '기타' 선택 시 사용자가 입력한 값 우선 사용
        if field == '기타':
            field_other = request.form.get('field_other', '').strip()
            if not field_other:
                flash("창업 분야에서 '기타'를 선택하셨다면 직접 입력해주세요.", "error")
                return redirect('/ahp/')
            field = field_other

        # 예비창업자가 아닌 경우 창업기업 이름을 필수로 입력 받음
        company_name = None
        if experience != '예비창업자':
            company_name = request.form.get('company_name', '').strip()
            if not company_name:
                flash("창업 연차가 예비창업자가 아닌 경우 창업기업 이름을 입력해주세요.", "error")
                return redirect('/ahp/')

        session['respondent_info'] = {
            "age": age,
            "region": region,
            "field": field,
            "experience": experience,
            "company_name": company_name, 
            "email": email
        }
        return redirect('/ahp/survey')
    except KeyError:
        flash("모든 정보를 선택해주세요.", "error")
        return redirect('/ahp/')

@app.route('/ahp/survey', methods=['GET', 'POST'])
def ahp_survey_step1():
    form_data = session.get('step1_form_data')
    criteria_pairs = get_pairwise_comparisons(AHP_HIERARCHY["criteria"])

    if request.method == 'POST':
        data = request.form.to_dict(flat=True)

        # Step1 비교값 파싱/일관성 검사
        try:
            criteria = AHP_HIERARCHY["criteria"]
            comparisons = {}
            for a, b in get_pairwise_comparisons(criteria):
                key = f"criteria_{a}_{b}"
                comparisons[(a, b)] = parse_ahp_value(data[key])

            _, _, _, _, cr, _ = calculate_ahp_weights(criteria, comparisons, cr_threshold=CR_THRESHOLD_INDIVIDUAL)

            if cr > CR_THRESHOLD_INDIVIDUAL:
                flash(
                    f"1단계(주요 항목) 응답의 일관성이 낮습니다. (CR={cr:.3f}, 기준: {CR_THRESHOLD_INDIVIDUAL} 미만) "
                    f"서로의 중요도 판단이 모순되지 않도록 다시 선택해 주세요.",
                    "error"
                )
                session['step1_form_data'] = data
                return redirect('/ahp/survey')

        except Exception:
            flash("모든 문항에 응답해주세요.", "error")
            session['step1_form_data'] = data
            return redirect('/ahp/survey')

        # 통과 시 저장 후 Step2로
        session['step1_form_data'] = data
        return redirect('/ahp/survey/step2')

    return render_template(
        'ahp_step1.html',
        hierarchy=AHP_HIERARCHY,
        criteria_pairs=criteria_pairs,
        sub_criteria_map=AHP_HIERARCHY["sub_criteria"],
        form_data=form_data
    )


@app.route('/ahp/survey/step2', methods=['GET', 'POST'])
def ahp_survey_step2():
    if 'step1_form_data' not in session:
        flash("먼저 1단계(주요 항목 그룹 비교)를 완료해주세요.", "error")
        return redirect('/ahp/survey')

    form_data = session.get('step2_form_data')

    sub_criteria_pairs = {
        crit: get_pairwise_comparisons(sub_crits)
        for crit, sub_crits in AHP_HIERARCHY["sub_criteria"].items()
    }

    if request.method == 'POST':
        data = request.form.to_dict(flat=True)
        action = data.get('action')

        # back일 때만 step2 임시저장
        if action == 'back':
            data.pop('action', None)
            session['step2_form_data'] = data
            return redirect('/ahp/survey')

        # submit은 이제 이 라우트로 안 오게 되지만, 혹시 대비
        flash("잘못된 접근입니다. 다시 시도해주세요.", "error")
        return redirect('/ahp/survey/step2')

    return render_template(
        'ahp_step2.html',
        hierarchy=AHP_HIERARCHY,
        sub_criteria_pairs=sub_criteria_pairs,
        form_data=form_data,
        CR_THRESHOLD_INDIVIDUAL=CR_THRESHOLD_INDIVIDUAL
    )


def process_ahp_submission(form: dict, validate_step1: bool = True):
    """Step1+Step2 입력(dict)을 바탕으로 AHP 분석/저장 수행.
       - validate_step1=True: Step1 CR도 확인(안전장치)
       - validate_step1=False: Step1 CR 검증은 생략(의도대로)하고 Step2만 검증/처리
    """
    try:
        # --- 1) Step1: 주요 기준(criteria) 행렬/가중치 계산(항상 필요) ---
        criteria = AHP_HIERARCHY["criteria"]
        criteria_comparisons = {}
        for item1, item2 in get_pairwise_comparisons(criteria):
            key = f"criteria_{item1}_{item2}"
            criteria_comparisons[(item1, item2)] = parse_ahp_value(form[key])

        crit_matrix, crit_weights, _, _, crit_cr, _ = calculate_ahp_weights(criteria, criteria_comparisons, cr_threshold=CR_THRESHOLD_INDIVIDUAL)

        # ✅ Step1 CR 검증은 옵션일 때만 + CR ≤ 0.2 기준 적용
        if validate_step1 and (crit_cr > CR_THRESHOLD_INDIVIDUAL):
            flash(
                f"1단계(주요 항목) 응답의 일관성이 낮아 결과를 계산할 수 없습니다. "
                f"(CR = {crit_cr:.3f}, 기준: {CR_THRESHOLD_INDIVIDUAL} 이하) "
                f"서로의 중요도 판단이 모순되지 않도록 다시 선택해 주세요.",
                "error"
            )
            session['step1_form_data'] = {k: str(v) for k, v in form.items() if k.startswith('criteria_')}
            return redirect('/ahp/survey')

        crit_weights_map = {crit: float(w) for crit, w in zip(criteria, crit_weights)}

        # --- 2) Step2: 하위 기준(sub-criteria) + CR 검증(여기는 항상 검증) ---
        sub_criteria_results = {}
        final_weights = {}
        sub_matrices = {}

        for crit, sub_crits in AHP_HIERARCHY["sub_criteria"].items():
            if len(sub_crits) > 1:
                sub_comparisons = {}
                for item1, item2 in get_pairwise_comparisons(sub_crits):
                    key = f"sub_{crit}_{item1}_{item2}"
                    sub_comparisons[(item1, item2)] = parse_ahp_value(form[key])

                sub_matrix, sub_weights, _, _, sub_cr, _ = calculate_ahp_weights(sub_crits, sub_comparisons, cr_threshold=CR_THRESHOLD_INDIVIDUAL)
                sub_matrices[crit] = sub_matrix

                # ✅ Step2는 항상 CR ≤ 0.2 기준으로 검증
                if sub_cr > CR_THRESHOLD_INDIVIDUAL:
                    flash(
                        f"[{crit}] 그룹 세부 항목 비교의 일관성이 낮아 결과를 계산할 수 없습니다. "
                        f"(CR = {sub_cr:.3f}, 기준: {CR_THRESHOLD_INDIVIDUAL} 이하) "
                        f"비교 판단이 일관되도록 다시 선택해 주세요.",
                        "error"
                    )
                    session['step2_form_data'] = {k: str(v) for k, v in form.items() if k.startswith('sub_')}
                    return redirect('/ahp/survey/step2')

                sub_weights_map = {sub: float(w) for sub, w in zip(sub_crits, sub_weights)}
                sub_criteria_results[crit] = {"weights": sub_weights_map, "cr": float(sub_cr)}

                for sub_crit, sub_weight in sub_weights_map.items():
                    final_weights[sub_crit] = crit_weights_map[crit] * sub_weight

            else:
                # 하위항목이 1개면 비교 불필요
                sub_crit = sub_crits[0]
                sub_criteria_results[crit] = {"weights": {sub_crit: 1.0}, "cr": 0.0}
                sub_matrices[crit] = np.array([[1.0]])
                final_weights[sub_crit] = crit_weights_map[crit] * 1.0

        sorted_final_weights = sorted(final_weights.items(), key=lambda item: item[1], reverse=True)

        respondent_info = session.get('respondent_info', {})

        try:
            app.logger.info(
                f"[AHP SAVE PRE] respondent={bool(respondent_info)}, "
                f"crit_cr={crit_cr:.4f}, threshold={CR_THRESHOLD_INDIVIDUAL}, "
                f"crit_matrix_shape={getattr(crit_matrix, 'shape', None)}, "
                f"sub_keys={list(sub_matrices.keys())}"
            )

            save_to_excel(
                respondent_info=respondent_info,
                crit_matrix=crit_matrix, crit_items=criteria,
                sub_matrices=sub_matrices, sub_items_map=AHP_HIERARCHY["sub_criteria"]
            )
        except Exception as e:
            app.logger.exception("Excel save error")
            flash(f"결과를 엑셀 파일에 저장하는 중 오류가 발생했습니다. ({type(e).__name__}: {e})", "warning")

        session['last_result'] = {
            "crit_weights": crit_weights_map,
            "crit_cr": float(crit_cr),
            "sub_criteria_results": sub_criteria_results,
            "final_weights": sorted_final_weights,
            "respondent_info": respondent_info,
        }

        session.pop('step1_form_data', None)
        session.pop('step2_form_data', None)

        return redirect('/ahp/result')

    except (KeyError, ValueError) as e:
        app.logger.error(f"Form processing error: {e}")
        flash("모든 항목에 응답해주세요.", "error")
        session['step2_form_data'] = {k: str(v) for k, v in form.items() if k.startswith('sub_')}
        session['step1_form_data'] = {k: str(v) for k, v in form.items() if k.startswith('criteria_')}
        return redirect('/ahp/survey/step2')
    except Exception as e:
        app.logger.exception("Unexpected error")
        flash("예기치 못한 오류가 발생했습니다. 관리자에게 문의해주세요.", "error")
        return redirect('/ahp/survey')



def validate_step2_only(step2_form: dict):
    for crit, sub_crits in AHP_HIERARCHY["sub_criteria"].items():
        if len(sub_crits) <= 1:
            continue

        comparisons = {}
        for item1, item2 in get_pairwise_comparisons(sub_crits):
            key = f"sub_{crit}_{item1}_{item2}"
            comparisons[(item1, item2)] = parse_ahp_value(step2_form.get(key))

        _, _, _, _, cr, ok = calculate_ahp_weights(sub_crits, comparisons, cr_threshold=CR_THRESHOLD_INDIVIDUAL)
        if not ok:
            return False, crit, cr

    return True, None, None

@app.route('/ahp/submit2', methods=['POST'])
def ahp_submit2():
    if 'step1_form_data' not in session:
        flash("먼저 1단계(주요 항목 그룹 비교)를 완료해주세요.", "error")
        return redirect('/ahp/survey')

    step1_data = session.get('step1_form_data', {})

    # Step2 데이터는 POST로 직접 들어옴
    step2_data = request.form.to_dict(flat=True)
    step2_data.pop('action', None)

    # step2 임시 저장(일관성 실패 시 step2로 돌아왔을 때 유지)
    session['step2_form_data'] = step2_data

    # ✅ (핵심) submit2에서는 Step2 CR만 검사
    ok, bad_crit, cr = validate_step2_only(step2_data)
    if not ok:
        flash(
            f"[{bad_crit}] 그룹 세부 항목 비교의 일관성이 낮아 결과를 계산할 수 없습니다. "
            f"(CR = {cr:.3f}, 기준: {CR_THRESHOLD_INDIVIDUAL} 미만) "
            f"비교 판단이 일관되도록 다시 선택해 주세요.",
            "error"
        )
        return redirect('/ahp/survey/step2')

    merged = {}
    merged.update(step1_data)
    merged.update(step2_data)

    # ✅ Step1은 이미 step1에서 검증했으므로 여기서는 생략
    return process_ahp_submission(merged, validate_step1=False)

@app.route('/ahp/result', methods=['GET'])
def ahp_result():
    data = session.get('last_result')
    if not data:
        flash("표시할 결과가 없습니다. 설문을 먼저 진행해주세요.", "error")
        return redirect('/ahp/')
    return render_template('ahp_result.html', **data)

@app.route('/ahp/reset', methods=['GET'])
def ahp_reset():
    session.clear()  # 또는 respondent_info / step1_form_data / step2_form_data만 선택적으로 pop
    return redirect('/ahp/')


def geometric_mean(values):
    """주어진 값들의 기하평균을 계산합니다."""
    values = [v for v in values if v > 0] # 0 또는 음수는 제외
    if not values:
        return 1
    return reduce(operator.mul, values) ** (1.0 / len(values))

ALLOWED_AHP_VALUES = {
    9.0, 7.0, 5.0, 3.0, 1.0,
    1/3, 1/5, 1/7, 1/9
}

def parse_ahp_value(raw):
    """
    AHP 라디오 값 파싱:
    - "3" -> 3.0
    - "1/3" -> 0.333...
    - "0.3333333" -> 0.333...
    """
    if raw is None:
        raise ValueError("AHP value is missing")

    s = str(raw).strip()
    if not s:
        raise ValueError("AHP value is empty")

    val = float(Fraction(s)) if "/" in s else float(s)

    if not any(abs(val - a) < 1e-9 for a in ALLOWED_AHP_VALUES):
        raise ValueError(f"Invalid AHP value: {s}")

    return val

def calculate_ahp_weights(items, comparisons, cr_threshold=CR_THRESHOLD_INDIVIDUAL):
    """쌍대비교 행렬을 생성하고 가중치 및 일관성 비율을 계산합니다."""
    n = len(items)
    matrix = np.ones((n, n))
    item_to_idx = {item: i for i, item in enumerate(items)}

    if n == 1:
        ci = 0
        cr = 0
        is_consistent = True
        return matrix, np.array([1.0]), 1.0, ci, cr, is_consistent

    if n > 1:
        for (item1, item2), value in comparisons.items():
            i, j = item_to_idx[item1], item_to_idx[item2]
            matrix[i, j] = value
            matrix[j, i] = 1 / value

    # 가중치 계산 (Eigenvector Method)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_eigenvalue = np.max(eigenvalues.real)
    max_eigenvector = eigenvectors[:, np.argmax(eigenvalues.real)].real
    weights = max_eigenvector / np.sum(max_eigenvector)

    # 일관성 검증
    ci = (max_eigenvalue - n) / (n - 1)
    cr = ci / RI[n] if n in RI and RI[n] > 0 else 0

    is_consistent = cr < cr_threshold

    return matrix, weights, max_eigenvalue, ci, cr, is_consistent

def make_sheet_name(prefix="A"):
    # 31자 제한 고려: prefix(1) + 6 + 1 + 6 + 1 + 6 = 21자
    return f"{prefix}{datetime.now():%y%m%d}_{datetime.now():%H%M%S}_{uuid.uuid4().hex[:6].upper()}"

def save_to_excel(respondent_info, crit_matrix, crit_items, sub_matrices, sub_items_map):
    filename = SURVEY_XLSX
    sheet_name = make_sheet_name("S")

    # 기록할 데이터프레임 준비
    dfs_to_write = []

    respondent_df = pd.DataFrame([respondent_info])
    dfs_to_write.append(("respondent", respondent_df))

    dfs_to_write.append(("title", pd.DataFrame([["주요 항목 그룹 비교 행렬"]])))
    crit_df = pd.DataFrame(crit_matrix, index=crit_items, columns=crit_items)
    dfs_to_write.append(("matrix", crit_df))

    for crit_name, matrix in sub_matrices.items():
        sub_items = sub_items_map[crit_name]
        dfs_to_write.append(("title", pd.DataFrame([[f"'{crit_name}' 그룹 내 세부 항목 비교 행렬"]])))
        sub_df = pd.DataFrame(matrix, index=sub_items, columns=sub_items)
        dfs_to_write.append(("matrix", sub_df))

    try:
        with excel_write_lock:
            file_exists = os.path.exists(filename)

            # ✅ 파일이 있으면 append + overlay
            if file_exists:
                writer = pd.ExcelWriter(
                    filename,
                    engine="openpyxl",
                    mode="a",
                    if_sheet_exists="overlay"   # 핵심!
                )
            else:
                writer = pd.ExcelWriter(
                    filename,
                    engine="openpyxl",
                    mode="w"
                )

            with writer:
                current_row = 0

                # 응답자 정보(헤더 포함)
                _, df0 = dfs_to_write[0]
                df0.to_excel(writer, sheet_name=sheet_name, index=False, header=True, startrow=current_row)
                current_row += len(df0) + 1 + 2

                # 나머지들
                for kind, df in dfs_to_write[1:]:
                    if kind == "title":
                        df.to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=current_row)
                        current_row += len(df) + 1
                    else:
                        df.to_excel(writer, sheet_name=sheet_name, index=True, header=True, startrow=current_row)
                        current_row += len(df) + 1 + 2

        app.logger.info(f"[SAVE OK] {filename} (sheet={sheet_name}), cwd={os.getcwd()}")
        return filename

    except Exception as e:
        app.logger.exception(f"[SAVE FAIL] filename={filename}, cwd={os.getcwd()}, err={e}")
        raise

def save_group_result_to_excel(out_path, num_respondents, crit_matrix, crit_items, sub_matrices, sub_items_map, final_weights):
    """그룹 분석 결과를 엑셀 파일에 저장합니다."""
    sheet_name = f"Group_Result_{num_respondents}명"

    with pd.ExcelWriter(out_path, engine='openpyxl', mode='w') as writer:
        info_df = pd.DataFrame([{"분석 참여 인원": num_respondents}])
        info_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
        current_row = len(info_df) + 2

        crit_df = pd.DataFrame(crit_matrix, index=crit_items, columns=crit_items)
        pd.DataFrame([["주요 항목 그룹 통합 비교 행렬 (기하평균)"]]).to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=current_row)
        current_row += 1
        crit_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row)
        current_row += len(crit_df) + 2

        for crit_name, matrix in sub_matrices.items():
            sub_items = sub_items_map[crit_name]
            sub_df = pd.DataFrame(matrix, index=sub_items, columns=sub_items)
            pd.DataFrame([[f"'{crit_name}' 그룹 내 세부 항목 통합 비교 행렬 (기하평균)"]]).to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=current_row)
            current_row += 1
            sub_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row)
            current_row += len(sub_df) + 2

        final_weights_df = pd.DataFrame(final_weights, columns=['세부 항목', '종합 가중치'])
        final_weights_df['순위'] = final_weights_df['종합 가중치'].rank(method='dense', ascending=False).astype(int)
        pd.DataFrame([["최종 우선순위 결과"]]).to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=current_row)
        current_row += 1
        final_weights_df[['순위', '세부 항목', '종합 가중치']].to_excel(writer, sheet_name=sheet_name, index=False, startrow=current_row)


def load_respondent_infos(xls: pd.ExcelFile):
    """
    각 시트의 맨 위(응답자 정보 1행)를 읽어 리스트로 반환
    반환 예:
      [{"sheet": "...", "age": "...", "region": "...", "field": "...", "experience": "...", "company_name": "..."}]
    """
    infos = []
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet, nrows=1)  # respondent_info 한 줄만
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
            infos.append({
                "sheet": sheet,
                "age": str(row.get("age", "")).strip(),
                "region": str(row.get("region", "")).strip(),
                "field": str(row.get("field", "")).strip(),
                "experience": str(row.get("experience", "")).strip(),
                "company_name": str(row.get("company_name", "")).strip() if row.get("company_name") is not None else "", 
                "email": str(row.get("email", "")).strip()
            })
        except Exception:
            # 시트 형식이 다르면 스킵
            continue
    return infos

def _norm(x) -> str:
    """엑셀에서 읽은 값을 비교 가능한 문자열로 정규화"""
    if x is None:
        return ""
    s = str(x).strip()
    # excel에서 nan이 문자열로 들어오는 경우 방지
    if s.lower() == "nan":
        return ""
    return s

def find_row_contains_text_anywhere(raw_df: pd.DataFrame, target: str):
    """시트 전체에서 target과 정확히 같은 셀을 가진 행 번호 반환"""
    t = _norm(target)
    for r_idx in range(len(raw_df)):
        row = raw_df.iloc[r_idx].tolist()
        if any(_norm(v) == t for v in row):
            return r_idx
    return None

def find_col_indices_by_headers(header_row: list, wanted_headers: list[str]):
    """
    header_row(행 전체)에서 wanted_headers에 해당하는 컬럼 인덱스 찾기
    공백/형변환/NaN 등에 강하도록 정규화해서 비교
    """
    header_norm = [_norm(v) for v in header_row]
    wanted_set = set(_norm(x) for x in wanted_headers)

    indices = [i for i, h in enumerate(header_norm) if h in wanted_set]
    return indices, header_norm

def find_header_row_containing_all_items(raw: pd.DataFrame, items: list[str], start_row=0, end_row=None):
    """
    raw: header=None 로 읽은 시트 전체 DataFrame
    items: 헤더 행에 반드시 포함되어야 하는 항목들
    start_row ~ end_row 범위에서 'items가 모두 등장하는 행'을 찾아 row index 반환
    """
    if end_row is None:
        end_row = len(raw)

    # 문자열 비교 안정화: strip + None/NaN 처리
    items_norm = [str(x).strip() for x in items]

    for r_idx in range(start_row, min(end_row, len(raw))):
        row = raw.iloc[r_idx].values
        values = [str(v).strip() for v in row if not pd.isna(v)]
        if all(item in values for item in items_norm):
            return r_idx

    return None


def run_group_ahp_analysis_from_sheets(xls: pd.ExcelFile, sheet_names: list[str]):
    """
    특정 sheet_names만 대상으로 그룹 AHP 분석 수행
    return:
      crit_weights_map, crit_cr, sub_criteria_results, sorted_final_weights,
      group_crit_matrix, group_sub_matrices
    """
    crit_items = AHP_HIERARCHY["criteria"]
    sub_criteria = AHP_HIERARCHY["sub_criteria"]
    n_crit = len(crit_items)

    all_crit_matrices = []
    all_sub_matrices = {crit: [] for crit in sub_criteria.keys()}

    for sheet in sheet_names:
        raw = pd.read_excel(xls, sheet_name=sheet, header=None)

        # --- 1) 주요 기준 헤더 찾기 ---
        crit_header_row = None
        for r_idx, row in raw.iterrows():
            values = [v for v in row.values if not pd.isna(v)]
            if all(item in values for item in crit_items):
                crit_header_row = r_idx
                break
        if crit_header_row is None:
            continue  # 형식 안 맞는 시트는 스킵

        crit_data_rows = raw.iloc[crit_header_row + 1: crit_header_row + 1 + n_crit, :]
        header_vals = list(raw.iloc[crit_header_row])
        crit_col_indices = [i for i, v in enumerate(header_vals) if v in crit_items]
        if not crit_col_indices:
            continue

        first_col = crit_col_indices[0]
        last_col = crit_col_indices[-1]
        crit_df = crit_data_rows.iloc[:, first_col - 1: last_col + 1]
        crit_df.columns = ["_row"] + crit_items
        crit_df = crit_df.set_index("_row")
        crit_df = crit_df.apply(pd.to_numeric, errors="raise")
        all_crit_matrices.append(crit_df.values)

        # --- 2) 하위 기준 행렬 찾기 ---
        for crit_name, sub_items in sub_criteria.items():
            if not sub_items:
                continue

            title = f"'{crit_name}' 그룹 내 세부 항목 비교 행렬"

            # (1) 제목 행 찾기: 있으면 좋고, 없어도 '헤더 탐색 시작 위치 힌트'로만 사용
            title_row = None
            for r_idx, v in enumerate(raw[0]):
                if str(v).strip() == title:
                    title_row = r_idx
                    break

            # (2) 헤더 행 찾기: title_row+1 고정 X
            # title이 있으면 그 아래로부터 20줄 정도만 탐색(빠르고 안전)
            # title이 없으면 시트 전체에서 탐색(느리지만 최후 수단)
            if title_row is not None:
                header_row = find_header_row_containing_all_items(
                    raw, sub_items,
                    start_row=title_row,     # 제목 포함 근처부터
                    end_row=title_row + 25   # 너무 멀리 가지 않게 제한
                )
            else:
                header_row = find_header_row_containing_all_items(raw, sub_items, start_row=0)

            if header_row is None:
                # 이 crit의 하위 행렬이 시트에 없으면 스킵
                continue

            # (3) header_row에서 sub_items가 있는 컬럼 위치 찾기
            header_vals = [str(v).strip() if not pd.isna(v) else "" for v in raw.iloc[header_row].tolist()]
            sub_col_indices = [i for i, v in enumerate(header_vals) if v in [str(x).strip() for x in sub_items]]
            if not sub_col_indices:
                continue

            first_col = sub_col_indices[0]
            last_col  = sub_col_indices[-1]
            n_sub = len(sub_items)

            # (4) 데이터는 헤더 바로 아래 n_sub행이 "행렬 데이터"일 가능성이 높음
            # 그래도 한 번 더 안전장치: 행 이름 열이 sub_items로 시작하는지 확인 가능
            sub_data_rows = raw.iloc[header_row + 1 : header_row + 1 + n_sub, :]

            sub_df = sub_data_rows.iloc[:, first_col - 1 : last_col + 1]
            sub_df.columns = ["_row"] + sub_items
            sub_df = sub_df.set_index("_row")
            sub_df = sub_df.apply(pd.to_numeric, errors="raise")

            all_sub_matrices[crit_name].append(sub_df.values)


    if not all_crit_matrices:
        raise ValueError("분석 가능한 주요 기준 행렬 데이터가 없습니다.")

    # --- 기하평균 통합 행렬 생성 ---
    group_crit_matrix = np.ones((n_crit, n_crit))
    for i in range(n_crit):
        for j in range(n_crit):
            if i != j:
                values = [m[i, j] for m in all_crit_matrices]
                group_crit_matrix[i, j] = geometric_mean(values)

    group_sub_matrices = {}
    for crit_name, sub_items in sub_criteria.items():
        matrices = all_sub_matrices[crit_name]
        if matrices:
            n_sub = len(sub_items)
            group_sub_matrix = np.ones((n_sub, n_sub))
            for i in range(n_sub):
                for j in range(n_sub):
                    if i != j:
                        values = [m[i, j] for m in matrices]
                        group_sub_matrix[i, j] = geometric_mean(values)
            group_sub_matrices[crit_name] = group_sub_matrix

    # --- 통합 행렬로 AHP 계산 ---
    crit_comparisons = {
        (crit_items[i], crit_items[j]): group_crit_matrix[i, j]
        for i, j in itertools.combinations(range(n_crit), 2)
    }
    _, crit_weights, _, _, crit_cr, _ = calculate_ahp_weights(crit_items, crit_comparisons, cr_threshold=CR_THRESHOLD_GROUP)
    crit_weights_map = {crit: float(w) for crit, w in zip(crit_items, crit_weights)}

    sub_criteria_results = {}
    final_weights = {}

    for crit, sub_crits in sub_criteria.items():
        if len(sub_crits) > 1:
            if crit not in group_sub_matrices:
                continue
            sub_matrix = group_sub_matrices[crit]
            sub_comparisons = {
                (sub_crits[i], sub_crits[j]): sub_matrix[i, j]
                for i, j in itertools.combinations(range(len(sub_crits)), 2)
            }
            _, sub_weights, _, _, sub_cr, _ = calculate_ahp_weights(sub_crits, sub_comparisons, cr_threshold=CR_THRESHOLD_GROUP)
            sub_weights_map = {sub: float(w) for sub, w in zip(sub_crits, sub_weights)}
            sub_criteria_results[crit] = {"weights": sub_weights_map, "cr": float(sub_cr)}

            for sub_crit, sub_weight in sub_weights_map.items():
                final_weights[sub_crit] = crit_weights_map[crit] * sub_weight

        elif len(sub_crits) == 1:
            sub_crit = sub_crits[0]
            sub_criteria_results[crit] = {"weights": {sub_crit: 1.0}, "cr": 0.0}
            final_weights[sub_crit] = crit_weights_map[crit] * 1.0

    sorted_final_weights = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)

    app.logger.info(
        f"[GROUP AHP] sheets={len(sheet_names)}, "
        f"crit_mats={len(all_crit_matrices)}, "
        f"sub_mats={{" + ",".join([f"{k}:{len(v)}" for k,v in all_sub_matrices.items()]) + "}}, "
        f"group_sub_keys={list(group_sub_matrices.keys())}, "
        f"final_weights_len={len(sorted_final_weights)}"
    )

    return (
        crit_weights_map, float(crit_cr),
        sub_criteria_results, sorted_final_weights,
        group_crit_matrix, group_sub_matrices
    )


@app.route('/ahp/restart')
def ahp_restart():
    # 설문 관련 세션 정리
    session.pop('respondent_info', None)
    session.pop('step1_form_data', None)
    session.pop('step2_form_data', None)
    return redirect('/ahp/')

@app.route('/ahp/group_result', methods=['GET', 'POST'])
def group_result_dashboard():
    filename = SURVEY_XLSX
    if not os.path.exists(filename):
        flash("분석할 설문 결과 파일이 없습니다.", "error")
        return redirect('/ahp/')

    xls = pd.ExcelFile(filename)
    infos = load_respondent_infos(xls)

    if not infos:
        flash("분석할 응답자 정보가 없습니다.", "error")
        return redirect('/ahp/')

    # 옵션/집계 만들기
    df_info = pd.DataFrame(infos)

    # 드롭다운 옵션 (실제 존재하는 값 기반)
    age_options = sorted([x for x in df_info["age"].dropna().unique() if str(x).strip()])
    region_options = sorted([x for x in df_info["region"].dropna().unique() if str(x).strip()])
    field_options = sorted([x for x in df_info["field"].dropna().unique() if str(x).strip()])
    exp_options = sorted([x for x in df_info["experience"].dropna().unique() if str(x).strip()])

    # 차트용 카운트
    def make_counts(col):
        c = df_info[col].fillna("").replace("", "미입력").value_counts()
        return list(c.index), list(map(int, c.values))

    age_labels, age_counts = make_counts("age")
    region_labels, region_counts = make_counts("region")
    field_labels, field_counts = make_counts("field")
    exp_labels, exp_counts = make_counts("experience")

    # 기본 선택값
    selected = {
        "age": "ALL",
        "region": "ALL",
        "field": "ALL",
        "experience": "ALL"
    }

    analysis_result = None
    download_file = None

    if request.method == 'POST':
        selected["age"] = request.form.get("age", "ALL")
        selected["region"] = request.form.get("region", "ALL")
        selected["field"] = request.form.get("field", "ALL")
        selected["experience"] = request.form.get("experience", "ALL")

        action = request.form.get("action", "analyze")

        # 필터 적용
        filtered = df_info.copy()
        if selected["age"] != "ALL":
            filtered = filtered[filtered["age"] == selected["age"]]
        if selected["region"] != "ALL":
            filtered = filtered[filtered["region"] == selected["region"]]
        if selected["field"] != "ALL":
            filtered = filtered[filtered["field"] == selected["field"]]
        if selected["experience"] != "ALL":
            filtered = filtered[filtered["experience"] == selected["experience"]]

        sheet_names = filtered["sheet"].tolist()

        if len(sheet_names) == 0:
            flash("선택한 조건에 해당하는 응답이 없습니다. 조건을 바꿔주세요.", "error")
        else:
            try:
                (
                    crit_weights_map, crit_cr,
                    sub_criteria_results, sorted_final_weights,
                    group_crit_matrix, group_sub_matrices
                ) = run_group_ahp_analysis_from_sheets(xls, sheet_names)

                analysis_result = {
                    "num": len(sheet_names),
                    "crit_weights": crit_weights_map,
                    "crit_cr": crit_cr,
                    "sub_results": sub_criteria_results,
                    "final_weights": sorted_final_weights
                }

                app.logger.info(f"[GROUP RESULT] final_weights_len={len(sorted_final_weights)}")

                if action == "export":
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    uid = uuid.uuid4().hex[:8]
                    out_name = f"group-result-{ts}-{uid}.xlsx"
                    out_path = os.path.join(DATA_DIR, out_name)

                    save_group_result_to_excel(
                        out_path=out_path,
                        num_respondents=len(sheet_names),
                        crit_matrix=group_crit_matrix, crit_items=AHP_HIERARCHY["criteria"],
                        sub_matrices=group_sub_matrices, sub_items_map=AHP_HIERARCHY["sub_criteria"],
                        final_weights=sorted_final_weights
                    )

                    download_file = out_name
                    flash("엑셀 파일로 내보내기를 완료했습니다.", "success")

            except Exception as e:
                app.logger.exception("Group analysis error")
                flash(f"그룹 분석 중 오류가 발생했습니다. ({type(e).__name__}: {e})", "error")

    return render_template(
        "group_dashboard.html",
        age_labels=age_labels, age_counts=age_counts,
        region_labels=region_labels, region_counts=region_counts,
        field_labels=field_labels, field_counts=field_counts,
        exp_labels=exp_labels, exp_counts=exp_counts,
        age_options=age_options, region_options=region_options,
        field_options=field_options, exp_options=exp_options,
        selected=selected,
        analysis_result=analysis_result,
        download_file=download_file
    )

@app.route('/ahp/group_result/download/<path:filename>')
def download_group_result(filename):
    # 안전하게 data 폴더에서만 내려주기
    return send_from_directory(DATA_DIR, filename, as_attachment=True)


if __name__ == '__main__':
    # templates 폴더가 ahp_app.py와 같은 위치에 있다고 가정합니다.
    # 만약 다른 위치에 있다면 template_folder 인자를 사용하세요.
    # 예: app = Flask(__name__, template_folder='../templates')
    app.run(host="127.0.0.1", port=5000, debug=False)


"""
AHP 쌍대비교 척도
------------------------------------------------------------------
| 중요도 | 정의                                       | 설명                                     |
|--------|--------------------------------------------|------------------------------------------|
| 1      | 두 항목이 동일하게 중요하다 (Equally important) | 두 항목이 목표에 동일하게 기여한다.        |
| 3      | 한 항목이 다른 항목보다 약간 더 중요하다 (Moderately more important) | 경험과 판단에 의해 한 항목이 약간 선호된다. |
| 5      | 한 항목이 다른 항목보다 훨씬 더 중요하다 (Strongly more important) | 경험과 판단에 의해 한 항목이 강력하게 선호된다.|
| 7      | 한 항목이 다른 항목보다 매우 강력하게 중요하다 (Very strongly more important) | 한 항목이 매우 강력하게 선호되며, 실제 데이터로 증명된다. |
| 9      | 한 항목이 다른 항목보다 절대적으로 더 중요하다 (Absolutely more important) | 한 항목이 다른 항목보다 절대적으로 선호되며, 그 증거가 명확하다. |
| 2,4,6,8| 위 인접한 판단 값 사이의 중간 값 (Intermediate values) | 타협이 필요할 때 사용된다.                 |
------------------------------------------------------------------
역수(Reciprocals): 항목 i와 j를 비교할 때, i가 j보다 3배 중요하면, j는 i보다 1/3배 중요하다.
"""
