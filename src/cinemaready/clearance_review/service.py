from collections import Counter
from math import isfinite, sqrt
from statistics import mean

from ..show_handoffs import service as intake_service
from ..show_handoffs.contracts import CinemareadyRecord
from .contracts import (
    RAW_FIELDS,
    ReviewRow,
    ReviewSheet,
    ReviewSummary,
    SealedForecast,
    STATUS_CONFLICT,
    STATUS_INTAKE_ISOLATED,
    STATUS_NO_FORECAST,
    STATUS_PENDING,
    STATUS_READING_CHANGED,
    STATUS_SCORED,
    REASON_CONFLICT,
    REASON_INTAKE,
    REASON_MATURED,
    REASON_PENDING,
    REASON_READING_CHANGED,
    REASON_UNKNOWN,
)


def seal_forecasts(assessments, records, receipt, feature_stamp, model_name, calibration):
    """把签发时的预测、校准区间和工位凭证冻结成清场估时复核底账。

    assessments 为当班单据经过同一校准后的评估结果，records 为入站采信的
    当班原始单据（含量测读数与版次），receipt 为开厅交接签发凭证。
    """
    readings_by_key = {row.key: row for row in records}
    if len(readings_by_key) != len(records):
        raise ArithmeticError('入站单据单号不唯一，无法冻结复核底账')
    sealed = []
    for assessment in assessments:
        ticket = readings_by_key.get(assessment.raw.key)
        if ticket is None:
            raise ArithmeticError('签发评估与入站单据无法一一配对')
        sealed.append(SealedForecast(
            key=ticket.key,
            cohort=ticket.cohort,
            revision=ticket.revision,
            readings=tuple(getattr(ticket, name) for name in RAW_FIELDS),
            estimate=assessment.estimate,
            lower=assessment.lower,
            upper=assessment.upper,
            uncertain=assessment.uncertain,
            run_id=receipt.stamp,
            feature_stamp=feature_stamp,
            model=model_name,
            calibration_method=calibration.method,
            calibration_size=calibration.size,
        ))
    return tuple(sorted(sealed, key=lambda item: item.key))


def _conflict_keys(issues):
    """同版冲突会隔离该单号的全部版次，复核单需要撤下这些单的已记误差。"""
    return frozenset(issue.key for issue in issues if issue.code == 'CONFLICTING_REVISION')


def _intake_lookup(issues):
    reasons = {}
    for issue in issues:
        reasons.setdefault(issue.key, []).append(issue.code)
    return reasons


def _reading_signature(record):
    return tuple(getattr(record, name) for name in RAW_FIELDS)


def _build_sealed_row(forecast, current, conflicted):
    """对每一张签发时的预测，按补录批次的最新状态给出复核结论。"""
    if conflicted:
        # 同版记录互相冲突时整单隔离（没有可采信的当前记录），撤下已记
        # 误差；澄清并换版后的后续批次会让该单重新进入入评分支。
        return ReviewRow(
            key=forecast.key, cohort=forecast.cohort, status=STATUS_CONFLICT,
            reason=REASON_CONFLICT, sealed_revision=forecast.revision,
            current_revision=None if current is None else current.revision,
            estimate=forecast.estimate, lower=forecast.lower, upper=forecast.upper,
            outcome=None, result_day=None if current is None else current.result_day,
            absolute_error=None, error_denominator=None, interval_hit=None,
            reading_changed=False, run_id=forecast.run_id,
            note='同版记录读数或实绩冲突，撤下已记误差，澄清版次后恢复',
        )
    if current is None:
        return ReviewRow(
            key=forecast.key, cohort=forecast.cohort, status=STATUS_PENDING,
            reason=REASON_PENDING, sealed_revision=forecast.revision, current_revision=None,
            estimate=forecast.estimate, lower=forecast.lower, upper=forecast.upper,
            outcome=None, result_day=None, absolute_error=None, error_denominator=None,
            interval_hit=None, reading_changed=False, run_id=forecast.run_id,
            note='完工实绩尚未到站，原预测保留计分资格',
        )
    if _reading_signature(current) != forecast.readings:
        # 回执改写了签发时的读数：旧预测依据的量测已不存在，退出计分。
        return ReviewRow(
            key=forecast.key, cohort=forecast.cohort, status=STATUS_READING_CHANGED,
            reason=REASON_READING_CHANGED, sealed_revision=forecast.revision,
            current_revision=current.revision, estimate=forecast.estimate,
            lower=forecast.lower, upper=forecast.upper, outcome=current.outcome,
            result_day=current.result_day, absolute_error=None, error_denominator=None,
            interval_hit=None, reading_changed=True, run_id=forecast.run_id,
            note='补录回执改写签发读数，旧预测退出计分，待重新签发',
        )
    if current.outcome is None or not current.observed:
        # 留架右截尾或实绩未成熟：晚到记录确认完工后才能入评。
        return ReviewRow(
            key=forecast.key, cohort=forecast.cohort, status=STATUS_PENDING,
            reason=REASON_PENDING, sealed_revision=forecast.revision,
            current_revision=current.revision, estimate=forecast.estimate,
            lower=forecast.lower, upper=forecast.upper, outcome=current.outcome,
            result_day=current.result_day, absolute_error=None, error_denominator=None,
            interval_hit=None, reading_changed=False, run_id=forecast.run_id,
            note='实绩尚未确认完工，继续等待晚到完工记录',
        )
    if not isfinite(forecast.estimate) or not isfinite(current.outcome):
        raise ValueError('复核计分要求有限的预测与实绩')
    denominator = _error_denominator(forecast, current)
    error = abs(forecast.estimate - current.outcome)
    hit = None
    if forecast.lower is not None:
        hit = forecast.lower <= current.outcome <= forecast.upper
    return ReviewRow(
        key=forecast.key, cohort=forecast.cohort, status=STATUS_SCORED,
        reason=REASON_MATURED, sealed_revision=forecast.revision,
        current_revision=current.revision, estimate=forecast.estimate,
        lower=forecast.lower, upper=forecast.upper, outcome=current.outcome,
        result_day=current.result_day, absolute_error=error,
        error_denominator=denominator, interval_hit=hit, reading_changed=False,
        run_id=forecast.run_id,
        note='晚到完工实绩已采信，按签发预测与校准区间对账',
    )


def _error_denominator(forecast, current):
    """复用校准时的物理残差尺度，保证复核误差分母与区间口径一致。"""
    values = tuple(
        getattr(current, name) if getattr(current, name) is not None else reading
        for name, reading in zip(RAW_FIELDS, forecast.readings, strict=True)
    )
    minutes = values[3]
    return max(3.0, minutes / 5)


def _unknown_row(record, issue_codes):
    unique_codes = tuple(dict.fromkeys(issue_codes))
    if not unique_codes:
        return ReviewRow(
            key=record.key, cohort=record.cohort, status=STATUS_NO_FORECAST,
            reason=REASON_UNKNOWN, sealed_revision=None, current_revision=record.revision,
            estimate=None, lower=None, upper=None, outcome=record.outcome,
            result_day=record.result_day, absolute_error=None, error_denominator=None,
            interval_hit=None, reading_changed=False, run_id=None,
            note='补录单号不在签发清册内，复核单不代签预测',
        )
    if REASON_CONFLICT in unique_codes:
        # 同版冲突在入站阶段已整单隔离，没有可采信的读数与实绩，先撤记待澄清。
        return ReviewRow(
            key=record.key, cohort=record.cohort, status=STATUS_CONFLICT,
            reason=REASON_CONFLICT, sealed_revision=None, current_revision=record.revision,
            estimate=None, lower=None, upper=None, outcome=None,
            result_day=record.result_day, absolute_error=None, error_denominator=None,
            interval_hit=None, reading_changed=False, run_id=None,
            note='同版记录读数冲突，整单隔离，澄清版次后恢复计分',
        )
    return ReviewRow(
        key=record.key, cohort=record.cohort, status=STATUS_INTAKE_ISOLATED,
        reason=REASON_INTAKE, sealed_revision=None, current_revision=record.revision,
        estimate=None, lower=None, upper=None, outcome=record.outcome,
        result_day=record.result_day, absolute_error=None, error_denominator=None,
        interval_hit=None, reading_changed=False, run_id=None,
        note='补录记录入站隔离：' + '、'.join(unique_codes),
    )


def build_review_sheet(sealed, backfill, as_of, *, intake_issues=()):
    """用补录批次对签发底账做清场估时复核。

    sealed 为签发时冻结的预测底账；backfill 为本次到站的全部原始单据
    （含晚到实绩与换版回执），入站校验复用第一环节，保证补录期间原模型、
    校准和工位凭证不被重算。intake_issues 为本批入站已登记的隔离问题
    （例如同版冲突在签发前已被整单隔离），复核单同样留痕撤记。
    同批复跑为纯函数，重复提交不重复计分。
    """
    if not sealed:
        raise ValueError('复核单需要至少一条签发预测')
    intake = intake_service.load_samples(backfill, as_of)
    all_issues = (*intake.issues, *intake_issues)
    conflicted = _conflict_keys(all_issues)
    intake_reasons = _intake_lookup(all_issues)
    current_by_key = {row.key: row for row in intake.entries}

    rows = []
    for forecast in sealed:
        current = current_by_key.get(forecast.key)
        # 即使最新版次被采信，同版冲突（全部版次隔离）仍须撤记。
        rows.append(_build_sealed_row(forecast, current, forecast.key in conflicted))

    sealed_keys = {item.key for item in sealed}
    for record in intake.entries:
        if record.key not in sealed_keys:
            rows.append(_unknown_row(record, ()))
    # 入站被拒的单号若不在签发清册内，也单独留痕，保证行可对账。
    known_rejected = set(sealed_keys) | set(current_by_key)
    rejected_seen = set()
    for issue in all_issues:
        if issue.key in known_rejected or issue.key in rejected_seen or not issue.key:
            continue
        rejected_seen.add(issue.key)
        placeholder = CinemareadyRecord(
            issue.key, '', 0, as_of, 1, None, None, True,
            None, None, None, None, None, None,
        )
        rows.append(_unknown_row(placeholder, intake_reasons[issue.key]))

    ordered = tuple(sorted(rows, key=lambda row: (row.status != STATUS_SCORED, row.key)))
    summary = _summarize(sealed, ordered)
    anchor = sealed[0]
    return ReviewSheet(
        run_id=anchor.run_id,
        feature_stamp=anchor.feature_stamp,
        model=anchor.model,
        calibration_method=anchor.calibration_method,
        rows=ordered,
        summary=summary,
    )


def _summarize(sealed, rows):
    counts = Counter(row.status for row in rows)
    scored = [row for row in rows if row.status == STATUS_SCORED]
    if scored:
        mae = mean(row.absolute_error for row in scored)
        rmse = sqrt(mean(row.absolute_error ** 2 for row in scored))
        covered = [row.interval_hit for row in scored if row.interval_hit is not None]
        coverage = mean(covered) if covered else None
    else:
        mae = rmse = coverage = None
    return ReviewSummary(
        sealed=len(sealed),
        scored=counts.get(STATUS_SCORED, 0),
        pending=counts.get(STATUS_PENDING, 0),
        withdrawn=counts.get(STATUS_READING_CHANGED, 0),
        conflicts=counts.get(STATUS_CONFLICT, 0),
        intake_rejected=counts.get(STATUS_INTAKE_ISOLATED, 0),
        unknown=counts.get(STATUS_NO_FORECAST, 0),
        mae=mae,
        rmse=rmse,
        coverage=coverage,
        scored_keys=tuple(row.key for row in scored),
        withdrawn_keys=tuple(row.key for row in rows if row.status == STATUS_READING_CHANGED),
        conflict_keys=tuple(row.key for row in rows if row.status == STATUS_CONFLICT),
    )


def review_evidence(sheet):
    """复核单明细表：原预测、采信实绩和误差分母逐行可查。"""
    return [row.evidence() for row in sheet.rows]


def review_metrics(sheet):
    return sheet.summary.evidence()
