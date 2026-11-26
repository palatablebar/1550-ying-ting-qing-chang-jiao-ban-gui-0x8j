"""清场估时复核单（M9）测试。

覆盖签发后复核的六条业务规则：
1. 晚到实绩确认完工后入评；
2. 回执改写签发读数时旧预测退出计分；
3. 同版实绩冲突撤下已记误差，澄清换版后恢复；
4. 补录期间原模型、校准和工位凭证保持冻结；
5. 同批复跑不重复计分（纯函数幂等）；
6. 复核单逐行可查原预测、采信实绩与误差分母。
"""
import pytest

from cinemaready.pipeline import open_fixture, ShiftDossier, preview, execute
from cinemaready.usher_windows import service as dispatch_service
from cinemaready.clearance_review.service import build_review_sheet, seal_forecasts
from cinemaready.clearance_review.contracts import (
    STATUS_CONFLICT,
    STATUS_NO_FORECAST,
    STATUS_PENDING,
    STATUS_READING_CHANGED,
    STATUS_SCORED,
)

REVIEW_DAY = 98


@pytest.fixture(scope='module')
def sealed_bundle():
    document = open_fixture()
    records = [dict(row) for row in document['records']]
    dossier = ShiftDossier(records, document['as_of'])
    calibration, estimates = dossier.grading()
    _, checked_live, drift = dossier.verification(estimates, calibration)
    plan = dispatch_service.plan_jobs(checked_live, drift.domain, 12, 'balanced')
    receipt = dossier.seal(plan, 'balanced')
    sealed = dossier.sealed_forecasts(checked_live, calibration, receipt)
    raw_target = next(row for row in records if row['key'] == sealed[0].key)
    return dossier, calibration, receipt, sealed, raw_target, records


def late_record(raw_target, **overrides):
    row = dict(raw_target)
    row.update(received=97, result_day=97, outcome=12.5, observed=True)
    row.update(overrides)
    return row


def test_late_result_enters_scoring_after_completion(sealed_bundle):
    dossier, _cal, receipt, sealed, raw_target, _records = sealed_bundle
    sheet = build_review_sheet(sealed, [late_record(raw_target)], REVIEW_DAY)
    row = next(item for item in sheet.rows if item.key == raw_target['key'])
    assert row.status == STATUS_SCORED
    assert row.outcome == 12.5 and row.absolute_error is not None
    assert row.error_denominator == pytest.approx(max(3.0, raw_target['minutes'] / 5))
    assert row.run_id == receipt.stamp
    assert sheet.summary.scored == 1
    assert sheet.summary.mae == pytest.approx(row.absolute_error)


def test_review_sheet_is_idempotent_for_same_backfill(sealed_bundle):
    _d, _c, _r, sealed, raw_target, _records = sealed_bundle
    backfill = [late_record(raw_target)]
    first = build_review_sheet(sealed, backfill, REVIEW_DAY)
    second = build_review_sheet(sealed, backfill, REVIEW_DAY)
    assert [row.evidence() for row in first.rows] == [row.evidence() for row in second.rows]
    assert first.summary.scored == second.summary.scored == 1


def test_backfill_order_does_not_change_review(sealed_bundle):
    _d, _c, _r, sealed, raw_target, records = sealed_bundle
    another_key = next(item.key for item in sealed if item.key != raw_target['key'])
    another_raw = next(row for row in records if row['key'] == another_key)
    first_batch = [late_record(raw_target), late_record(another_raw, outcome=9.0)]
    forward = build_review_sheet(sealed, first_batch, REVIEW_DAY)
    backward = build_review_sheet(sealed, list(reversed(first_batch)), REVIEW_DAY)
    assert [row.evidence() for row in forward.rows] == [row.evidence() for row in backward.rows]


def test_overwritten_sealed_reading_withdraws_old_forecast(sealed_bundle):
    _d, _c, _r, sealed, raw_target, _records = sealed_bundle
    changed = late_record(raw_target, seats=raw_target['seats'] + 10)
    sheet = build_review_sheet(sealed, [changed], REVIEW_DAY)
    row = next(item for item in sheet.rows if item.key == raw_target['key'])
    assert row.status == STATUS_READING_CHANGED
    assert row.reading_changed is True and row.absolute_error is None
    assert sheet.summary.scored == 0 and sheet.summary.withdrawn == 1
    # 即便换版澄清，读数已不是签发时读数，旧预测仍然退评。
    revised = late_record(raw_target, seats=raw_target['seats'] + 10, revision=2)
    sheet_revised = build_review_sheet(sealed, [revised], REVIEW_DAY)
    assert next(item for item in sheet_revised.rows if item.key == raw_target['key']).status == STATUS_READING_CHANGED


def test_conflicting_same_version_results_withdraw_recorded_error(sealed_bundle):
    _d, _c, _r, sealed, raw_target, _records = sealed_bundle
    conflict_a = late_record(raw_target, outcome=12.5)
    conflict_b = late_record(raw_target, outcome=13.0)
    sheet = build_review_sheet(sealed, [conflict_a, conflict_b], REVIEW_DAY)
    row = next(item for item in sheet.rows if item.key == raw_target['key'])
    assert row.status == STATUS_CONFLICT and row.absolute_error is None
    assert sheet.summary.scored == 0 and sheet.summary.conflicts == 1
    assert raw_target['key'] in sheet.summary.conflict_keys


def test_clarified_revision_restores_scoring(sealed_bundle):
    _d, _c, _r, sealed, raw_target, _records = sealed_bundle
    conflict_a = late_record(raw_target, outcome=12.5)
    conflict_b = late_record(raw_target, outcome=13.0)
    clarified = late_record(raw_target, revision=2)
    sheet = build_review_sheet(sealed, [conflict_a, conflict_b, clarified], REVIEW_DAY)
    row = next(item for item in sheet.rows if item.key == raw_target['key'])
    assert row.status == STATUS_SCORED and row.current_revision == 2
    assert row.absolute_error is not None
    assert sheet.summary.scored == 1 and sheet.summary.conflicts == 0


def test_frozen_model_calibration_and_workstation_credentials(sealed_bundle):
    dossier, calibration, receipt, sealed, raw_target, _records = sealed_bundle
    sheet = build_review_sheet(sealed, [late_record(raw_target)], REVIEW_DAY)
    assert sheet.feature_stamp == dossier.feature_stamp.fingerprint
    assert sheet.model == dossier.forecaster.name
    assert sheet.calibration_method == calibration.method
    assert sheet.run_id == receipt.stamp


def test_censored_or_missing_result_waits_for_completion(sealed_bundle):
    _d, _c, _r, sealed, raw_target, _records = sealed_bundle
    censored = late_record(raw_target, observed=False, outcome=0.0)
    sheet = build_review_sheet(sealed, [censored], REVIEW_DAY)
    row = next(item for item in sheet.rows if item.key == raw_target['key'])
    assert row.status == STATUS_PENDING and sheet.summary.scored == 0
    missing = late_record(raw_target, outcome=None, result_day=None, observed=True)
    sheet_missing = build_review_sheet(sealed, [missing], REVIEW_DAY)
    assert next(item for item in sheet_missing.rows if item.key == raw_target['key']).status == STATUS_PENDING


def test_unknown_backfill_key_gets_no_substitute_forecast(sealed_bundle):
    _d, _c, _r, sealed, raw_target, _records = sealed_bundle
    unknown = late_record(raw_target, key='CH-NEW-999', revision=1)
    sheet = build_review_sheet(sealed, [unknown], REVIEW_DAY)
    row = next(item for item in sheet.rows if item.key == 'CH-NEW-999')
    assert row.status == STATUS_NO_FORECAST and row.estimate is None
    assert sheet.summary.unknown == 1


def test_intake_rejected_backfill_is_listed_not_scored(sealed_bundle):
    _d, _c, _r, sealed, raw_target, _records = sealed_bundle
    broken = late_record(raw_target, seats='not-a-number')
    sheet = build_review_sheet(sealed, [broken], REVIEW_DAY)
    row = next(item for item in sheet.rows if item.key == raw_target['key'])
    # 补录被入站拒绝时，原封预测保持待实绩，不会产生任何计分。
    assert row.status == STATUS_PENDING and row.absolute_error is None
    assert sheet.summary.scored == 0
    assert sheet.summary.intake_rejected == 0  # 密封单不重复计入入站隔离行


def test_review_row_exposes_forecast_actual_and_error_denominator(sealed_bundle):
    _d, _c, _r, sealed, raw_target, _records = sealed_bundle
    sheet = build_review_sheet(sealed, [late_record(raw_target)], REVIEW_DAY)
    evidence = next(item for item in sheet.rows if item.key == raw_target['key']).evidence()
    assert evidence['原预测'] is not None
    assert evidence['校准区间'] != '—'
    assert evidence['采信实绩'] == 12.5
    assert evidence['误差分母'] is not None and evidence['绝对误差'] is not None
    assert evidence['签发凭证']


def test_sealed_forecast_must_pair_with_intake_records(sealed_bundle):
    dossier, calibration, receipt, sealed, _raw, _records = sealed_bundle
    # 抽掉一条评估结果，冻结时按评估结果回查入站单据会找不到配对。
    _, estimates = dossier.grading()
    _, checked_live, _ = dossier.verification(estimates, calibration)
    partial_assessments = tuple(item for item in checked_live if item.raw.key != sealed[-1].key)
    # 用与评估结果不一致的当班单据集合触发配对失败。
    dropped = tuple(row for row in dossier.history.live if row.key != sealed[0].key)
    with pytest.raises(ArithmeticError, match='配对'):
        seal_forecasts(partial_assessments, dropped,
                       receipt, dossier.feature_stamp.fingerprint,
                       dossier.forecaster.name, calibration)
    # 复核底账为空时拒绝出单。
    with pytest.raises(ValueError):
        build_review_sheet((), [{}], REVIEW_DAY)


def test_preview_review_sheet_conserves_sealed_ledger():
    report = preview()
    review = report['clearance_review']
    sealed_total = next(item['数量'] for item in review['summary'] if item['核对项'] == '签发预测')
    assert sealed_total == len(report['rows'])
    statuses = [row['复核状态'] for row in review['rows']]
    assert '入评' not in statuses  # 样例当班单尚无完工实绩
    assert any(status == '同版实绩冲突' for status in statuses)


def test_review_section_is_available_per_stage_route():
    report = preview()
    section = report['sections'][8]
    assert section['key'] == 'clearance_review'
    # 环节页 records 同时包含汇总指标与逐行明细，前端按键集合分组渲染。
    assert '复核凭证' in section['summary']
    detail_rows = report['clearance_review']['rows']
    assert detail_rows, '复核单逐行明细不能为空'
    assert section['records'] == [*report['clearance_review']['summary'], *detail_rows]


def test_backfill_through_execute_keeps_base_predictions_frozen():
    document = open_fixture()
    base = execute([dict(row) for row in document['records']], as_of=document['as_of'])
    # 补录晚到实绩只改变复核单，不改变签发预测、模型凭证和 run_id。
    live = next(row for row in document['records'] if row['day'] >= 88 and row['key'] in
                {item['key'] for item in base['rows']})
    updated = dict(live)
    updated.update(received=97, result_day=97, outcome=12.5, observed=True)
    backfill = [updated]
    reviewed = execute([dict(row) for row in document['records']],
                       as_of=document['as_of'], backfill=backfill, review_as_of=98)
    assert reviewed['feature_stamp'] == base['feature_stamp']
    assert reviewed['rows'] == base['rows']
    assert reviewed['run_id'] == base['run_id']
    target = next(row for row in reviewed['clearance_review']['rows'] if row['单号'] == live['key'])
    assert target['复核状态'] == '入评' and target['采信实绩'] == 12.5


def test_backfill_is_rejected_without_sealing_records():
    from cinemaready.entry import app
    response = app.test_client().post(
        '/api/cinemaready/run', json={'backfill': [{'key': 'X'}]})
    assert response.status_code == 422
    assert '签发批次' in response.get_json()['message']


def test_review_day_cannot_precede_sealing_cutoff():
    document = open_fixture()
    with pytest.raises(ValueError, match='复核工日'):
        execute([dict(row) for row in document['records']],
                as_of=document['as_of'], review_as_of=document['as_of'] - 1)


def test_api_accepts_backfill_and_returns_review_sheet():
    from cinemaready.entry import app
    document = open_fixture()
    live = next(row for row in document['records'] if row['day'] >= 88)
    updated = dict(live)
    updated.update(received=97, result_day=97, outcome=12.5, observed=True)
    response = app.test_client().post('/api/cinemaready/run', json={
        'records': [dict(row) for row in document['records']],
        'as_of': document['as_of'], 'review_as_of': 98, 'backfill': [updated],
    })
    assert response.status_code == 200
    target = next(row for row in response.get_json()['clearance_review']['rows']
                  if row['单号'] == live['key'])
    assert target['复核状态'] == '入评'
