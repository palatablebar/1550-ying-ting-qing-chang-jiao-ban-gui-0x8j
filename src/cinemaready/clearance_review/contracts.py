from dataclasses import dataclass

# 复核单上每一行的处置状态。
# 入评：晚到实绩已确认完工，可以与签发时的预测对账计分。
STATUS_SCORED = '入评'
# 待实绩：签发后尚无成熟完工实绩，预测保留、暂不计分。
STATUS_PENDING = '待实绩'
# 读数变更退评：补录回执改写了签发时的量测读数，旧预测退出计分。
STATUS_READING_CHANGED = '读数变更退评'
# 同版实绩冲突：同一版次出现互相冲突的记录，撤下已记误差，澄清（换版）后恢复。
STATUS_CONFLICT = '同版实绩冲突'
# 入站隔离：补录批次未通过入站校验，不能采信。
STATUS_INTAKE_ISOLATED = '入站隔离'
# 无原预测：补录批次中出现签发清册之外的单号，复核单不代签预测。
STATUS_NO_FORECAST = '无原预测'

# 复核单原因码。
REASON_MATURED = 'LATE_RESULT_ACCEPTED'
REASON_PENDING = 'AWAITING_RESULT'
REASON_READING_CHANGED = 'SEALED_READING_OVERWRITTEN'
REASON_CONFLICT = 'CONFLICTING_REVISION'
REASON_INTAKE = 'INTAKE_REJECTED'
REASON_UNKNOWN = 'FORECAST_NOT_SEALED'

RAW_FIELDS = ('seats', 'filled', 'spill', 'minutes', 'bins', 'stairs')


@dataclass(frozen=True, slots=True)
class SealedForecast:
    """开厅交接签发时冻结的单场预测与校准区间。"""

    key: str
    cohort: str
    revision: int
    readings: tuple[float | None, ...]
    estimate: float
    lower: float | None
    upper: float | None
    uncertain: bool
    run_id: str
    feature_stamp: str
    model: str
    calibration_method: str
    calibration_size: int


@dataclass(frozen=True, slots=True)
class ReviewRow:
    """清场估时复核单上的一行对账记录。"""

    key: str
    cohort: str
    status: str
    reason: str
    sealed_revision: int | None
    current_revision: int | None
    estimate: float | None
    lower: float | None
    upper: float | None
    outcome: float | None
    result_day: int | None
    absolute_error: float | None
    error_denominator: float | None
    interval_hit: bool | None
    reading_changed: bool
    run_id: str | None
    note: str

    def evidence(self):
        return {
            '单号': self.key,
            '作业区': self.cohort,
            '复核状态': self.status,
            '原因码': self.reason,
            '签发版次': self.sealed_revision if self.sealed_revision is not None else '—',
            '当前版次': self.current_revision if self.current_revision is not None else '—',
            '原预测': None if self.estimate is None else round(self.estimate, 4),
            '校准区间': ('—' if self.lower is None else f'{self.lower:.2f}–{self.upper:.2f}'),
            '采信实绩': self.outcome,
            '完工工日': self.result_day if self.result_day is not None else '—',
            '绝对误差': None if self.absolute_error is None else round(self.absolute_error, 4),
            '误差分母': None if self.error_denominator is None else round(self.error_denominator, 4),
            '区间命中': self.interval_hit,
            '读数变更': '是' if self.reading_changed else '否',
            '签发凭证': self.run_id if self.run_id is not None else '—',
            '说明': self.note,
        }


@dataclass(frozen=True, slots=True)
class ReviewSummary:
    """复核单的计分汇总，误差分母逐条可回溯。"""

    sealed: int
    scored: int
    pending: int
    withdrawn: int
    conflicts: int
    intake_rejected: int
    unknown: int
    mae: float | None
    rmse: float | None
    coverage: float | None
    scored_keys: tuple[str, ...]
    withdrawn_keys: tuple[str, ...]
    conflict_keys: tuple[str, ...]

    def evidence(self):
        rows = [
            {'核对项': '签发预测', '数量': self.sealed},
            {'核对项': '入评实绩', '数量': self.scored},
            {'核对项': '待实绩', '数量': self.pending},
            {'核对项': '读数变更退评', '数量': self.withdrawn},
            {'核对项': '同版冲突撤记', '数量': self.conflicts},
            {'核对项': '入站隔离', '数量': self.intake_rejected},
            {'核对项': '无原预测', '数量': self.unknown},
            {'核对项': '入评平均绝对误差', '数量': None if self.mae is None else round(self.mae, 4)},
            {'核对项': '入评均方根误差', '数量': None if self.rmse is None else round(self.rmse, 4)},
            {'核对项': '校准区间覆盖率', '数量': None if self.coverage is None else round(self.coverage, 4)},
        ]
        return rows


@dataclass(frozen=True, slots=True)
class ReviewSheet:
    run_id: str
    feature_stamp: str
    model: str
    calibration_method: str
    rows: tuple[ReviewRow, ...]
    summary: ReviewSummary

    def scored_rows(self):
        return tuple(row for row in self.rows if row.status == STATUS_SCORED)
