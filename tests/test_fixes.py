import chinese_calendar as cal
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Optional, Union, List


def create_china_holidays_from_date_list(
        date_series: Optional[Union[pd.Series, pd.DatetimeIndex, List]] = None,
        include_weekends_in_holidays: bool = False
) -> pd.DataFrame:
    """
    从日期列表生成中国节假日相关的特征数据

    Args:
        date_series: 日期序列，默认为None则生成一年的数据
        include_weekends_in_holidays: 是否将周末也标记为节假日

    Returns:
        DataFrame: 包含日期特征的数据框
    """
    # 如果没有提供日期序列，则生成默认的一年数据
    if date_series is None:
        start_date = date.today().replace(month=1, day=1)
        end_date = start_date.replace(year=start_date.year + 1)
        date_range = pd.date_range(start=start_date, end=end_date - timedelta(days=1), freq='D')
    else:
        if isinstance(date_series, list):
            date_range = pd.to_datetime(date_series)
        elif isinstance(date_series, pd.DatetimeIndex):
            date_range = date_series
        elif isinstance(date_series, pd.Series):
            date_range = pd.to_datetime(date_series)
        else:
            raise ValueError("date_series must be a pandas Series, DatetimeIndex, or List")

    dates = []
    years = []
    months = []
    yearweeks = []
    yeardays = []
    weeks = []
    weekends = []
    workdays = []
    holiday_legals = []
    holiday_recesses = []
    holiday_names = []

    for dt in date_range:
        current_date = dt.date()

        # 基础时间特征
        dates.append(current_date)
        years.append(current_date.year)
        months.append(current_date.month)
        yearweeks.append(int(dt.strftime('%U')))
        yeardays.append(current_date.timetuple().tm_yday)
        weeks.append(current_date.weekday())

        # 周末判断 (0=Monday, 6=Sunday)
        is_weekend = current_date.weekday() >= 5
        weekends.append(is_weekend)

        # 使用chinese_calendar获取节假日信息
        is_workday, holiday_name = cal.get_holiday_detail(current_date)
        is_holiday = not is_workday

        # 工作日判断
        workdays.append(is_workday)

        # 法定节假日和假期标记
        if holiday_name is not None:
            holiday_legals.append(True)
            holiday_recesses.append(True)
            holiday_names.append(holiday_name.name if hasattr(holiday_name, 'name') else str(holiday_name))
        else:
            # 检查是否是法定调休工作日
            is_legal_holiday = cal.is_holiday(current_date)
            is_legal_workday = cal.is_workday(current_date)

            if is_legal_holiday:
                holiday_legals.append(True)
                holiday_recesses.append(True)
                holiday_names.append('Public Holiday')
            elif include_weekends_in_holidays and is_weekend:
                holiday_legals.append(False)
                holiday_recesses.append(True)
                holiday_names.append('Weekend')
            else:
                holiday_legals.append(False)
                holiday_recesses.append(False)
                holiday_names.append(None)

            if is_legal_workday and not is_workday:
                # 调休工作日
                workdays.append(True)

    df = pd.DataFrame({
        'date': dates,
        'year': years,
        'month': months,
        'yearweek': yearweeks,
        'yearday': yeardays,
        'week': weeks,
        'weekend': weekends,
        'workday': workdays,
        'holiday_legal': holiday_legals,
        'holiday_recess': holiday_recesses,
        'holiday_name': holiday_names
    })

    return df


# 示例用法
if __name__ == "__main__":
    # 创建一个小范围的测试数据
    test_dates = pd.date_range(start='2024-09-28', end='2024-10-10', freq='D')
    result_df = create_china_holidays_from_date_list(test_dates)
    print(result_df)



