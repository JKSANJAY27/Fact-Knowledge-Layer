from __future__ import annotations
import re
from typing import Tuple, Optional
from fact_layer.models import TemporalContext, PeriodType


def parse_numeric_value(raw_str: str) -> Optional[float]:
    """
    Extracts a numeric float value from a string, supporting negative parentheses,
    commas, currencies, and scientific notation.
    """
    if not raw_str:
        return None

    cleaned = raw_str.strip()

    # Check for negative in parentheses: (1,234.5) -> -1234.5
    is_negative = False
    paren_match = re.search(r"\(\s*([0-9.,]+)\s*\)", cleaned)
    if paren_match:
        is_negative = True
        cleaned = paren_match.group(1)
    elif cleaned.startswith("-"):
        is_negative = True
        cleaned = cleaned[1:]

    # Remove currency symbols and common non-numeric punctuation
    cleaned = re.sub(r"[₹$€£%,\s]", "", cleaned)

    # Extract valid float match
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", cleaned)
    if not match:
        return None

    try:
        val = float(match.group(0))
        return -val if is_negative else val
    except ValueError:
        return None


def normalize_unit_and_value(raw_val: str, raw_unit: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Normalizes a numeric value and unit to a canonical base:
    - INR amounts normalized to base INR (crore -> 1e7, lakh -> 1e5, million -> 1e6, billion -> 1e9)
    - USD amounts normalized to base USD
    - Percentages normalized to ratio (0.0 - 1.0) with canonical unit '%'
    - BPS normalized to percentage / ratio
    """
    base_val = parse_numeric_value(raw_val)
    if base_val is None:
        return None, raw_unit or "Unknown"

    combined_text = f"{raw_val} {raw_unit}".lower()

    # Percentage detection
    if "%" in combined_text or "percent" in combined_text:
        canonical_unit = "%"
        # normalized_value as fractional decimal (8.2% -> 0.082)
        norm_val = base_val / 100.0
        return norm_val, canonical_unit

    # Basis points (bps)
    if "bps" in combined_text or "basis point" in combined_text:
        canonical_unit = "BPS"
        # 1 bps = 0.0001
        norm_val = base_val * 0.0001
        return norm_val, canonical_unit

    # Multiplier detection
    multiplier = 1.0
    if re.search(r"\b(cr|crore|crores)\b", combined_text):
        multiplier = 1e7
    elif re.search(r"\b(lakh|lakhs|lac|lacs)\b", combined_text):
        multiplier = 1e5
    elif re.search(r"\b(billion|bn)\b", combined_text):
        multiplier = 1e9
    elif re.search(r"\b(million|mn)\b", combined_text):
        multiplier = 1e6
    elif re.search(r"\b(trillion|tn)\b", combined_text):
        multiplier = 1e12
    elif re.search(r"\b(thousand|k)\b", combined_text):
        multiplier = 1e3

    # Currency detection
    if re.search(r"(₹|inr|rs|rupee|rupees)", combined_text):
        canonical_unit = "INR"
        return base_val * multiplier, canonical_unit
    elif re.search(r"(\$|usd|dollar|dollars)", combined_text):
        canonical_unit = "USD"
        return base_val * multiplier, canonical_unit
    elif re.search(r"(€|eur|euro|euros)", combined_text):
        canonical_unit = "EUR"
        return base_val * multiplier, canonical_unit

    # Counts / metrics
    if multiplier != 1.0:
        return base_val * multiplier, raw_unit.strip()

    return base_val, raw_unit.strip() or "Standard"


def normalize_temporal_expression(raw_expr: str) -> TemporalContext:
    """
    Normalizes complex temporal expressions to standard ISO periods and canonical names.
    Examples:
      - 'FY2024', 'FY 2023-24', 'year ended March 31 2024' -> FY2024 (2023-04-01 to 2024-03-31)
      - 'Q4 FY24', 'quarter ended March 31, 2024' -> Q4_FY2024 (2024-01-01 to 2024-03-31)
      - 'As of March 31, 2024' -> 2024-03-31 (point)
    """
    if not raw_expr:
        return TemporalContext(
            period_type="unknown",
            canonical_period="Unknown",
            raw_period_text=raw_expr
        )

    expr = raw_expr.strip().lower()

    # 1. Quarter patterns: Q1, Q2, Q3, Q4 with fiscal year
    # Example: 'q4 fy24', 'q4 2024', 'q4 fy2024'
    q_match = re.search(r"q([1-4])\s*(?:fy|financial year)?\s*([0-9]{2,4})", expr)
    if q_match:
        quarter = int(q_match.group(1))
        yr_str = q_match.group(2)
        year = int(yr_str) if len(yr_str) == 4 else 2000 + int(yr_str)

        # In Indian fiscal year:
        # FY2024 begins April 1, 2023 and ends March 31, 2024.
        # Q1 FY2024: Apr 1, 2023 - Jun 30, 2023
        # Q2 FY2024: Jul 1, 2023 - Sep 30, 2023
        # Q3 FY2024: Oct 1, 2023 - Dec 31, 2023
        # Q4 FY2024: Jan 1, 2024 - Mar 31, 2024
        prev_year = year - 1
        q_dates = {
            1: (f"{prev_year}-04-01", f"{prev_year}-06-30"),
            2: (f"{prev_year}-07-01", f"{prev_year}-09-30"),
            3: (f"{prev_year}-10-01", f"{prev_year}-12-31"),
            4: (f"{year}-01-01", f"{year}-03-31")
        }
        start_d, end_d = q_dates[quarter]
        return TemporalContext(
            period_type="quarter",
            start_date=start_d,
            end_date=end_d,
            canonical_period=f"Q{quarter}_FY{year}",
            raw_period_text=raw_expr
        )

    # 2. 'Quarter ended March 31, 2024' / 'Three months ended 31 March 2024'
    if ("quarter ended" in expr or "three months ended" in expr) and re.search(r"march\s*(?:31|31st|31th),?\s*([0-9]{4})", expr):
        y_match = re.search(r"march\s*(?:31|31st|31th),?\s*([0-9]{4})", expr)
        year = int(y_match.group(1))
        return TemporalContext(
            period_type="quarter",
            start_date=f"{year}-01-01",
            end_date=f"{year}-03-31",
            canonical_period=f"Q4_FY{year}",
            raw_period_text=raw_expr
        )
    if ("quarter ended" in expr or "three months ended" in expr) and re.search(r"december\s*(?:31|31st),?\s*([0-9]{4})", expr):
        y_match = re.search(r"december\s*(?:31|31st),?\s*([0-9]{4})", expr)
        year = int(y_match.group(1))
        fy = year + 1
        return TemporalContext(
            period_type="quarter",
            start_date=f"{year}-10-01",
            end_date=f"{year}-12-31",
            canonical_period=f"Q3_FY{fy}",
            raw_period_text=raw_expr
        )

    # 3. 'Year ended March 31 2024' / '12 months ended 31st March 2024'
    ended_match = re.search(r"(?:year|12 months|twelve months)\s*ended\s*(?:on\s*)?(?:march\s*31(?:st)?|31(?:st)?\s*march),?\s*([0-9]{4})", expr)
    if ended_match:
        year = int(ended_match.group(1))
        return TemporalContext(
            period_type="fiscal_year",
            start_date=f"{year-1}-04-01",
            end_date=f"{year}-03-31",
            canonical_period=f"FY{year}",
            raw_period_text=raw_expr
        )

    # 4. Standard Fiscal Year strings: 'FY2024', 'FY 2023-24', '2023-24', 'FY24'
    fy_split_match = re.search(r"(?:fy|financial year)?\s*([0-9]{4})\s*[-–/]\s*([0-9]{2,4})", expr)
    if fy_split_match:
        start_y = int(fy_split_match.group(1))
        end_str = fy_split_match.group(2)
        end_y = int(end_str) if len(end_str) == 4 else (start_y // 100) * 100 + int(end_str)
        return TemporalContext(
            period_type="fiscal_year",
            start_date=f"{start_y}-04-01",
            end_date=f"{end_y}-03-31",
            canonical_period=f"FY{end_y}",
            raw_period_text=raw_expr
        )

    fy_single_match = re.search(r"\bfy\s*([0-9]{2,4})\b", expr)
    if fy_single_match:
        yr_str = fy_single_match.group(1)
        year = int(yr_str) if len(yr_str) == 4 else 2000 + int(yr_str)
        return TemporalContext(
            period_type="fiscal_year",
            start_date=f"{year-1}-04-01",
            end_date=f"{year}-03-31",
            canonical_period=f"FY{year}",
            raw_period_text=raw_expr
        )

    # 5. Point in time: 'As of March 31, 2024' or 'At 31 March 2024'
    as_of_match = re.search(r"(?:as of|as on|at)\s*(?:march\s*31st?|31st?\s*march),?\s*([0-9]{4})", expr)
    if as_of_match:
        year = int(as_of_match.group(1))
        return TemporalContext(
            period_type="point",
            start_date=f"{year}-03-31",
            end_date=f"{year}-03-31",
            canonical_period=f"{year}-03-31",
            raw_period_text=raw_expr
        )

    # Specific date format YYYY-MM-DD
    date_match = re.search(r"\b([12][0-9]{3})-([01][0-9])-([0-3][0-9])\b", expr)
    if date_match:
        d_str = date_match.group(0)
        return TemporalContext(
            period_type="point",
            start_date=d_str,
            end_date=d_str,
            canonical_period=d_str,
            raw_period_text=raw_expr
        )

    # 6. Standalone 4-digit year: '2024'
    year_match = re.search(r"\b(19[89][0-9]|20[0-3][0-9])\b", expr)
    if year_match:
        year = int(year_match.group(1))
        # If institutional report context, often represents FY or CY.
        return TemporalContext(
            period_type="calendar_year",
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
            canonical_period=f"CY{year}",
            raw_period_text=raw_expr
        )

    # Fallback to cleaned text
    cleaned_period = re.sub(r"[^a-zA-Z0-9_\-]", "", raw_expr.strip().upper())
    return TemporalContext(
        period_type="unknown",
        canonical_period=cleaned_period or "Unknown",
        raw_period_text=raw_expr
    )
