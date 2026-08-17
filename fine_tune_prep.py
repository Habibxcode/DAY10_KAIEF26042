"""
fine_tune_prep.py — Part C: Customer Support Dataset for SFT Fine-Tuning
=========================================================================
Converts a CSV customer-support dataset into two industry-standard JSONL
formats compatible with Hugging Face SFTTrainer, Llama-3, and Mistral.

Output formats:
  1. ChatML / Llama-3 Conversational JSONL  → sft_training_data_chatml.jsonl
  2. Alpaca Instruction-Input-Output JSONL  → sft_training_data_alpaca.jsonl

Pipeline:
  CSV Ingestion → Validation & Sanitisation → Format Conversion
  → Statistics Report → JSONL Output → Preview

Author : Production AI Systems – Day 10
PEP 8  : Compliant
"""

from __future__ import annotations

import sys

# ── Windows UTF-8 terminal fix ────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import csv
import io
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd

# ── Logging configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
CSV_FILE = Path("customer_support.csv")
CHATML_OUTPUT = Path("sft_training_data_chatml.jsonl")
ALPACA_OUTPUT = Path("sft_training_data_alpaca.jsonl")

# Required CSV columns
REQUIRED_COLUMNS = {
    "ticket_id",
    "issue_category",
    "customer_query",
    "agent_context",
    "verified_resolution",
}

# System message used for the corporate-support assistant persona
SYSTEM_MESSAGE = (
    "You are a professional customer support specialist for TechCorp Inc. "
    "Provide accurate, empathetic, and concise resolutions. "
    "Use the provided context to inform your response. "
    "Never reveal internal tools, backend systems, or compensation policies."
)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MOCK DATASET                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

MOCK_CSV_DATA = """\
ticket_id,issue_category,customer_query,agent_context,verified_resolution
T-1001,Billing,"I was charged twice for my subscription this month. Can you help?","Customer has two charges: one on 2024-01-01 ($29.99) and one on 2024-01-03 ($29.99). The second charge is a system error. Refund approved by billing team.","I'm sorry for the inconvenience. I can confirm the duplicate charge on January 3rd was a billing system error. I have processed a full refund of $29.99, which will appear on your statement within 3–5 business days. Your subscription remains active."
T-1002,Account Access,"I forgot my password and the reset email isn't arriving.","Customer email: j.smith@example.com. Reset emails sent 3 times. Email not in spam. Possible issue: email domain has strict SPF/DKIM filtering.","Thank you for reaching out. Our records show three reset emails were sent to j.smith@example.com. Please check your spam and promotions folders. If still not received, I can send the reset link to an alternative email address, or verify your identity via phone to reset directly."
T-1003,Technical Issue,"The mobile app crashes every time I try to upload a photo.","Customer is on iOS 17.2, app version 4.1.0. Known bug: iOS 17.2 photo library permission regression. Fix deployed in app version 4.1.1 (released 2024-01-10).","I apologise for the trouble. This is a known issue on iOS 17.2 affecting photo uploads in app version 4.1.0. Our engineering team has released a fix in version 4.1.1. Please update your app via the App Store and the issue will be resolved. If you need further assistance after updating, don't hesitate to contact us."
T-1004,Shipping,"My order #ORD-88421 hasn't arrived and it's been 10 days.","Order placed 2024-01-02, estimated delivery 2024-01-08. Carrier: FedEx. Tracking shows 'In Transit – Delayed' since 2024-01-06. Weather delay in the Memphis hub. Expected new delivery date: 2024-01-14.","I understand how frustrating a delayed delivery can be. Your order #ORD-88421 is experiencing a weather-related delay at the Memphis distribution hub. The updated estimated delivery date is January 14th. I've applied a $10 store credit to your account for the inconvenience. You can track your package at fedex.com using tracking number in your order confirmation email."
T-1005,Refund Policy,"I want to return a product I bought 45 days ago. Is that possible?","Standard return window is 30 days. Customer is 15 days past the window. Product is a software licence key — non-refundable per ToS Section 4.2. However, customer is a Premium member (3-year tenure). Manager discretion allows one-time exception.","Thank you for being a valued Premium member for three years. While our standard return window is 30 days, I've reviewed your account and as a one-time courtesy, I can offer you a full store credit equivalent to the purchase price. Unfortunately, software licence keys are non-refundable per our terms, but the store credit can be used on any future purchase. Would that work for you?"
T-1006,Subscription,"How do I upgrade from the Basic plan to the Pro plan mid-cycle?","Pro plan costs $49.99/month vs. Basic at $19.99/month. Mid-cycle upgrades are prorated. Customer has 18 days remaining in current cycle. Upgrade is instant. Proration credit: (18/31) * $19.99 = $11.60 credit applied.","Upgrading from Basic to Pro is seamless and takes effect immediately. You'll be charged the prorated difference for the remaining 18 days of your current cycle — approximately $38.39 — and then $49.99 on your next renewal date. To upgrade, go to Account Settings → Subscription → Upgrade Plan. Let me know if you'd like me to initiate the upgrade on your behalf."
T-1007,Data Privacy,"I want to download all my personal data. How do I request that?","GDPR/CCPA data export request. Customer in the EU (GDPR applies). Data export must be provided within 30 days per GDPR Article 20. Export includes: account data, purchase history, support tickets, usage logs.","You have the right to download all your personal data under GDPR Article 20 (Right to Data Portability). To request your data export: log in → Account Settings → Privacy → Request Data Export. You'll receive a downloadable archive within 30 days. The archive includes your account details, purchase history, support ticket history, and usage logs. If you need assistance with the process, I'm happy to guide you."
T-1008,Product Feature,"Does the Pro plan include team collaboration features?","Pro plan includes: unlimited storage, priority support, API access, and up to 5 team seats. Team collaboration features (shared workspaces, role-based permissions, team analytics) are only in the Enterprise plan ($99/month, minimum 10 seats).","The Pro plan includes up to 5 team seats, API access, unlimited storage, and priority support — excellent for small teams. However, advanced team collaboration features such as shared workspaces, role-based permissions, and team analytics are available exclusively on our Enterprise plan. I'd be happy to arrange a free 14-day Enterprise trial or connect you with our sales team for a custom quote."
T-1009,Security,"I noticed a login from an unrecognised device in my account activity.","Login from IP 185.220.101.45 (Tor exit node, Germany) on 2024-01-11 at 02:14 UTC. Customer's usual location: New York, USA. Account not compromised yet but suspicious. Security team flagged for monitoring.","I take your security very seriously. I can see a login from an unrecognised device originating from Germany on January 11th, which is unusual for your account. As a precaution, I have immediately terminated all active sessions and locked that device out. Please reset your password now using a strong, unique password. I also strongly recommend enabling two-factor authentication under Account Settings → Security. If you did not initiate this login, your account is safe now."
T-1010,General Inquiry,"What are your customer support hours and available contact channels?","Support hours: Mon–Fri 09:00–20:00 EST, Sat 10:00–16:00 EST. Channels: Live chat (24/7 bot, human Mon–Fri), Email (response within 24h), Phone (Mon–Fri 09:00–18:00 EST), Community forum (24/7 self-serve).","Our customer support team is available Monday through Friday, 9:00 AM to 8:00 PM Eastern Time, and Saturday from 10:00 AM to 4:00 PM Eastern Time. You can reach us via: Live Chat (24/7 AI assistance, human agents available weekdays), Email support with a 24-hour response guarantee, and Phone support on weekdays from 9 AM to 6 PM EST. Our Community Forum is also available 24/7 for self-service answers."
"""

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATA CLASSES                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝


@dataclass
class ValidationReport:
    """Summary of data quality checks performed on the CSV."""

    total_rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    null_field_errors: int = 0
    sanitised_fields: int = 0


@dataclass
class DatasetStats:
    """Basic statistics for the converted dataset."""

    row_count: int = 0
    avg_prompt_chars: float = 0.0
    avg_completion_chars: float = 0.0
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SANITISATION UTILITIES                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _sanitise_field(value: str) -> str:
    """
    Clean a single text field:
      1. Strip leading/trailing whitespace.
      2. Remove ASCII control characters (except newline/tab).
      3. Normalise Unicode to NFC form.
      4. Collapse runs of 3+ blank lines to two.

    Args:
        value: Raw field string.

    Returns:
        Sanitised field string.
    """
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    # Remove control characters (0x00-0x1F except \t and \n)
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)
    # Normalise Unicode
    value = unicodedata.normalize("NFC", value)
    # Collapse excessive blank lines
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value


def _estimate_tokens(text: str) -> int:
    """
    Lightweight token estimator (~4 chars/token heuristic).
    Replace with tiktoken for production-grade counts.

    Args:
        text: Input string.

    Returns:
        Estimated token count.
    """
    return max(1, len(text) // 4)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CSV INGESTION & VALIDATION                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class CSVIngestionPipeline:
    """
    Loads, validates, and sanitises the customer-support CSV.

    Quality checks performed:
      • Presence of all required columns.
      • No null or empty required fields per row.
      • Field-level sanitisation (whitespace, control chars, Unicode).
    """

    def __init__(self, csv_path: Path = CSV_FILE) -> None:
        self._path = csv_path
        self.report = ValidationReport()

    def _ensure_csv_exists(self) -> None:
        """Create the mock CSV if the file does not exist on disk."""
        if not self._path.exists():
            logger.info("'%s' not found — creating mock dataset.", self._path)
            self._path.write_text(MOCK_CSV_DATA, encoding="utf-8")
            logger.info("Mock CSV written to '%s' (%d bytes).", self._path, self._path.stat().st_size)

    def load_and_validate(self) -> pd.DataFrame:
        """
        Load the CSV, validate schema and data quality, and return a
        clean DataFrame.

        Returns:
            Validated and sanitised DataFrame.

        Raises:
            ValueError: If required columns are missing.
        """
        self._ensure_csv_exists()
        df = pd.read_csv(self._path, dtype=str)

        # ── Schema validation ────────────────────────────────────────────────
        missing_cols = REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"CSV is missing required columns: {missing_cols}. "
                f"Found: {list(df.columns)}"
            )
        logger.info("Schema validation passed. Columns: %s", list(df.columns))

        self.report.total_rows = len(df)

        # ── Row-level validation & sanitisation ──────────────────────────────
        valid_mask = pd.Series([True] * len(df))

        for col in REQUIRED_COLUMNS:
            # Flag rows with null or blank fields
            null_mask = df[col].isna() | (df[col].str.strip() == "")
            invalid_count = null_mask.sum()
            if invalid_count:
                logger.warning(
                    "Column '%s': %d rows with null/empty values — rows dropped.",
                    col, invalid_count,
                )
                self.report.null_field_errors += int(invalid_count)
                valid_mask &= ~null_mask

            # Sanitise field values
            df[col] = df[col].fillna("").apply(_sanitise_field)
            self.report.sanitised_fields += len(df)

        df = df[valid_mask].reset_index(drop=True)
        self.report.valid_rows = len(df)
        self.report.invalid_rows = self.report.total_rows - self.report.valid_rows

        logger.info(
            "Validation complete: %d valid rows, %d invalid rows dropped.",
            self.report.valid_rows,
            self.report.invalid_rows,
        )
        return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FORMAT CONVERTERS                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class ChatMLConverter:
    """
    Converts a DataFrame row into ChatML / Llama-3 conversational format.

    Schema:
    {
        "messages": [
            {"role": "system",    "content": "<system_message>"},
            {"role": "user",      "content": "<customer_query> [Context: <agent_context>]"},
            {"role": "assistant", "content": "<verified_resolution>"}
        ]
    }
    """

    def convert_row(self, row: Dict[str, str]) -> Dict:
        """
        Convert a single CSV row to ChatML format.

        Args:
            row: Dict with keys from REQUIRED_COLUMNS.

        Returns:
            ChatML-formatted dict.
        """
        user_content = (
            f"{row['customer_query']}\n\n"
            f"[Support Context — Issue Category: {row['issue_category']} | "
            f"Ticket: {row['ticket_id']} | "
            f"Agent Notes: {row['agent_context']}]"
        )
        return {
            "messages": [
                {"role": "system",    "content": SYSTEM_MESSAGE},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": row["verified_resolution"]},
            ]
        }

    def convert_dataframe(self, df: pd.DataFrame) -> Iterator[Dict]:
        """Yield ChatML dicts for every row in *df*."""
        for _, row in df.iterrows():
            yield self.convert_row(row.to_dict())


class AlpacaConverter:
    """
    Converts a DataFrame row into Alpaca instruction-tuning format.

    Schema:
    {
        "instruction": "<task_instruction>",
        "input":       "<customer_query_and_context>",
        "output":      "<verified_resolution>"
    }
    """

    _INSTRUCTION = (
        "You are a customer support specialist. "
        "Read the customer's query and the provided agent context carefully, "
        "then provide a professional, empathetic, and accurate resolution."
    )

    def convert_row(self, row: Dict[str, str]) -> Dict:
        """
        Convert a single CSV row to Alpaca format.

        Args:
            row: Dict with keys from REQUIRED_COLUMNS.

        Returns:
            Alpaca-formatted dict.
        """
        input_text = (
            f"Issue Category: {row['issue_category']}\n"
            f"Ticket ID: {row['ticket_id']}\n"
            f"Customer Query: {row['customer_query']}\n"
            f"Agent Context: {row['agent_context']}"
        )
        return {
            "instruction": self._INSTRUCTION,
            "input":       input_text,
            "output":      row["verified_resolution"],
        }

    def convert_dataframe(self, df: pd.DataFrame) -> Iterator[Dict]:
        """Yield Alpaca dicts for every row in *df*."""
        for _, row in df.iterrows():
            yield self.convert_row(row.to_dict())


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  JSONL WRITER                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class JSONLWriter:
    """Writes an iterator of dicts to a newline-delimited JSON file."""

    @staticmethod
    def write(records: Iterator[Dict], output_path: Path) -> int:
        """
        Write *records* to *output_path* in JSONL format.

        Args:
            records    : Iterator of serialisable dicts.
            output_path: Destination file path.

        Returns:
            Number of records written.
        """
        count = 0
        with open(output_path, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        logger.info("Wrote %d records to '%s'.", count, output_path)
        return count


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STATISTICS ENGINE                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class DatasetStatisticsEngine:
    """
    Computes and displays basic dataset statistics from a JSONL file.

    Metrics:
      • Row count
      • Average prompt character length
      • Average completion character length
      • Estimated average prompt tokens
      • Estimated average completion tokens
    """

    @staticmethod
    def compute_chatml_stats(jsonl_path: Path) -> DatasetStats:
        """Compute statistics from a ChatML JSONL file."""
        prompt_chars: List[int] = []
        completion_chars: List[int] = []

        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                messages = record.get("messages", [])
                # Prompt = system + user messages concatenated
                prompt = " ".join(
                    m["content"] for m in messages if m["role"] != "assistant"
                )
                completion = next(
                    (m["content"] for m in messages if m["role"] == "assistant"), ""
                )
                prompt_chars.append(len(prompt))
                completion_chars.append(len(completion))

        count = len(prompt_chars)
        return DatasetStats(
            row_count=count,
            avg_prompt_chars=sum(prompt_chars) / count if count else 0.0,
            avg_completion_chars=sum(completion_chars) / count if count else 0.0,
            avg_prompt_tokens=sum(_estimate_tokens(str(c)) for c in prompt_chars) / count if count else 0.0,
            avg_completion_tokens=sum(_estimate_tokens(str(c)) for c in completion_chars) / count if count else 0.0,
        )

    @staticmethod
    def compute_alpaca_stats(jsonl_path: Path) -> DatasetStats:
        """Compute statistics from an Alpaca JSONL file."""
        prompt_chars: List[int] = []
        completion_chars: List[int] = []

        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                prompt = record.get("instruction", "") + " " + record.get("input", "")
                completion = record.get("output", "")
                prompt_chars.append(len(prompt))
                completion_chars.append(len(completion))

        count = len(prompt_chars)
        return DatasetStats(
            row_count=count,
            avg_prompt_chars=sum(prompt_chars) / count if count else 0.0,
            avg_completion_chars=sum(completion_chars) / count if count else 0.0,
            avg_prompt_tokens=sum(_estimate_tokens(str(c)) for c in prompt_chars) / count if count else 0.0,
            avg_completion_tokens=sum(_estimate_tokens(str(c)) for c in completion_chars) / count if count else 0.0,
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DISPLAY HELPERS                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _print_section(title: str) -> None:
    width = 72
    pad = (width - len(title) - 2) // 2
    print("\n" + "═" * pad + f" {title} " + "═" * (width - pad - len(title) - 2))


def _print_stats(label: str, stats: DatasetStats) -> None:
    _print_section(f"STATS: {label}")
    print(f"  Rows                     : {stats.row_count}")
    print(f"  Avg Prompt Chars         : {stats.avg_prompt_chars:,.1f}")
    print(f"  Avg Completion Chars     : {stats.avg_completion_chars:,.1f}")
    print(f"  Avg Prompt Tokens (est.) : {stats.avg_prompt_tokens:,.1f}")
    print(f"  Avg Completion Tokens    : {stats.avg_completion_tokens:,.1f}")


def _print_preview(label: str, jsonl_path: Path) -> None:
    _print_section(f"PREVIEW: {label}")
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        first_line = fh.readline()
    record = json.loads(first_line)
    print(json.dumps(record, indent=2, ensure_ascii=False)[:1200])
    if len(first_line) > 1200:
        print("  … [truncated for display]")


def _print_validation_report(report: ValidationReport) -> None:
    _print_section("VALIDATION REPORT")
    print(f"  Total rows read          : {report.total_rows}")
    print(f"  Valid rows               : {report.valid_rows}")
    print(f"  Invalid rows dropped     : {report.invalid_rows}")
    print(f"  Null/empty field errors  : {report.null_field_errors}")
    print(f"  Fields sanitised         : {report.sanitised_fields}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE ORCHESTRATOR                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class FineTunePrepPipeline:
    """
    Top-level orchestrator for the fine-tuning data preparation pipeline.

    Steps:
      1. Load & validate CSV.
      2. Convert to ChatML JSONL.
      3. Convert to Alpaca JSONL.
      4. Compute and display statistics.
      5. Print schema-validity previews.
    """

    def __init__(self) -> None:
        self._ingestion = CSVIngestionPipeline()
        self._chatml_converter = ChatMLConverter()
        self._alpaca_converter = AlpacaConverter()
        self._writer = JSONLWriter()
        self._stats_engine = DatasetStatisticsEngine()

    def run(self) -> None:
        """Execute the full pipeline."""
        print("\n" + "█" * 72)
        print("█" + " " * 16 + "FINE-TUNE DATA PREP PIPELINE — DAY 10" + " " * 17 + "█")
        print("█" + " " * 14 + "SFTTrainer-Compatible Dataset Generation" + " " * 16 + "█")
        print("█" * 72)

        # ── Step 1: Ingest & Validate ────────────────────────────────────────
        _print_section("STEP 1: CSV INGESTION & VALIDATION")
        df = self._ingestion.load_and_validate()
        _print_validation_report(self._ingestion.report)

        # ── Step 2: ChatML Conversion ────────────────────────────────────────
        _print_section("STEP 2: CHATML CONVERSION")
        chatml_count = self._writer.write(
            self._chatml_converter.convert_dataframe(df),
            CHATML_OUTPUT,
        )
        print(f"  ✅ {chatml_count} records → '{CHATML_OUTPUT}'")

        # ── Step 3: Alpaca Conversion ────────────────────────────────────────
        _print_section("STEP 3: ALPACA CONVERSION")
        alpaca_count = self._writer.write(
            self._alpaca_converter.convert_dataframe(df),
            ALPACA_OUTPUT,
        )
        print(f"  ✅ {alpaca_count} records → '{ALPACA_OUTPUT}'")

        # ── Step 4: Statistics ───────────────────────────────────────────────
        chatml_stats = self._stats_engine.compute_chatml_stats(CHATML_OUTPUT)
        alpaca_stats = self._stats_engine.compute_alpaca_stats(ALPACA_OUTPUT)
        _print_stats("ChatML JSONL", chatml_stats)
        _print_stats("Alpaca JSONL", alpaca_stats)

        # ── Step 5: Schema Previews ──────────────────────────────────────────
        _print_preview("ChatML JSONL (first record)", CHATML_OUTPUT)
        _print_preview("Alpaca JSONL (first record)", ALPACA_OUTPUT)

        _print_section("PIPELINE COMPLETE")
        print(f"  📁 {CHATML_OUTPUT}  ({CHATML_OUTPUT.stat().st_size:,} bytes)")
        print(f"  📁 {ALPACA_OUTPUT}  ({ALPACA_OUTPUT.stat().st_size:,} bytes)")
        print("\n  Ready for Hugging Face SFTTrainer 🚀\n")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝


if __name__ == "__main__":
    pipeline = FineTunePrepPipeline()
    pipeline.run()
