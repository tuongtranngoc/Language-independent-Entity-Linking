"""
Colorful console output for CQ Evaluation using rich.

Provides structured, colorful terminal output for the evaluation process.
Uses a "Tokyo Night" inspired color theme.
"""

import os
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

# ============================================================================
# Tokyo Night Color Theme
# ============================================================================
COLORS = {
    "primary": "#7AA2F7",  # Soft blue - headers, titles
    "secondary": "#BB9AF7",  # Soft purple - emphasis
    "success": "#9ECE6A",  # Soft green - success
    "warning": "#E0AF68",  # Soft amber - warnings
    "error": "#F7768E",  # Soft red/pink - errors
    "text": "#A9B1D6",  # Soft gray-blue - regular text
    "muted": "#565F89",  # Muted gray - less important
    "accent": "#7DCFFF",  # Bright cyan - accents
    "border": "#3B4261",  # Border color
}

# Rich styles
STYLE_PRIMARY = Style(color=COLORS["primary"], bold=True)
STYLE_SECONDARY = Style(color=COLORS["secondary"])
STYLE_SUCCESS = Style(color=COLORS["success"])
STYLE_WARNING = Style(color=COLORS["warning"])
STYLE_ERROR = Style(color=COLORS["error"])
STYLE_TEXT = Style(color=COLORS["text"])
STYLE_MUTED = Style(color=COLORS["muted"])
STYLE_ACCENT = Style(color=COLORS["accent"], bold=True)


class EvalConsolePrinter:
    """
    Rich console printer for CQ Evaluation verbose output.

    Displays colorful, structured output showing:
    - Configuration summary
    - CQ processing status
    - Experiment execution results
    - Scoring progress
    - Final summary
    """

    def __init__(self, enabled: bool = True, log_file: str | None = None):
        """
        Initialize the console printer.

        Args:
            enabled: Whether colorful printing is enabled.
            log_file: Optional path to write plain-text logs (no color).
        """
        self.enabled = enabled
        self.console = Console() if enabled else None
        self._log_fh = None
        self._file_console = None
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            self._log_fh = open(log_file, 'a', encoding='utf-8')
            self._file_console = Console(file=self._log_fh, no_color=True, highlight=False)

    def __del__(self):
        if self._log_fh:
            self._log_fh.close()

    def _emit(self, text: Text) -> None:
        """Write to terminal and, if configured, to the log file."""
        if self.enabled and self.console:
            self.console.print(text)
        if self._file_console:
            self._file_console.print(text)

    def print_header(
        self,
        num_cqs: int,
        num_experiments: int,
        scoring_enabled: bool,
        scoring_models: list[str] | None = None,
        num_criteria: int = 0,
        debug_enabled: bool = False,
    ) -> None:
        """Print the evaluation configuration header."""
        if not self.enabled:
            return

        # Main title
        title = Text()
        title.append("◆ ", style=STYLE_ACCENT)
        title.append("CQ Evaluation", style=Style(color=COLORS["primary"], bold=True))
        title.append(" ━ Competency Question Evaluation", style=STYLE_MUTED)

        # Configuration table
        config_table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 2),
            expand=True,
        )
        config_table.add_column("key", style=STYLE_MUTED, width=18)
        config_table.add_column("value", style=STYLE_TEXT)
        config_table.add_column("key2", style=STYLE_MUTED, width=18)
        config_table.add_column("value2", style=STYLE_TEXT)

        config_table.add_row(
            "CQs Found",
            Text(str(num_cqs), style=STYLE_ACCENT),
            "Experiments",
            Text(str(num_experiments), style=STYLE_ACCENT),
        )

        scoring_text = Text("Enabled", style=STYLE_SUCCESS) if scoring_enabled else Text("Disabled", style=STYLE_MUTED)
        config_table.add_row(
            "Scoring",
            scoring_text,
            "Criteria",
            Text(str(num_criteria), style=STYLE_ACCENT) if scoring_enabled else Text("N/A", style=STYLE_MUTED),
        )

        if scoring_models:
            models_text = ", ".join(scoring_models)
            config_table.add_row(
                "Scoring Models",
                Text(models_text, style=STYLE_SECONDARY),
                "Debug Mode",
                Text("On", style=STYLE_WARNING) if debug_enabled else Text("Off", style=STYLE_MUTED),
            )

        # Wrap in panel
        panel = Panel(
            config_table,
            title=title,
            title_align="left",
            border_style=COLORS["border"],
            padding=(1, 2),
        )

        self.console.print()
        self.console.print(panel)
        self.console.print()

    def print_cq_start(self, cq_name: str, cq_index: int, total_cqs: int) -> None:
        """Print the start of processing a CQ."""
        if not self.enabled:
            return

        header = Text()
        header.append(f" CQ {cq_index + 1}/{total_cqs}: ", style=STYLE_PRIMARY)
        header.append(cq_name, style=STYLE_ACCENT)

        rule = Rule(
            header,
            style=COLORS["border"],
            characters="═",
        )
        self.console.print()
        self.console.print(rule)

    def print_question(self, question: str, max_length: int = 100) -> None:
        """Print the question being processed."""
        if not self.enabled:
            return

        text = Text()
        text.append("  Question: ", style=STYLE_MUTED)
        display_question = question[:max_length] + "..." if len(question) > max_length else question
        text.append(display_question, style=STYLE_TEXT)
        self._emit(text)

    def print_experiment_start(self, experiment_name: str) -> None:
        """Print the start of an experiment run."""
        if not self.enabled:
            return

        text = Text()
        text.append("  ▸ ", style=STYLE_SECONDARY)
        text.append("Running: ", style=STYLE_MUTED)
        text.append(experiment_name, style=STYLE_SECONDARY)
        self._emit(text)

    def print_experiment_success(self, experiment_name: str, answer_preview: str = "") -> None:
        """Print successful experiment completion."""
        if not self.enabled:
            return

        text = Text()
        text.append("    ✓ ", style=STYLE_SUCCESS)
        text.append(experiment_name, style=STYLE_SUCCESS)
        text.append(" - SUCCESS", style=STYLE_SUCCESS)
        if answer_preview:
            preview = answer_preview[:50] + "..." if len(answer_preview) > 50 else answer_preview
            text.append(f" ({len(answer_preview)} chars)", style=STYLE_MUTED)
        self._emit(text)

    def print_experiment_error(self, experiment_name: str, error: str) -> None:
        """Print experiment error."""
        if not self.enabled:
            return

        text = Text()
        text.append("    ✗ ", style=STYLE_ERROR)
        text.append(experiment_name, style=STYLE_ERROR)
        text.append(" - FAILED - ", style=STYLE_ERROR)
        text.append(str(error)[:100], style=STYLE_ERROR)
        self._emit(text)

    def print_translation_status(self, status: str = "done") -> None:
        """Print translation status."""
        if not self.enabled:
            return

        text = Text()
        text.append("      ↳ ", style=STYLE_MUTED)
        text.append("Translation: ", style=STYLE_MUTED)
        text.append(status, style=STYLE_SUCCESS if status == "done" else STYLE_WARNING)
        self._emit(text)

    def print_scoring_start(self, criteria_name: str) -> None:
        """Print start of scoring for a criteria."""
        if not self.enabled:
            return

        text = Text()
        text.append("      ↳ ", style=STYLE_MUTED)
        text.append("Scoring: ", style=STYLE_MUTED)
        text.append(criteria_name, style=STYLE_SECONDARY)
        self._emit(text)

    def print_score_result(self, model_name: str, score: Any) -> None:
        """Print a score result."""
        if not self.enabled:
            return

        text = Text()
        text.append("        • ", style=STYLE_MUTED)
        text.append(model_name, style=STYLE_ACCENT)
        text.append(": ", style=STYLE_MUTED)

        if score is None:
            text.append("N/A", style=STYLE_MUTED)
        elif isinstance(score, str) and "ERROR" in score:
            text.append(str(score), style=STYLE_ERROR)
        else:
            text.append(str(score), style=STYLE_SUCCESS)
        self._emit(text)

    def print_scoring_error(self, error: str) -> None:
        """Print scoring error."""
        if not self.enabled:
            return

        text = Text()
        text.append("        ✗ ", style=STYLE_ERROR)
        text.append("Scoring error: ", style=STYLE_ERROR)
        text.append(str(error)[:80], style=STYLE_ERROR)
        self._emit(text)

    def print_cleanup_done(self, path: str) -> None:
        """Print cleanup completion message."""
        if not self.enabled:
            return

        text = Text()
        text.append("◇ ", style=STYLE_MUTED)
        text.append("Cleaned up: ", style=STYLE_MUTED)
        text.append(path, style=STYLE_TEXT)
        self._emit(text)

    def print_debug_enabled(self, debug_path: str) -> None:
        """Print debug output path."""
        if not self.enabled:
            return

        text = Text()
        text.append("◇ ", style=STYLE_ACCENT)
        text.append("Debug output: ", style=STYLE_MUTED)
        text.append(debug_path, style=STYLE_ACCENT)
        self._emit(text)

    def print_info(self, message: str) -> None:
        """Print an informational message."""
        if not self.enabled:
            return

        text = Text()
        text.append("◇ ", style=STYLE_MUTED)
        text.append(message, style=STYLE_TEXT)
        self._emit(text)

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        if not self.enabled:
            return

        text = Text()
        text.append("⚠ ", style=STYLE_WARNING)
        text.append(message, style=STYLE_WARNING)
        self._emit(text)

    def print_error(self, message: str) -> None:
        """Print an error message."""
        if not self.enabled:
            return

        text = Text()
        text.append("✗ ", style=STYLE_ERROR)
        text.append(message, style=STYLE_ERROR)
        self._emit(text)

    def print_summary(
        self,
        output_path: str,
        num_cqs: int,
        num_experiments: int,
        debug_path: str | None = None,
    ) -> None:
        """Print final summary."""
        if not self.enabled:
            return

        self.console.print()
        self.console.print(Rule(style=COLORS["border"], characters="═"))

        # Summary table
        summary_table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 2),
        )
        summary_table.add_column("metric", style=STYLE_MUTED)
        summary_table.add_column("value", style=STYLE_ACCENT)

        summary_table.add_row("CQs Processed", str(num_cqs))
        summary_table.add_row("Experiments Run", str(num_experiments))
        summary_table.add_row("Results Saved", output_path)
        if debug_path:
            summary_table.add_row("Debug Output", debug_path)

        self.console.print(summary_table, justify="center")
        self.console.print(Rule(style=COLORS["border"], characters="═"))
        self.console.print()

    def print_batch_result(self, batch_index: int, success: bool, message: str = "") -> None:
        """Print batch processing result (similar to the image example)."""
        if not self.enabled:
            return

        text = Text()
        text.append(f"Batch {batch_index}: ", style=STYLE_WARNING)

        if success:
            text.append("SUCCESS", style=STYLE_SUCCESS)
            if message:
                text.append(f" - {message}", style=STYLE_SUCCESS)
        else:
            text.append("FAILED", style=STYLE_ERROR)
            if message:
                text.append(f" - {message}", style=STYLE_ERROR)

        self._emit(text)


# Global printer instance
_printer: EvalConsolePrinter | None = None


def get_printer(enabled: bool = True, log_file: str | None = None) -> EvalConsolePrinter:
    """Get or create the global printer instance.

    Passing ``log_file`` on the first call (or after ``reset_printer()``)
    configures plain-text file logging alongside the terminal output.
    """
    global _printer
    if _printer is None:
        _printer = EvalConsolePrinter(enabled=enabled, log_file=log_file)
    return _printer


def reset_printer() -> None:
    """Reset the global printer instance."""
    global _printer
    _printer = None