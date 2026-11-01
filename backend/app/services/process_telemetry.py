"""Process Telemetry Analysis Service for student coding behavioral evidence."""
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from app.models.process_event import ProcessEvent
from app.schemas.telemetry import ProcessFeaturesResponse, TimelineEventSummary


class ProcessTelemetryService:
    """Analyzes fine-grained editor telemetry to derive problem-solving dynamics."""

    BURST_THRESHOLD_SECONDS = 0.5

    @classmethod
    def compute_features(
        cls,
        session_id: str,
        events: List[ProcessEvent],
    ) -> ProcessFeaturesResponse:
        """Derive comprehensive behavioral features and ethical timeline from process events."""
        if not events:
            return ProcessFeaturesResponse(
                session_id=session_id,
                total_events=0,
                edit_count=0,
                paste_count=0,
                run_count=0,
                error_count=0,
                total_chars_pasted=0,
                total_chars_edited=0,
                paste_ratio=0.0,
                coding_duration_seconds=0.0,
                average_edit_interval_seconds=0.0,
                burst_count=0,
                burstiness_ratio=0.0,
                time_to_first_run_seconds=None,
                error_recovery_rate=1.0,
                typing_speed_cpm=0.0,
                process_profile="Uninitialized Session",
                timeline_events=[],
            )

        # Ensure events are ordered chronologically
        sorted_events = sorted(events, key=lambda e: e.timestamp or datetime.min)
        t0 = sorted_events[0].timestamp
        tn = sorted_events[-1].timestamp
        total_duration = max(0.0, (tn - t0).total_seconds())

        total_events = len(sorted_events)
        edit_count = sum(1 for e in sorted_events if e.event_type == "edit")
        paste_count = sum(1 for e in sorted_events if e.event_type == "paste")
        run_count = sum(1 for e in sorted_events if e.event_type == "run")
        error_count = sum(
            1 for e in sorted_events 
            if e.event_type == "error" or (e.event_metadata and e.event_metadata.get("is_error"))
        )

        total_chars_pasted = sum(e.paste_length or 0 for e in sorted_events if e.event_type == "paste")
        total_chars_edited = sum(abs(e.delta_chars or 0) for e in sorted_events if e.event_type == "edit")

        total_content = total_chars_pasted + total_chars_edited
        paste_ratio = (total_chars_pasted / total_content) if total_content > 0 else 0.0

        # Intervals and burst analysis
        intervals: List[float] = []
        burst_count = 0
        last_edit_ts: Optional[datetime] = None

        for e in sorted_events:
            if e.event_type in ("edit", "paste"):
                if last_edit_ts is not None:
                    delta = max(0.0, (e.timestamp - last_edit_ts).total_seconds())
                    intervals.append(delta)
                    if delta < cls.BURST_THRESHOLD_SECONDS and e.event_type == "edit":
                        burst_count += 1
                last_edit_ts = e.timestamp

        avg_interval = (sum(intervals) / len(intervals)) if intervals else 0.0
        burstiness_ratio = (burst_count / edit_count) if edit_count > 0 else 0.0

        # Time to first run
        time_to_first_run: Optional[float] = None
        for e in sorted_events:
            if e.event_type in ("run", "error"):
                time_to_first_run = round(max(0.0, (e.timestamp - t0).total_seconds()), 1)
                break

        # Error recovery rate
        error_indices = [
            i for i, e in enumerate(sorted_events)
            if e.event_type == "error" or (e.event_metadata and e.event_metadata.get("is_error"))
        ]
        if not error_indices:
            error_recovery_rate = 1.0
        else:
            recovered = 0
            for idx in error_indices:
                # Recovered if followed by subsequent edits or a non-error run
                subsequent = sorted_events[idx + 1:]
                if any(s.event_type in ("edit", "run") for s in subsequent):
                    recovered += 1
            error_recovery_rate = round(recovered / len(error_indices), 2)

        # Typing speed (chars per minute)
        duration_minutes = total_duration / 60.0
        typing_speed_cpm = round((total_chars_edited / duration_minutes), 1) if duration_minutes > 0.1 else 0.0

        # Pedagogical Profile classification
        if paste_ratio > 0.65 and edit_count < 8:
            profile = "Rapid Block Importer"
        elif run_count >= 5 and error_recovery_rate >= 0.5:
            profile = "Iterative Trial-and-Error Debugger"
        elif run_count == 0 and total_duration > 45:
            profile = "Cautious Non-Testing Author"
        elif burstiness_ratio > 0.35 and paste_ratio < 0.2:
            profile = "Fluent Continuous Typist"
        else:
            profile = "Balanced Incremental Developer"

        # Generate timeline milestone summaries
        timeline_events: List[TimelineEventSummary] = []
        for e in sorted_events:
            offset = round(max(0.0, (e.timestamp - t0).total_seconds()), 1)
            if e.event_type == "paste":
                timeline_events.append(
                    TimelineEventSummary(
                        offset_seconds=offset,
                        event_type="paste",
                        label="Pasted Code Block",
                        detail=f"+{e.paste_length} characters pasted at L{e.cursor_line or 1}",
                    )
                )
            elif e.event_type == "run":
                is_err = bool(e.event_metadata and e.event_metadata.get("is_error"))
                timeline_events.append(
                    TimelineEventSummary(
                        offset_seconds=offset,
                        event_type="error" if is_err else "run",
                        label="Sandbox Run Error" if is_err else "Sandbox Execution Attempt",
                        detail=e.event_metadata.get("message") if e.event_metadata else "Executed code in sandbox",
                    )
                )
            elif e.event_type == "submit":
                timeline_events.append(
                    TimelineEventSummary(
                        offset_seconds=offset,
                        event_type="submit",
                        label="Submitted for AI Viva Assessment",
                        detail="Locked snapshot sent to question generator",
                    )
                )

        return ProcessFeaturesResponse(
            session_id=session_id,
            total_events=total_events,
            edit_count=edit_count,
            paste_count=paste_count,
            run_count=run_count,
            error_count=error_count,
            total_chars_pasted=total_chars_pasted,
            total_chars_edited=total_chars_edited,
            paste_ratio=round(paste_ratio, 3),
            coding_duration_seconds=round(total_duration, 1),
            average_edit_interval_seconds=round(avg_interval, 2),
            burst_count=burst_count,
            burstiness_ratio=round(burstiness_ratio, 3),
            time_to_first_run_seconds=time_to_first_run,
            error_recovery_rate=error_recovery_rate,
            typing_speed_cpm=typing_speed_cpm,
            process_profile=profile,
            timeline_events=timeline_events[:50],  # Keep top milestones
        )
