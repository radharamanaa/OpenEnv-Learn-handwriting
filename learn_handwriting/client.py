# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Learn Handwriting Environment Client."""

import json
import logging
import os
from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import LearnHandwritingAction, LearnHandwritingObservation, LearnHandwritingState

logger = logging.getLogger(__name__)

# Set LEARN_HANDWRITING_ENV_LOG=1 (or true/yes/debug) for WebSocket request/response logs.
_WS_LOG_ENV = "LEARN_HANDWRITING_ENV_LOG"


def _ws_verbose_logging_enabled() -> bool:
    return os.environ.get(_WS_LOG_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "debug",
        "all",
    )


def _json_for_log(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, default=str, ensure_ascii=False)
    except Exception:
        return repr(obj)


def configure_openenv_ws_logging() -> None:
    """Route ``learn_handwriting.client`` logs to stdout when ``LEARN_HANDWRITING_ENV_LOG`` is set.

    Call once at the start of training or smoke scripts so HF Jobs / notebooks show
    WebSocket traffic (INFO) and full success payloads (DEBUG on this module).
    No-op when the env var is unset or falsey.
    """
    import sys

    if not _ws_verbose_logging_enabled():
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    print(
        f"[learn_handwriting] LEARN_HANDWRITING_ENV_LOG is set; OpenEnv WebSocket logging to stdout.",
        flush=True,
    )


class LearnHandwritingEnv(
    EnvClient[LearnHandwritingAction, LearnHandwritingObservation, LearnHandwritingState]
):
    """
    Client for the Learn Handwriting Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Debugging: set environment variable ``LEARN_HANDWRITING_ENV_LOG=1`` (or ``true``)
    to log every WebSocket request and a short summary of each successful response.
    Full successful responses are logged at DEBUG on this logger. On server
    ``VALIDATION_ERROR``, the full error frame is logged at ERROR and Pydantic
    ``errors`` are appended to the raised ``RuntimeError``.

    Example:
        >>> # Connect to a running server
        >>> with LearnHandwritingEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(LearnHandwritingAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = LearnHandwritingEnv.from_docker_image("learn_handwriting-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(LearnHandwritingAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: LearnHandwritingAction) -> Dict:
        """Convert LearnHandwritingAction to JSON payload for the step message."""
        return {
            "action_type": action.action_type,
            "x1": action.x1,
            "y1": action.y1,
            "x2": action.x2,
            "y2": action.y2,
            "x3": action.x3,
            "y3": action.y3,
            "radius": action.radius,
            "rx": action.rx,
            "ry": action.ry,
        }

    async def _send_and_receive(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send one WebSocket frame, receive reply; log details when enabled."""
        verbose = _ws_verbose_logging_enabled()
        if verbose:
            logger.info(
                "[learn_handwriting OpenEnv] → send type=%r\n%s",
                message.get("type"),
                _json_for_log(message),
            )

        await self._send(message)
        response = await self._receive()

        if response.get("type") == "error":
            err = response.get("data") or {}
            logger.error(
                "[learn_handwriting OpenEnv] ← error type=%r code=%r message=%r full_response=\n%s",
                response.get("type"),
                err.get("code"),
                err.get("message"),
                _json_for_log(response),
            )
            detail = ""
            raw_errors = err.get("errors")
            if raw_errors is not None:
                ej = _json_for_log(raw_errors)
                if len(ej) > 8000:
                    ej = ej[:8000] + "\n…(truncated for RuntimeError)"
                detail = f" validation_errors={ej}"
            raise RuntimeError(
                f"Server error: {err.get('message', 'Unknown error')} "
                f"(code: {err.get('code', 'UNKNOWN')}){detail}"
            )

        if verbose:
            data = response.get("data") or {}
            obs = data.get("observation") or {}
            logger.info(
                "[learn_handwriting OpenEnv] ← ok type=%r done=%r reward=%r "
                "match_percentage=%r target=%r strokes_used=%r",
                response.get("type"),
                data.get("done"),
                data.get("reward"),
                obs.get("match_percentage"),
                obs.get("target_character"),
                obs.get("strokes_used"),
            )
            logger.debug(
                "[learn_handwriting OpenEnv] ← full response:\n%s",
                _json_for_log(response),
            )

        return response

    def _parse_result(self, payload: Dict) -> StepResult[LearnHandwritingObservation]:
        """Parse server response into StepResult[LearnHandwritingObservation]."""
        obs_data = payload.get("observation", {})
        observation = LearnHandwritingObservation(
            target_character=obs_data.get("target_character", ""),
            strokes_used=obs_data.get("strokes_used", 0),
            pixels_matched_this_stroke=obs_data.get("pixels_matched_this_stroke", 0),
            total_matched_pixels=obs_data.get("total_matched_pixels", 0),
            match_percentage=obs_data.get("match_percentage", 0.0),
            pixels_wasted_this_stroke=obs_data.get("pixels_wasted_this_stroke", 0),
            total_drawn_pixels=obs_data.get("total_drawn_pixels", 0),
            max_allowed_pixels=obs_data.get("max_allowed_pixels", 0),
            ink_remaining=obs_data.get("ink_remaining", 0),
            integrity_violated=obs_data.get("integrity_violated", False),
            char_bbox_x1=obs_data.get("char_bbox_x1", 0),
            char_bbox_y1=obs_data.get("char_bbox_y1", 0),
            char_bbox_x2=obs_data.get("char_bbox_x2", 99),
            char_bbox_y2=obs_data.get("char_bbox_y2", 99),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> LearnHandwritingState:
        """Parse server response into LearnHandwritingState."""
        return LearnHandwritingState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            target_character=payload.get("target_character", ""),
            strokes_used=payload.get("strokes_used", 0),
            canvas=payload.get("canvas", [[0] * 100 for _ in range(100)]),
        )
