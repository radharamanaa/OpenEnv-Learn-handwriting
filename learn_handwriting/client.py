# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Learn Handwriting Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import LearnHandwritingAction, LearnHandwritingObservation, LearnHandwritingState


class LearnHandwritingEnv(
    EnvClient[LearnHandwritingAction, LearnHandwritingObservation, LearnHandwritingState]
):
    """
    Client for the Learn Handwriting Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

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
            "x1": action.x1,
            "y1": action.y1,
            "x2": action.x2,
            "y2": action.y2,
            "width": action.width,
        }

    def _parse_result(self, payload: Dict) -> StepResult[LearnHandwritingObservation]:
        """Parse server response into StepResult[LearnHandwritingObservation]."""
        obs_data = payload.get("observation", {})
        observation = LearnHandwritingObservation(
            target_character=obs_data.get("target_character", ""),
            strokes_used=obs_data.get("strokes_used", 0),
            pixels_matched_this_stroke=obs_data.get("pixels_matched_this_stroke", 0),
            total_matched_pixels=obs_data.get("total_matched_pixels", 0),
            match_percentage=obs_data.get("match_percentage", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
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
