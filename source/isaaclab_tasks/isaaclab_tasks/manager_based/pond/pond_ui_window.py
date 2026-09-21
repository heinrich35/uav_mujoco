# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom UI window for the Pond environment — gosling camera preview viewport."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import omni
from isaaclab.envs.ui.base_env_window import BaseEnvWindow

if TYPE_CHECKING:
    from ..pond_rl_env import PondRLEnv

CAM_PRIM_PATH = "/World/envs/env_0/Gosling/gosling_cam"


class PondEnvWindow(BaseEnvWindow):
    """Window manager for the Pond environment.

    Creates a second viewport window showing the gosling's forward camera,
    leaving the main scene viewport unchanged.
    """

    def __init__(self, env: PondRLEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        asyncio.ensure_future(self._setup_camera_view())

    async def _setup_camera_view(self):
        """Create a dedicated gosling camera viewport."""
        # Wait for the rendering subsystem to fully initialise
        for _ in range(30):
            await omni.kit.app.get_app().next_update_async()

        try:
            import omni.kit.viewport.utility as vp_util

            # create_viewport_window creates a NEW viewport window
            vp_window = vp_util.create_viewport_window(
                "Gosling Camera",
                width=640,
                height=480,
            )

            if vp_window is None:
                print("[PondUI] create_viewport_window returned None")
                return

            # vp_window has a .viewport_api attribute that we can use
            api = getattr(vp_window, "viewport_api", None)
            if api is not None and hasattr(api, "set_active_camera"):
                api.set_active_camera(CAM_PRIM_PATH)
                print("[PondUI] Gosling camera viewport created")
            else:
                print("[PondUI] Viewport created but no camera API — set camera manually")

        except Exception as e:
            print(f"[PondUI] Camera viewport error: {e}")

    def __del__(self):
        try:
            super().__del__()
        except Exception:
            pass
