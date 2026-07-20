# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum


class EmbodimentTag(Enum):
    GR1 = "gr1"
    """
    The GR1 dataset.
    """
    AGILEX = "agilex"
    """
    The Agilex dataset.
    """
    OXE_DROID = "oxe_droid"
    """
    The OxE Droid dataset.
    """

    OXE_BRIDGE = "oxe_bridge"
    """
    The OxE Bridge dataset.
    """

    OXE_RT1 = "oxe_rt1"
    """
    The OxE RT-1 dataset.
    """

    AGIBOT_GENIE1 = "agibot_genie1"
    """
    The AgiBot Genie-1 with gripper dataset.
    """

    NEW_EMBODIMENT = "new_embodiment"
    """
    Any new embodiment for finetuning.
    """

    FRANKA = 'franka'
    """
    The Franka Emika Panda robot.
    """
    FRANKA_FR3_DUAL = 'franka_fr3_dual'
    """
    The Dual Franka Emika Panda robot.
    """
    UR = 'ur'
    """
    The Universal Robots UR1 robot.
    """
    HUMAN = "human"
    """
    Human first-person egocentric embodiment.
    """
    EGODEX = 'egodex'
    """
    The EgoDex dataset.
    """
    PandaOmron = 'panda_omron'
    """
    The Panda Omron robot for robocasa.
    """
# Embodiment tag string: to projector index in the Action Expert Module
EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.HUMAN.value: 0,
    EmbodimentTag.NEW_EMBODIMENT.value: 31,
    EmbodimentTag.OXE_DROID.value: 17,
    EmbodimentTag.OXE_BRIDGE.value: 18,
    EmbodimentTag.OXE_RT1.value: 19,
    EmbodimentTag.AGIBOT_GENIE1.value: 26,
    EmbodimentTag.GR1.value: 24,
    EmbodimentTag.FRANKA.value: 25,
    EmbodimentTag.AGILEX.value: 1,
    EmbodimentTag.FRANKA_FR3_DUAL.value: 2,
    EmbodimentTag.UR.value: 3,
    EmbodimentTag.PandaOmron.value: 4,
}

# Robot type to embodiment tag mapping
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "libero_franka": EmbodimentTag.FRANKA,
    "droid": EmbodimentTag.FRANKA,
    "bridge": EmbodimentTag.OXE_BRIDGE,
    "oxe_rt1": EmbodimentTag.OXE_RT1,
    "fractal": EmbodimentTag.OXE_RT1,
    "demo_sim_franka_delta_joints": EmbodimentTag.FRANKA,
    "custom_robot_config": EmbodimentTag.NEW_EMBODIMENT,
    "SO101": EmbodimentTag.NEW_EMBODIMENT,
    "fold_towel": EmbodimentTag.NEW_EMBODIMENT,
    "gr1": EmbodimentTag.GR1,
    "gr1_joint_eef": EmbodimentTag.GR1,
    "agibot_genie": EmbodimentTag.AGIBOT_GENIE1,
    "robomind_franka_1rgb": EmbodimentTag.FRANKA,
    "robomind_franka_3rgb": EmbodimentTag.FRANKA,
    "robomind_franka_fr3_dual": EmbodimentTag.FRANKA_FR3_DUAL,
    "robomind_ur_1rgb": EmbodimentTag.UR,
    "agilex": EmbodimentTag.AGILEX,
    "robotwin_joint": EmbodimentTag.AGILEX,
    "robotwin_eef": EmbodimentTag.AGILEX,
    "human": EmbodimentTag.HUMAN,
    "PandaOmron": EmbodimentTag.PandaOmron,
}
