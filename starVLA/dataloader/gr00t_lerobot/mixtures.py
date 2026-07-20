"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""



# Dataset mixture name mapped to a list of tuples containing:
## {nakename: [(data_name, sampling_weight, robot_type)] }
DATASET_NAMED_MIXTURES = {

    "custom_dataset": [
        ("custom_dataset_name", 1.0, "custom_robot_config"),
    ],
    "custom_dataset_2": [
        ("custom_dataset_name_1", 1.0, "custom_robot_config"),
        ("custom_dataset_name_2", 1.0, "custom_robot_config"),
    ],
    "libero_all": [
        ("libero_plus_lerobot", 1.0, "libero_franka"),
        ("libero_merged_no_noops_20hz", 1.0, "libero_franka"),
    ],
    "libero_mix_hz":[
        ("libero_merged_no_noops_5hz", 1.0, "libero_franka"),
        ("libero_merged_no_noops_10hz", 1.0, "libero_franka"),
        ("libero_merged_no_noops_20hz", 1.0, "libero_franka"),
    ],
    "libero": [
        # ("libero_object_image", 1.0, "libero_franka"),
        # ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        # ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        # ("libero_10_image", 1.0, "libero_franka"),
        # ("libero", 1.0, "libero_franka"),
        ("libero_merged_no_noops_20hz", 1.0, "libero_franka"),
        # ("libero_10", 1.0, "libero_franka"),
                # ("libero_90_no_noops_lerobot", 1.0, "libero_franka"),
    ],
    "real_pick_place": [
        ("pp_filtered_real_merged_v3.0", 1.0, "libero_franka"),
    ],
    "real_open_drawer":[
        ("open_drawer_180_lerobot_v30", 1.0, "libero_franka"),
    ],
    "real_fold_towel":[
        ("fold_towel_gop10", 1.0, "fold_towel"),
    ],
    


    "fourier_gr1_unified_1000": [
        ("gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
    ],

    "BEHAVIOR_challenge": [
        ("BEHAVIOR_challenge", 1.0, "R1Pro"),
    ],


    "SO101_pick": [
        ("pick_dataset_name", 1.0, "SO101"),
    ],
    "community_so101_pick_pen": [
        ("EverNorif/so101-pick-pen", 1.0, "SO101"),
    ],

    "arx_x5": [
        ("arx_x5", 1.0, "arx_x5"),
    ],

    "robotwin_joint": [
        ("RoboTwin_merged", 1.0, "robotwin_joint"),
    ],
    "agilex_joint": [
        ("robomind_agilex_3rgb", 1.0, "robotwin_joint"),
    ],
    "robotwin_eef": [
        ("robotwin_eef_all_v30/robotwin_eef_all_v30_merged", 1.0, "robotwin_eef"),
    ],
    "robotwin_eef_30hz": [
        ("robotwin_eef_all_v30_merged_slow30fps", 1.0, "robotwin_eef"),
    ],
    "humanoid_merged_v30_robotwin_eef": [
        ("humanoid_merged_v30_robotwin_eef_state_t3", 1.0, "robotwin_eef"),
    ],
    "robocoin": [
        ("robocoin_agilex_robotwin_eef",1.0,"robotwin_eef"),
    ],
    "robotwin_task2": [
        ("place_a2b_left", 1.0, "robotwin_joint"),
        ("place_a2b_right", 1.0, "robotwin_joint"),
    ],

    "multi_robot": [
        ("LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        # ("OXE_LEROBOT_DATASET/bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
    ],
}
