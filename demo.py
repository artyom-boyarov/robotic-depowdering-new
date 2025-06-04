__author__ = "Art Boyarov"

import time
import flexivrdk
import spdlog
import threading
import numpy as np

EXT_FORCE_THRESHOLD = 0.5
EXT_TORQUE_THRESHOLD = 0.5
ROBOT_SN = "Rizon 4s-WxfLRm" # TODO: Change
GRIPPER_NAME = "Grav" # TODO: Change
logger = spdlog.ConsoleLogger("demo")

## TODO: Change to now set global variables for grasp pre-pose and grasp pose.

def check_robot_collisions(robot, logger, stop_event):
    while not stop_event.is_set():
        ext_force = np.array(
            robot.states().ext_wrench_in_world[0:3]
        )
        if np.linalg.norm(ext_force) > EXT_FORCE_THRESHOLD:
            logger.error("Robot is in collision, stopping robot")
            robot.Stop()

def main():
    logger.info("Starting fragile part pick-and-place demo")
    mode = flexivrdk.Mode

    
    try:

        robot = flexivrdk.Robot(ROBOT_SN)
        if robot.fault():
            logger.warn("Fault detected on robot, attempting to clear")
            if not robot.ClearFault():
                logger.error("Failed to clear fault, exiting...")
                return
            
            logger.info("Fault cleared, proceeding with demo")
        
        logger.info("Enabling robot")
        robot.Enable()
        while not robot.operational():
            time.sleep(1)
        
        robot.SwitchMode(mode.NRT_PLAN_EXECUTION)
        logger.info("Robot is operational, proceeding with demo")
        
        logger.info("Updating grasp pose variables")


        grasp_pose = np.array([
            0.746,
            -0.0016,
            0.144,
            0.0,
            135.0,
            0.0
        ])
        grasp_pre_pose = np.array([
            0.746,
            -0.0016,
            0.244,
            0.0,
            135.0,
            0.0
        ])
         # TODO: Get approach direction to do grasp bite pose
        robot.SetGlobalVariables({
            "GRASP_WIDTH": 20.0,
            "GRASP_FORCE": 20.0,
            "GRASP_POSE" : flexivrdk.Coord( # Note: has to be in metres.
                grasp_pose[0:3],
                grasp_pose[3:6],
                ["WORLD", "WORLD_ORIGIN"]
            ),
            "GRASP_PRE_POSE" : flexivrdk.Coord( # Note: has to be in metres.
                grasp_pre_pose[0:3],
                grasp_pre_pose[3:6],
                ["WORLD", "WORLD_ORIGIN"]
            )
        })

        plan_list = robot.plan_list()
        logger.info(plan_list)
        assert "MSEC2025_demo" in plan_list

        robot.ExecutePlan("MSEC2025_demo", True)

        while robot.busy():
            plan_info = robot.plan_info()
            logger.info(f"Currently executing {plan_info.pt_name}")

        logger.info("Finished pick and place demo")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return