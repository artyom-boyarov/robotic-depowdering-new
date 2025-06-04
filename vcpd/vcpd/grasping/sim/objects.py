import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from vcpd.vcpd.grasping.sim.utils import step_simulation
from scipy.spatial.transform import Rotation
import pybullet as p
import numpy as np
import os
from vcpd.vcpd.grasping.grasp_evaluation import GraspGeometry


class RigidObject(object):
    def __init__(self, obj_name, **kwargs):
        """
        Construct a rigid object for pybullet.
        :param obj_name: a string of object name.
        :param vis_params: parameters of p.createVisualShape.
        :param col_params: parameters of p.createCollisionShape.
        :param body_params: parameters of p.createMultiBody.
        """
        self.obj_name = obj_name
        keys = kwargs.keys()
        if 'vis_params' in keys and 'col_params' in keys and 'body_params' in keys:
            vis_params, col_params, body_params = kwargs['vis_params'], kwargs['col_params'], kwargs['body_params']
            self.body_params = body_params
            
            self.obj_id = p.createMultiBody(baseCollisionShapeIndex=p.createCollisionShape(**col_params),
                                            baseVisualShapeIndex=p.createVisualShape(**vis_params),
                                            **body_params)
        elif 'fileName' in keys:
            self.obj_id = p.loadURDF(kwargs['fileName'],
                                     basePosition=kwargs['basePosition'], baseOrientation=kwargs['baseOrientation'])
            p.changeDynamics(self.obj_id, linkIndex=-1, mass=kwargs['mass'])
        else:
            raise ValueError('Invalid arguments for RigidObject initialization.')
        p.changeVisualShape(self.obj_id, -1, rgbaColor=kwargs['color'])

    def change_dynamics(self, *args, **kwargs):
        p.changeDynamics(self.obj_id, -1, *args, **kwargs)

    def change_color(self, rgb=None, a=1.0):
        if rgb is None:
            p.changeVisualShape(self.obj_id, -1, rgbaColor=np.random.uniform(size=3).tolist()+[a])
        else:
            p.changeVisualShape(self.obj_id, -1, rgbaColor=np.asarray(rgb).tolist() + [a])

    def get_pose(self):
        position, quaternion = p.getBasePositionAndOrientation(self.obj_id)
        return np.asarray(position), np.asarray(quaternion)

    def set_pose(self, position, quaternion):
        
        # originalPosition, originalQuaternion = self.get_pose()
        # position_scaled = [0.001 * pos for pos in position]
        # newPosition = position_scaled + originalPosition
        # originalQuaternion = Rotation.from_quat(originalQuaternion)
        # newQuaternion = originalQuaternion * Rotation.from_quat(quaternion)
        p.resetBasePositionAndOrientation(self.obj_id,
                                          position,
                                          quaternion)

    def wait_for_stable_condition(self, threshold=0.001, num_step=30):
        stable = False
        while not stable:
            if self.is_stable(threshold, num_step):
                stable = True

    def is_stable(self, threshold=0.001, num_steps=30):
        pose1 = np.concatenate(self.get_pose(), axis=0)
        step_simulation(num_steps)
        pose2 = np.concatenate(self.get_pose(), axis=0)
        fluctuation = np.sqrt(np.sum((np.asanyarray(pose1) - np.asanyarray(pose2)) ** 2))
        # print('fluctuation: {}'.format(fluctuation))
        return fluctuation < threshold

    @property
    def transform(self):
        pos, quat = self.get_pose()
        pose = np.eye(4, dtype=np.float32)
        pose[0:3, 0:3] = Rotation.from_quat(quat).as_matrix()
        pose[0:3, 3] = pos
        return pose


BASE_OFFSET = np.array([0,0,0])
BASE_ROTATION = Rotation.from_euler('xyz', [0, 0, 1.5707963267948966], degrees=False)
OUTER_BAR_LEFT_OFFSET = np.array([0, -0.0825, -0.0325])
OUTER_BAR_LEFT_ROTATION = Rotation.from_euler('xyz', [0, 0, 1.5707963267948966], degrees=False)
OUTER_BAR_RIGHT_OFFSET = np.array([0, -0.0825, 0.0325])
OUTER_BAR_RIGHT_ROTATION = Rotation.from_euler('xyz', [0, 0, 1.5707963267948966], degrees=False)
INNER_BAR_LEFT_OFFSET = np.array([0, -0.0995, -0.0165])
INNER_BAR_LEFT_ROTATION = Rotation.from_euler('xyz', [0, 0, 1.5707963267948966], degrees=False).as_quat()
INNER_BAR_RIGHT_OFFSET = np.array([0, -0.0995, 0.0165])
INNER_BAR_RIGHT_ROTATION = Rotation.from_euler('xyz', [0, 0, 1.5707963267948966], degrees=False).as_quat()
MOUNT_LEFT_OFFSET = np.array([0, -0.1405, -0.0325])
MOUNT_LEFT_ROTATION = Rotation.from_euler('xyz', [3.141592653589793, 0, 1.5707963267948966], degrees=False).as_quat()
MOUNT_RIGHT_OFFSET = np.array([0, -0.1405, 0.0325])
MOUNT_RIGHT_ROTATION = Rotation.from_euler('xyz', [0, 0, 1.5707963267948966], degrees=False).as_quat()
TIP_LEFT_OFFSET = np.array([0, -0.164, -0.0225])
TIP_LEFT_ROTATION = Rotation.from_euler('xyz', [0, -1.5707963267948966, 1.5707963267948966], degrees=False).as_quat()
TIP_RIGHT_OFFSET = np.array([0, -0.164, 0.0225])
TIP_RIGHT_ROTATION = Rotation.from_euler('xyz', [1.5707963267948966, 1.5707963267948966, 0], degrees=False).as_quat()
INNER_BAR_LENGTH = 0.058
OUTER_BAR_LENGTH = 0.058

class Rizon4sGripper(object):
    def __init__(self, asset_path):
        self.components = [
            'base', 
            'finger_tip_left', 
            'finger_tip_right',
            'finger_mount_left',
            'finger_mount_right',
            'inner_bar_left', 
            'inner_bar_right', 
            'outer_bar_left',
            'outer_bar_right'
        ]
        self.color=[0.7, 0.7, 0.7, 1.0]

        # TODO: Look at grav.urdf to find correct orientations and base positions for all of the parts.
        self.vertex_sets = dict()
        folder_name = 'grav'
        
        # Base
        vis_path = os.path.join(asset_path, folder_name, 'visual', 'base.obj')
        col_path = os.path.join(asset_path, folder_name, 'collision', 'base.obj')
        vis_params = {'shapeType': p.GEOM_MESH, 'fileName': vis_path, 'meshScale': [-.001] * 3}
        col_params = {'shapeType': p.GEOM_MESH, 'fileName': col_path, 'meshScale': [-.001] * 3}
        body_params = {
            'baseMass': 1.186, 
            'basePosition': BASE_OFFSET, 
            'baseOrientation': BASE_ROTATION.as_quat()
        }
        self.__setattr__('base', RigidObject('base',
                                                vis_params=vis_params,
                                                col_params=col_params,
                                                body_params=body_params,
                                                color=self.color))
        # Finger outer bars
        vis_path = os.path.join(asset_path, folder_name, 'visual', 'outer_bar.obj')
        col_path = os.path.join(asset_path, folder_name, 'collision', 'outer_bar.obj')
        vis_params = {'shapeType': p.GEOM_MESH, 'fileName': vis_path, 'meshScale': [-.001] * 3}
        col_params = {'shapeType': p.GEOM_MESH, 'fileName': col_path, 'meshScale': [-.001] * 3}
        body_params = {
            'baseMass': 0.0001, 
                            # x -z y 
            'basePosition': OUTER_BAR_LEFT_OFFSET, 
            'baseOrientation': OUTER_BAR_LEFT_ROTATION.as_quat()
        }
        self.__setattr__('outer_bar_left', RigidObject('outer_bar_left',
                                                vis_params=vis_params,
                                                col_params=col_params,
                                                body_params=body_params,
                                                color=self.color))
        vis_path = os.path.join(asset_path, folder_name, 'visual', 'outer_bar.obj')
        col_path = os.path.join(asset_path, folder_name, 'collision', 'outer_bar.obj')
        vis_params = {'shapeType': p.GEOM_MESH, 'fileName': vis_path, 'meshScale': [-.001] * 3}
        col_params = {'shapeType': p.GEOM_MESH, 'fileName': col_path, 'meshScale': [-.001] * 3}
        body_params = {
            'baseMass': 0.0001, 
            'basePosition': OUTER_BAR_RIGHT_OFFSET, 
            'baseOrientation': OUTER_BAR_RIGHT_ROTATION.as_quat()
        }
        self.__setattr__('outer_bar_right', RigidObject('outer_bar_right',
                                                vis_params=vis_params,
                                                col_params=col_params,
                                                body_params=body_params,
                                                color=self.color))
    
        # Finger inner bars
        vis_path = os.path.join(asset_path, folder_name, 'visual', 'inner_bar.obj')
        col_path = os.path.join(asset_path, folder_name, 'collision', 'inner_bar.obj')
        vis_params = {'shapeType': p.GEOM_MESH, 'fileName': vis_path, 'meshScale': [-.001] * 3}
        col_params = {'shapeType': p.GEOM_MESH, 'fileName': col_path, 'meshScale': [-.001] * 3}
        body_params = {
            'baseMass': 0.0001, 
            'basePosition': INNER_BAR_LEFT_OFFSET, 
            'baseOrientation': INNER_BAR_LEFT_ROTATION
        }
        self.__setattr__('inner_bar_left', RigidObject('inner_bar_left',
                                                vis_params=vis_params,
                                                col_params=col_params,
                                                body_params=body_params,
                                                color=self.color))
        vis_path = os.path.join(asset_path, folder_name, 'visual', 'inner_bar.obj')
        col_path = os.path.join(asset_path, folder_name, 'collision', 'inner_bar.obj')
        vis_params = {'shapeType': p.GEOM_MESH, 'fileName': vis_path, 'meshScale': [-.001] * 3}
        col_params = {'shapeType': p.GEOM_MESH, 'fileName': col_path, 'meshScale': [-.001] * 3}
        body_params = {
            'baseMass': 0.0001, 
            'basePosition': INNER_BAR_RIGHT_OFFSET, 
            'baseOrientation': INNER_BAR_RIGHT_ROTATION
        }
        self.__setattr__('inner_bar_right', RigidObject('inner_bar_left',
                                                vis_params=vis_params,
                                                col_params=col_params,
                                                body_params=body_params,
                                                color=self.color))
        
        # Finger mounts
        vis_path = os.path.join(asset_path, folder_name, 'visual', 'finger_mount.obj')
        col_path = os.path.join(asset_path, folder_name, 'collision', 'finger_mount.obj')
        vis_params = {'shapeType': p.GEOM_MESH, 'fileName': vis_path, 'meshScale': [-.001] * 3}
        col_params = {'shapeType': p.GEOM_MESH, 'fileName': col_path, 'meshScale': [-.001] * 3}
        body_params = {
            'baseMass': 0.0001, 
                            # base + left_outer_bar + left_finger_mount offset
            'basePosition': MOUNT_LEFT_OFFSET, 
            'baseOrientation': MOUNT_LEFT_ROTATION
        }
        self.__setattr__('finger_mount_left', RigidObject('finger_mount_left',
                                                vis_params=vis_params,
                                                col_params=col_params,
                                                body_params=body_params,
                                                color=self.color))
        vis_path = os.path.join(asset_path, folder_name, 'visual', 'finger_mount.obj')
        col_path = os.path.join(asset_path, folder_name, 'collision', 'finger_mount.obj')
        vis_params = {'shapeType': p.GEOM_MESH, 'fileName': vis_path, 'meshScale': [-.001] * 3}
        col_params = {'shapeType': p.GEOM_MESH, 'fileName': col_path, 'meshScale': [-.001] * 3}
        body_params = {
            'baseMass': 0.0001, 
            'basePosition': MOUNT_RIGHT_OFFSET, 
            'baseOrientation': MOUNT_RIGHT_ROTATION
        }
        self.__setattr__('finger_mount_right', RigidObject('finger_mount_right',
                                                vis_params=vis_params,
                                                col_params=col_params,
                                                body_params=body_params,
                                                color=self.color))
    

        # Finger tips
        vis_path = os.path.join(asset_path, folder_name, 'visual', 'finger_tip.obj')
        col_path = os.path.join(asset_path, folder_name, 'collision', 'finger_tip.obj')
        vis_params = {'shapeType': p.GEOM_MESH, 'fileName': vis_path, 'meshScale': [.001] * 3}
        col_params = {'shapeType': p.GEOM_MESH, 'fileName': col_path, 'meshScale': [.001] * 3}
        body_params = {
            'baseMass': 0.0001, 
            'basePosition': TIP_LEFT_OFFSET, 
            'baseOrientation': TIP_LEFT_ROTATION
        }
        self.__setattr__('finger_tip_left', RigidObject('finger_tip_left',
                                                vis_params=vis_params,
                                                col_params=col_params,
                                                body_params=body_params,
                                                color=self.color))
        vis_path = os.path.join(asset_path, folder_name, 'visual', 'finger_tip.obj')
        col_path = os.path.join(asset_path, folder_name, 'collision', 'finger_tip.obj')
        vis_params = {'shapeType': p.GEOM_MESH, 'fileName': vis_path, 'meshScale': [.001] * 3}
        col_params = {'shapeType': p.GEOM_MESH, 'fileName': col_path, 'meshScale': [.001] * 3}
        body_params = {
            'baseMass': 0.0001, 
            'basePosition': TIP_RIGHT_OFFSET, 
            'baseOrientation': TIP_RIGHT_ROTATION
        }
        self.__setattr__('finger_tip_right', RigidObject('finger_tip_right',
                                                vis_params=vis_params,
                                                col_params=col_params,
                                                body_params=body_params,
                                                color=self.color))


        
        self._max_width = 0.1
        self._curr_width = 0.0

    def set_pose(self, position, quaternion):
        # old_position = position
        # position = [pos * 0.001 for pos in old_position]
        # print(position)
        # self.base.set_pose(np.array(position) + np.array(BASE_OFFSET), (BASE_ROTATION * Rotation.from_quat(quaternion)).as_quat())
        # base_offset, base_quat = self.base.get_pose()
        # print("Base is at ", base_offset, " with pose ", base_quat)
        # print("Left outer bar has original offset", OUTER_BAR_LEFT_OFFSET, " but now ", Rotation.from_quat(base_quat).apply(np.array(OUTER_BAR_LEFT_OFFSET)))

        # self.outer_bar_left.set_pose(
        #     np.array(base_offset) + (Rotation.from_quat(quaternion)).apply(np.array(OUTER_BAR_LEFT_OFFSET)),
        #     (OUTER_BAR_LEFT_ROTATION * Rotation.from_quat(quaternion)).as_quat()
        # )
        # self.outer_bar_right.set_pose(
        #     np.array(base_offset) + (Rotation.from_quat(quaternion).apply(np.array(OUTER_BAR_RIGHT_OFFSET))), 
        #     (OUTER_BAR_RIGHT_ROTATION * Rotation.from_quat(quaternion)).as_quat()
        # )
        


        for com in self.components:
            self.__getattribute__(com).set_pose(
                Rotation.from_quat(quaternion).apply(self.__getattribute__(com).body_params['basePosition']) + np.array(position), 
                self.__getattribute__(com).body_params['baseOrientation'] * quaternion
            )
        curr_width = self._curr_width
        self._curr_width = self._max_width
        self.set_gripper_width(curr_width)

    def get_pose(self):
        return self.__getattribute__(self.components[0]).get_pose()

    def set_gripper_width(self, width):
        # if width > self._max_width:
        #     print('warning! the maximal width is 0.08 for panda gripper, set to 0.08 instead')
        #     width = self._max_width
        width = min(0.08, max(0.0, width))
        # left_delta = np.eye(4)
        # left_delta[1, 3] = (width - self._curr_width) / 2
        # left_pose = self.finger_tip_left.transform @ left_delta
        # self.finger_tip_left.set_pose(left_pose[0:3, 3], Rotation.from_matrix(left_pose[0:3, 0:3]).as_quat())
        # right_delta = np.eye(4)
        # right_delta[1, 3] = -(width - self._curr_width) / 2
        # right_pose = self.finger_tip_right.transform @ right_delta
        # self.finger_tip_right.set_pose(right_pose[0:3, 3], Rotation.from_matrix(right_pose[0:3, 0:3]).as_quat())
        self._curr_width = width

    def place_fingers(self, width, grasp_pt, x_dir, y_dir, z_dir, orientation) -> GraspGeometry:
        # width += 0.019 # 0.0095 * 2
        # TODO: Left finger might be up or down depending on z_dir.
        # Cross z and y to get x.
        grasp = GraspGeometry()
        grasp.found = False
        if width > self._max_width: return grasp
        try:
            y_dir = (1/np.linalg.norm(y_dir)) * y_dir
            z_dir = (1/np.linalg.norm(z_dir)) * z_dir
            x_dir = (1/np.linalg.norm(x_dir)) * x_dir
            grasp_pt_left = grasp_pt + ((width + 0.0075) / 2) * y_dir
            grasp_pt_right = grasp_pt - ((width + 0.0075) / 2) * y_dir

            # 0.01 is distance from finger tip face to origin in finger tip cad model.
            # z = axis of approach. Fingers parallel to this axis.
            # y = axis perpendicular to faces of fingers.
            finger_left_base_pt = grasp_pt_left - (0.038 * z_dir) + (0.01 * y_dir)
            finger_right_base_pt = grasp_pt_right - (0.038 * z_dir) - (0.01 * y_dir)

            finger_left_rot = Rotation.from_euler("yxz", [0, 0, -90], degrees=True)
            finger_right_rot = Rotation.from_euler("yxz", [0, 0, 90], degrees=True)

            finger_mount_left_base_pt = finger_left_base_pt - (0.0235 * z_dir) + (0.01 * y_dir)
            finger_mount_right_base_pt = finger_right_base_pt - (0.0235 * z_dir) - (0.01 * y_dir)

            finger_mount_left_rot = Rotation.from_euler("zyx", [90, 0, -90], degrees=True)
            finger_mount_right_rot = Rotation.from_euler("zyx", [-90, 0, 90], degrees=True)

            if np.abs((0.01 - width) / OUTER_BAR_LENGTH) > 1: return grasp

            # TODO: Check this. Doesn't seem like we even use theta.
            finger_outer_bar_left_theta = np.arcsin((0.01 - (width/ 2)) / OUTER_BAR_LENGTH)
            finger_outer_bar_right_theta = -finger_outer_bar_left_theta
            finger_outer_bar_y_displacement = 0.01 - (width/2)
            finger_outer_bar_z_displacement = np.sqrt((0.058**2) - finger_outer_bar_y_displacement**2)

            finger_outer_bar_left_base_pt = finger_mount_left_base_pt - (finger_outer_bar_z_displacement * z_dir) + (finger_outer_bar_y_displacement * y_dir)
            finger_outer_bar_right_base_pt = finger_mount_right_base_pt - (finger_outer_bar_z_displacement * z_dir) - (finger_outer_bar_y_displacement * y_dir)

            finger_outer_bar_left_rot = Rotation.from_euler("zyx", [90, 0, -90 + np.rad2deg(finger_outer_bar_left_theta)], degrees=True)
            finger_outer_bar_right_rot = Rotation.from_euler("zyx", [-90, 0, 90 + np.rad2deg(finger_outer_bar_right_theta)], degrees=True)

            finger_inner_bar_left_base_pt = finger_outer_bar_left_base_pt + (0.017 * z_dir) - (0.016 * y_dir)
            finger_inner_bar_right_base_pt = finger_outer_bar_right_base_pt + (0.017 * z_dir) + (0.016 * y_dir)
            finger_inner_bar_left_rot = finger_outer_bar_left_rot
            finger_inner_bar_right_rot = finger_outer_bar_right_rot

            base_pt = finger_outer_bar_right_base_pt  + (0.0325 * y_dir) - (0.0825 * z_dir)
            base_rot = Rotation.from_euler("zyx", [90, 0, -90], degrees=True)
            # print("Finger bar left angle:", finger_inner_bar_left_theta, " right angle:", finger_inner_bar_right_theta)

            # print("Gripper base point:", base_pt)
            # print("Grasp point center:", grasp_pt)
            # print("Gripper width:", width)

            self.finger_tip_left.set_pose(
                finger_left_base_pt, 
                (Rotation.from_quat(orientation) * finger_left_rot).as_quat()
            )
            self.finger_tip_right.set_pose(
                finger_right_base_pt, 
                (Rotation.from_quat(orientation) * finger_right_rot).as_quat()
            )
            self.finger_mount_left.set_pose(
                finger_mount_left_base_pt, 
                (Rotation.from_quat(orientation) * finger_mount_left_rot).as_quat()
            )
            self.finger_mount_right.set_pose(
                finger_mount_right_base_pt, 
                (Rotation.from_quat(orientation) * finger_mount_right_rot).as_quat()
            )
            self.outer_bar_left.set_pose(
                finger_outer_bar_left_base_pt, 
                (Rotation.from_quat(orientation) * finger_outer_bar_left_rot).as_quat()
            )
            self.outer_bar_right.set_pose(
                finger_outer_bar_right_base_pt, 
                (Rotation.from_quat(orientation) * finger_outer_bar_right_rot).as_quat()
            )
            self.inner_bar_left.set_pose(
                finger_inner_bar_left_base_pt, 
                (Rotation.from_quat(orientation) * finger_inner_bar_left_rot).as_quat()
            )
            self.inner_bar_right.set_pose(
                finger_inner_bar_right_base_pt, 
                (Rotation.from_quat(orientation) * finger_inner_bar_right_rot).as_quat()
            )
            self.base.set_pose(
                base_pt, 
                (Rotation.from_quat(orientation) * base_rot).as_quat()
            )

            grasp = GraspGeometry()
            grasp.base_pos = base_pt
            grasp.contact1pos = grasp_pt_left
            grasp.contact1rot = (Rotation.from_quat(orientation) * finger_left_rot).as_matrix()
            grasp.contact2pos = grasp_pt_right
            grasp.contact2rot = (Rotation.from_quat(orientation) * finger_right_rot).as_matrix()
            grasp.found = True
            return grasp
        except:
            return grasp
        # x = input()

    def get_gripper_width(self):
        return self._curr_width

    def reset_gripper(self):
        self.set_pose([0, 0, 0], [0, 0, 0, 1])
        self.set_gripper_width(self._max_width)

    def is_collided(self, exemption, threshold=0.0, show_col=False):
        gripper_ids = [self.__getattribute__(com).obj_id for com in self.components]
        for obj_id in range(p.getNumBodies()):
            if obj_id not in gripper_ids and obj_id not in exemption:
                for gripper_id in gripper_ids:
                    contacts = p.getClosestPoints(gripper_id, obj_id, threshold)
                    if len(contacts) != 0:
                        p.changeVisualShape(obj_id, -1, rgbaColor=[1, 0, 0, 1]) if show_col else None
                        return True
        return False

    def change_color(self, rgb=None, a=1.0):
        if rgb is None:
            rgba = np.random.uniform(size=3).tolist()+[a]
        else:
            rgba = np.asarray(rgb).tolist()+[a]
        [p.changeVisualShape(self.__getattribute__(com).obj_id, -1, rgbaColor=rgba) for com in self.components]

    def remove_gripper(self):
        [p.removeBody(self.__getattribute__(com).obj_id) for com in self.components]
