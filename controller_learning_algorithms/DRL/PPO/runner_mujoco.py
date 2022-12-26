import math
import tempfile
import os
import concurrent.futures
from typing import List, Optional
import cv2
import numpy.typing as npt

import mujoco
import mujoco_viewer
import numpy as np
from config import (ACTION_CONSTRAINT, NUM_OBS_TIMES, NUM_OBSERVATIONS,
                    NUM_PARALLEL_AGENT, NUM_STEPS)
from interaction_buffer import Buffer
import torch

try:
    import logging

    old_len = len(logging.root.handlers)

    from dm_control import mjcf

    new_len = len(logging.root.handlers)

    assert (
        old_len + 1 == new_len
    ), "dm_control not adding logging handler as expected. Maybe they fixed their annoying behaviour? https://github.com/deepmind/dm_control/issues/314https://github.com/deepmind/dm_control/issues/314"

    logging.root.removeHandler(logging.root.handlers[-1])
except Exception as e:
    print("Failed to fix absl logging bug", e)
    pass

from pyrr import Quaternion, Vector3
from revolve2.core.physics.actor.urdf import to_urdf as physbot_to_urdf
from revolve2.core.physics.running import (
    ActorControl,
    ActorState,
    Batch,
    BatchResults,
    Environment,
    EnvironmentResults,
    EnvironmentState,
    RecordSettings,
    Runner,
)

class LocalRunner(Runner):
    """Runner for simulating using Mujoco."""

    _headless: bool
    _start_paused: bool
    _num_simulators: int

    def __init__(
        self,
        headless: bool = False,
        start_paused: bool = False,
        num_simulators: int = 1,
    ):
        """
        Initialize this object.

        :param headless: If True, the simulation will not be rendered. This drastically improves performance.
        :param start_paused: If True, start the simulation paused. Only possible when not in headless mode.
        :param num_simulators: The number of simulators to deploy in parallel. They will take one core each but will share space on the main python thread for calculating control.
        """
        assert (
            headless or num_simulators == 1
        ), "Cannot have parallel simulators when visualizing."

        assert not (
            headless and start_paused
        ), "Cannot start simulation paused in headless mode."

        self._headless = headless
        self._start_paused = start_paused
        self._num_simulators = num_simulators

    def _run_environment(
        cls,
        env_index: int,
        env_descr: Environment,
        headless: bool,
        record_settings: Optional[RecordSettings],
        start_paused: bool,
        control_step: float,
        sample_step: float,
        simulation_time: int,
        num_joints: int,
    ) -> EnvironmentResults:

        torch.manual_seed(env_index)
        
        logging.info(f"Environment {env_index}")
        model = mujoco.MjModel.from_xml_string(cls._make_mjcf(env_descr))

        # TODO initial dof state
        data = mujoco.MjData(model)

        initial_targets = [
            dof_state
            for posed_actor in env_descr.actors
            for dof_state in posed_actor.dof_states
        ]
        cls._set_dof_targets(data, initial_targets)

        for posed_actor in env_descr.actors:
            posed_actor.dof_states

        if not headless:
            viewer = mujoco_viewer.MujocoViewer(
                model,
                data,
            )
            viewer._render_every_frame = False  # Private but functionality is not exposed and for now it breaks nothing.
            viewer._paused = start_paused

        if record_settings is not None:
            video_step = 1 / record_settings.fps
            video_file_path = f"{record_settings.video_directory}/{env_index}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(
                video_file_path,
                fourcc,
                record_settings.fps,
                (viewer.viewport.width, viewer.viewport.height),
            )

            viewer._hide_menu = True

        last_control_time = 0.0
        last_sample_time = 0.0
        last_video_time = 0.0  # time at which last video frame was saved

        results = EnvironmentResults([])

        # sample initial state
        results.environment_states.append(
            EnvironmentState(0.0, cls._get_actor_states(env_descr, data, model))
        )

        old_position = results.environment_states[0].actor_states[0].position
        new_observation = [[] for _ in range(NUM_OBSERVATIONS)]
        pos_sliding = np.zeros(NUM_OBS_TIMES*num_joints)
        timestep = 0

        while (time := data.time) < simulation_time and timestep <= NUM_STEPS:
            # do control if it is time
            if timestep == 0  or time >= last_control_time + control_step:
                last_control_time = math.floor(time / control_step) * control_step
                control_user = ActorControl()
                
                results.environment_states.append(
                    EnvironmentState(
                        time, cls._get_actor_states(env_descr, data, model)
                    )
                )

                # get hinges current position and head orientation
                hinges_pos = np.array(data.qpos[-num_joints:])
                orientation = np.array(cls._get_actor_state(0, data, model).orientation)
                pos_sliding = np.concatenate((hinges_pos, pos_sliding.squeeze()[:num_joints*(NUM_OBS_TIMES - 1)]))
                #velocities = np.array(data.qvel[-num_joints:])
                
                new_observation[0] = np.array(pos_sliding, dtype=np.float32)
                new_observation[1] = np.array(orientation, dtype=np.float32)
                #new_observation[2] = np.array(velocities, dtype=np.float32)
                
                new_action, new_value, new_logp = env_descr.controller.control(control_step, control_user, new_observation)
                actor_targets = control_user._dof_targets
                actor_targets.sort(key=lambda t: t[0])
                targets = [
                    target
                    for actor_target in actor_targets
                    for target in actor_target[1]
                ]
                
                if timestep < NUM_STEPS:
                    cls._set_dof_targets(data, targets)

                action = new_action
                logp = new_logp
                value = new_value
                observation = new_observation.copy()
                timestep += 1                                          

            # step simulation
            mujoco.mj_step(model, data)

            # render if not headless. also render when recording and if it time for a new video frame.
            if not headless or (
                record_settings is not None and time >= last_video_time + video_step
            ):
                viewer.render()

            # capture video frame if it's time
            if record_settings is not None and time >= last_video_time + video_step:
                last_video_time = int(time / video_step) * video_step

                # https://github.com/deepmind/mujoco/issues/285 (see also record.cc)
                img: npt.NDArray[np.uint8] = np.empty(
                    (viewer.viewport.height, viewer.viewport.width, 3),
                    dtype=np.uint8,
                )

                mujoco.mjr_readPixels(
                    rgb=img,
                    depth=None,
                    viewport=viewer.viewport,
                    con=viewer.ctx,
                )
                img = np.flip(img, axis=0)  # img is upside down initially
                video.write(img)


        if not headless:
            viewer.close()

        if record_settings is not None:
            video.release()

        # sample one final time
        results.environment_states.append(
            EnvironmentState(time, cls._get_actor_states(env_descr, data, model))
        )

        timestep = 0

        return results

    async def run_batch(
        self, batch: Batch, controller, num_agents, record_settings: Optional[RecordSettings] = None
    ) -> BatchResults:
        """
        Run the provided batch by simulating each contained environment.

        :param batch: The batch to run.
        :param record_settings: Optional settings for recording the runnings. If None, no recording is made.
        :returns: List of simulation states in ascending order of time.
        """
        logging.info("Starting simulation batch with mujoco.")

        control_step = 1 / batch.control_frequency
        sample_step = 1 / batch.sampling_frequency

        num_joints =  len(batch.environments[0].actors[0].actor.joints)

        if record_settings is not None:
            os.makedirs(record_settings.video_directory, exist_ok=False)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self._num_simulators
        ) as executor:
            futures = [
                executor.submit(
                    self._run_environment,
                    env_index,
                    env_descr,
                    self._headless,
                    record_settings,
                    self._start_paused,
                    control_step,
                    sample_step,
                    batch.simulation_time,
                    num_joints
                )
                for env_index, env_descr in enumerate(batch.environments)
            ]
            results = BatchResults([future.result() for future in futures])

        logging.info("Finished batch.")

        return results

    def _calculate_velocity(self, state1, state2):
        """
        Calculate the velocity for all agents at a timestep
        """
        old_d = math.sqrt(state1.x**2 + state1.y**2)
        new_d = math.sqrt(state2.x**2 + state2.y**2)
        return new_d-old_d

    def _update_obs_stats(self, new_obs, step_num):
        """
        """
        for i, obs in enumerate(new_obs):
            delta = obs.mean() - self._obs_mean[i]
            self._obs_mean[i] += delta / (step_num + 1)
            delta_2 = obs.mean() - self._obs_mean[i]
            self._obs_m2[i] += delta * delta_2

    def _reset_mean_m2(self):
        self._obs_mean = [0,] * NUM_OBSERVATIONS
        self._obs_m2 = [0,] * NUM_OBSERVATIONS
        self._rew_mean = 0
        self._rew_m2 = 0

    def _normalize_observation(self, observation, step_num):
        if step_num < 2: 
            return observation
        else:
            for i, obs in enumerate(observation):
                obs -= self._obs_mean[i]
                obs /= self._obs_m2[i] / step_num
            return observation

    def _update_rew_stats(self, new_rew, step_num):
        """
        """
        delta = new_rew - self._rew_mean
        self._rew_mean += delta / (step_num + 1)
        delta_2 = new_rew - self._rew_mean
        self._rew_m2 += delta * delta_2

    def _normalize_reward(self, reward, step_num):
        if step_num < 2: 
            return reward
        else:
            reward /= self._rew_m2 / step_num
            return reward

    @staticmethod
    def _make_mjcf(env_descr: Environment) -> str:
        env_mjcf = mjcf.RootElement(model="environment")

        env_mjcf.compiler.angle = "radian"

        env_mjcf.option.timestep = 0.0005
        env_mjcf.option.integrator = "RK4"

        env_mjcf.option.gravity = [0, 0, -9.81]

        env_mjcf.worldbody.add(
            "light",
            pos=[0, 0, 100],
            ambient=[0.5, 0.5, 0.5],
            directional=True,
            castshadow=False,
        )
        env_mjcf.asset.add(
            "texture",
            name="grid",
            type="2d",
            builtin="checker",
            width="512",
            height="512",
            rgb1=".1 .2 .3",
            rgb2=".2 .3 .4",
        )
        env_mjcf.asset.add(
            "material",
            name="grid",
            texture="grid",
            texrepeat="1 1",
            texuniform="true",
            reflectance=".2"
        )
        env_mjcf.worldbody.add(
            "geom",
            name="ground",
            size=[10, 10, 1],
            type="plane",
            material="grid",
        )
        env_mjcf.visual.headlight.active = 0

        # add the following to solve the error "Pre-allocated contact buffer is full"
        env_mjcf.size.nconmax = 150

        for actor_index, posed_actor in enumerate(env_descr.actors):
            urdf = physbot_to_urdf(
                posed_actor.actor,
                f"robot_{actor_index}",
                Vector3(),
                Quaternion(),
            )

            model = mujoco.MjModel.from_xml_string(urdf)

            # mujoco can only save to a file, not directly to string,
            # so we create a temporary file.
            botfile = tempfile.NamedTemporaryFile(
                mode="r+", delete=False, suffix=".urdf"
            )
            mujoco.mj_saveLastXML(botfile.name, model)
            robot = mjcf.from_file(botfile)
            botfile.close()

            LocalRunner._set_parameters(robot)

            for joint in posed_actor.actor.joints:
                robot.actuator.add(
                    "position",
                    kp=1,
                    joint=robot.find(
                        namespace="joint",
                        identifier=joint.name,
                    ),
                )
                robot.actuator.add(
                    "velocity",
                    kv=0.05,
                    joint=robot.find(namespace="joint", identifier=joint.name),
                )

            attachment_frame = env_mjcf.attach(robot)
            attachment_frame.add("freejoint")
            attachment_frame.pos = [
                posed_actor.position.x,
                posed_actor.position.y,
                posed_actor.position.z,
            ]

            attachment_frame.quat = [
                posed_actor.orientation.x,
                posed_actor.orientation.y,
                posed_actor.orientation.z,
                posed_actor.orientation.w,
            ]

        xml = env_mjcf.to_xml_string()
        if not isinstance(xml, str):
            raise RuntimeError("Error generating mjcf xml.")

        return xml

    @classmethod
    def _get_actor_states(
        cls, env_descr: Environment, data: mujoco.MjData, model: mujoco.MjModel
    ) -> List[ActorState]:
        return [
            cls._get_actor_state(i, data, model) for i in range(len(env_descr.actors))
        ]

    @staticmethod
    def _get_actor_state(
        robot_index: int, data: mujoco.MjData, model: mujoco.MjModel
    ) -> ActorState:
        bodyid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"robot_{robot_index}/",  # the slash is added by dm_control. ugly but deal with it
        )
        assert bodyid >= 0

        qindex = model.body_jntadr[bodyid]

        # explicitly copy because the Vector3 and Quaternion classes don't copy the underlying structure
        position = Vector3([n for n in data.qpos[qindex : qindex + 3]])
        orientation = Quaternion([n for n in data.qpos[qindex + 3 : qindex + 3 + 4]])

        return ActorState(position, orientation)

    @staticmethod
    def _set_dof_targets(data: mujoco.MjData, targets: List[float]) -> None:
        if len(targets) * 2 != len(data.ctrl):
            raise RuntimeError("Need to set a target for every dof")
        for i, target in enumerate(targets):
            data.ctrl[2 * i] = target
            data.ctrl[2 * i + 1] = 0

    @staticmethod
    def _set_recursive_parameters(element):
        if element.tag == "body":
            for sub_element in element.body._elements:
                LocalRunner._set_recursive_parameters(sub_element)

        if element.tag == "geom":
            element.friction = [0.7, 0.1, 0.1]

    @staticmethod
    def _set_parameters(robot):
        for element in robot.worldbody.body._elements:
            LocalRunner._set_recursive_parameters(element)