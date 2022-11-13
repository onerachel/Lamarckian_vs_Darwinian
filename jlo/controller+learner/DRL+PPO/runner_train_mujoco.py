import math
import tempfile
from typing import List

import mujoco
import mujoco_viewer
import numpy as np
from config import (ACTION_CONSTRAINT, NUM_OBS_TIMES, NUM_OBSERVATIONS,
                    NUM_PARALLEL_AGENT, NUM_STEPS)
from interaction_buffer import Buffer

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
from revolve2.actor_controller import ActorController
from revolve2.core.physics.actor.urdf import to_urdf as physbot_to_urdf
from revolve2.core.physics.running import (ActorControl, ActorState, Batch,
                                           BatchResults, Environment,
                                           EnvironmentResults,
                                           EnvironmentState, Runner)


class LocalRunnerTrain(Runner):
    """Runner for simulating using Mujoco."""

    _headless: bool
    _controller: ActorController
    _num_agents: int
    _obs_mean: List[float]
    _obs_m2: List[float]
    _rew_mean: float
    _rew_m2: float


    def __init__(self, headless: bool = False):
        """
        Initialize this object.

        :param headless: If True, the simulation will not be rendered. This drastically improves performance.
        """
        self._headless = headless

    async def run_batch(self, batch: Batch, controller: ActorController, num_agents: int) -> BatchResults:
        """
        Run the provided batch by simulating each contained environment.

        :param batch: The batch to run.
        :returns: List of simulation states in ascending order of time.
        """
        logging.info("Starting simulation batch with mujoco.")
        self._controller = controller
        self._num_agents = num_agents
        self._obs_mean = [0,] * NUM_OBSERVATIONS
        self._obs_m2 = [0,] * NUM_OBSERVATIONS
        self._rew_mean = 0
        self._rew_m2 = 0

        control_step = 1 / batch.control_frequency

        results = BatchResults([EnvironmentResults([]) for _ in batch.environments])

        num_joints =  len(batch.environments[0].actors[0].actor.joints)
        obs_dims = (num_joints*NUM_OBS_TIMES, 4,)
        buffer = Buffer(obs_dims, num_joints, self._num_agents)
        sum_rewards = np.zeros((NUM_STEPS, NUM_PARALLEL_AGENT))
        sum_values = np.zeros((NUM_STEPS, NUM_PARALLEL_AGENT))

        for env_index, env_descr in enumerate(batch.environments):
            logging.info(f"Environment {env_index}")

            model = mujoco.MjModel.from_xml_string(self._make_mjcf(env_descr))

            # TODO initial dof state
            data = mujoco.MjData(model)

            model.jnt_stiffness = [1.0] * (num_joints + 1)
            model.dof_damping = [0.05] * len(data.qvel)

            initial_targets = [
                dof_state
                for posed_actor in env_descr.actors
                for dof_state in posed_actor.dof_states
            ]

            self._set_dof_targets(data, initial_targets)

            for posed_actor in env_descr.actors:
                posed_actor.dof_states

            if not self._headless:
                viewer = mujoco_viewer.MujocoViewer(
                    model,
                    data,
                )

            last_control_time = 0.0

            # sample initial state
            results.environment_results[env_index].environment_states.append(
                EnvironmentState(0.0, self._get_actor_states(env_descr, data, model))
            )

            old_position = results.environment_results[env_index].environment_states[0].actor_states[0].position
            new_observation = [[] for _ in range(NUM_OBSERVATIONS)]
            pos_sliding = np.zeros(NUM_OBS_TIMES*num_joints)
            buffer.reset_step_count()
            timestep = 0

            while (time := data.time) < batch.simulation_time and timestep <= NUM_STEPS:
                # do control if it is time
                if timestep == 0  or time >= last_control_time + control_step:
                    last_control_time = math.floor(time / control_step) * control_step
                    control = ActorControl()
                    
                    results.environment_results[env_index].environment_states.append(
                        EnvironmentState(
                            time, self._get_actor_states(env_descr, data, model)
                        )
                    )

                    # get hinges current position and head orientation
                    hinges_pos = np.array(data.qpos[-num_joints:])
                    orientation = np.array(self._get_actor_state(0, data, model).orientation)
                    pos_sliding = np.concatenate((hinges_pos, pos_sliding.squeeze()[:num_joints*(NUM_OBS_TIMES - 1)]))
                    #velocities = np.array(data.qvel[-num_joints:])
                    
                    new_observation[0] = np.array(pos_sliding, dtype=np.float32)
                    new_observation[1] = np.array(orientation, dtype=np.float32)
                    #new_observation[2] = np.array(velocities, dtype=np.float32)
                    
                    #self._update_obs_stats(new_observation, timestep)
                    #new_observation = self._normalize_observation(new_observation, timestep)
                    
                    new_action, new_value, new_logp = batch.control(env_index, control_step, control, new_observation)
                    actor_targets = control._dof_targets
                    actor_targets.sort(key=lambda t: t[0])
                    targets = [
                        target
                        for actor_target in actor_targets
                        for target in actor_target[1]
                    ]
                    
                    if timestep < NUM_STEPS:
                        self._set_dof_targets(data, targets)

                    if timestep > 0:
                        # get the new positions of each agent
                        new_position = results.environment_results[env_index].environment_states[-1].actor_states[0].position
                        
                        # compute the rewards from the new and old positions of the agents
                        reward = self._calculate_velocity(old_position, new_position)
                        #self._update_rew_stats(reward, timestep)
                        #reward = self._normalize_reward(reward, timestep)           

                        # insert data of the current state in the replay buffer
                        buffer.insert_single(
                                        idx=env_index,
                                        obs=observation,
                                        act=action,
                                        logp=logp,
                                        val=value,
                                        rew=reward
                                        )

                        sum_rewards[timestep-1, env_index] = reward
                        sum_values[timestep-1, env_index] = value
                        old_position = new_position.copy()

                    if timestep >= NUM_STEPS:
                        buffer.set_single_last_value(idx=env_index, last_value=value)

                    action = new_action
                    logp = new_logp
                    value = new_value
                    observation = new_observation.copy()
                    timestep += 1                                          

                # step simulation
                mujoco.mj_step(model, data)

                if not self._headless:
                    viewer.render()

            if not self._headless:
                viewer.close()

            # sample one final time
            results.environment_results[env_index].environment_states.append(
                EnvironmentState(time, self._get_actor_states(env_descr, data, model))
            )

        # do training    
        logging.info(f"Average cumulative reward after {NUM_STEPS} steps: {np.mean(np.sum(sum_rewards, axis=0))}")
        logging.info(f"Average state value: {np.mean(np.mean(sum_values, axis=0))}")
        sum_rewards = np.zeros((NUM_STEPS, NUM_PARALLEL_AGENT))
        sum_values = np.zeros((NUM_STEPS, NUM_PARALLEL_AGENT))

        self._controller.train(buffer)

        timestep = 0
        buffer = Buffer(obs_dims, num_joints, self._num_agents)  

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
            "geom",
            name="ground",
            type="plane",
            size=[10, 10, 1],
            rgba=[0.2, 0.2, 0.2, 1],
        )
        env_mjcf.worldbody.add(
            "light",
            pos=[0, 0, 100],
            ambient=[0.5, 0.5, 0.5],
            directional=True,
            castshadow=False,
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

            for joint in posed_actor.actor.joints:
                robot.actuator.add(
                    "position",
                    kp=5.0,
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
