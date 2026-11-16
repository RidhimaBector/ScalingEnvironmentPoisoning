"""LunarLander with region-wise engine failure, as a poisonable victim environment.

Ported from DBB-EPA/src_lunarlander/envs/env_lunarlander_attack.py ("Spiking Pitch
Black", AAMAS 2022) to the gymnasium API and the VictimEnvironment interface. The
physics and the attack are unchanged apart from three mechanical edits: reset() and
step() return gymnasium-style tuples, the region index is clamped to the screen, and
rendering is dropped.

The attack: the screen is split into 10 vertical regions. Each region has a failure
level in [0, 5]; when the lander fires an engine inside a region, the thrust loses its
vertical component with probability level / 5. The attacker raises levels by up to 1
per region per attack step (it cannot lower them, as in the original).

The lander physics derives from the LunarLander environment of OpenAI Gym
(MIT License, Copyright (c) 2016 OpenAI).
"""

import math
import numpy as np

# Box2D classes by their b2-prefixed names. Standard builds export them from the
# top-level package; the Alliance (Compute Canada) wheel only from Box2D.Box2D and
# ships no Box2D.b2 alias module, which is what gym's own lunar lander imports.
try:
    from Box2D import b2World
    import Box2D as _b2
except ImportError:
    import Box2D.Box2D as _b2
b2World = _b2.b2World
edgeShape, circleShape, fixtureDef = _b2.b2EdgeShape, _b2.b2CircleShape, _b2.b2FixtureDef
polygonShape, revoluteJointDef, contactListener = _b2.b2PolygonShape, _b2.b2RevoluteJointDef, _b2.b2ContactListener

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

from core.victim_environment import VictimEnvironment

FPS = 50
SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0   # Set 1500 to make game harder

LANDER_POLY =[
    (-14, +17), (-17, 0), (-17 ,-10),
    (+17, -10), (+17, 0), (+14, +17)
    ]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400

CHUNKS = 11


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class _LanderPhysics(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    continuous = False

    def __init__(self, initial_random: float = INITIAL_RANDOM):
        # Size of the random kick the lander gets at the start of an episode
        # (1000 = original; larger is harder).
        self.initial_random = float(initial_random)
        self.seed()
        self.viewer = None

        self.world = b2World()
        self.moon = None
        self.lander = None
        self.particles = []

        self.prev_reward = None

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)
            
        ''' attack action space '''
        self.attack_space = spaces.Box(0, +1, (CHUNKS-1,), dtype=np.float32)
        ''' attack hyper-parameter condition '''
        self.p_attack = np.zeros((CHUNKS-1, 2))
        for i in range(CHUNKS-1):
            self.p_attack[i][0] = 1.0
        self.hyper_condition = np.zeros(CHUNKS-1)
        self.hyper_min = 0
        self.hyper_max = 5

        self.reset()
        
    ''' attack function '''
    def attack(self, attack_actions):
        # aggregte attack_actions with previous effect
        self.hyper_condition = self.hyper_condition + attack_actions
        # in range [0,5]?
        # if > upper_bound: set hyper_condition = 5 & set corresponding attack =0
        for i in range(len(self.hyper_condition)):
            if self.hyper_condition[i] > self.hyper_max:
                self.hyper_condition[i] = 5
                attack_actions[i] = 0.0
            if self.hyper_condition[i] < self.hyper_min:
                self.hyper_condition[i] = 0
                attack_actions[i] = 0.0
        # compute probability of attack 
        for i in range(len(self.hyper_condition)):
            self.p_attack[i][1] = self.hyper_condition[i]/5 # prob of attack
            self.p_attack[i][0] = 1 - self.hyper_condition[i]/5 # prob of non-attack
            
        return attack_actions
        
    ''' attack-reset function '''
    def clear_attack(self):
        # self.p_attack -- probability of attack or not
        # self.p_attack[0] = not attack
        # self.p_attack[1] = attack
        self.p_attack = np.zeros((CHUNKS-1, 2))
        for i in range(CHUNKS-1):
            self.p_attack[i][0] = 1.0
            
        # self.hyper_condition -- cumulative attack_force -- \sum(attack_actions)    
        self.hyper_condition = np.zeros(CHUNKS-1)
        
    # Generate seed
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
#         ''' reset attack settings '''
# #         self.reset_attack()
        
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        # terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H/2, size=(CHUNKS+1,))
        chunk_x = [W/(CHUNKS-1)*i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS//2-1]
        self.helipad_x2 = chunk_x[CHUNKS//2+1]
        self.helipad_y = H/4
        height[CHUNKS//2-2] = self.helipad_y
        height[CHUNKS//2-1] = self.helipad_y
        height[CHUNKS//2+0] = self.helipad_y
        height[CHUNKS//2+1] = self.helipad_y
        height[CHUNKS//2+2] = self.helipad_y
        smooth_y = [0.33*(height[i-1] + height[i+0] + height[i+1]) for i in range(CHUNKS)]

        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(CHUNKS-1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i+1], smooth_y[i+1])
            self.moon.CreateEdgeFixture(
                vertices=[p1,p2],
                density=0,
                friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        initial_y = VIEWPORT_H/SCALE
        self.lander = self.world.CreateDynamicBody(
            position=(VIEWPORT_W/SCALE/2, initial_y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[(x/SCALE, y/SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,   # collide only with ground
                restitution=0.0)  # 0.99 bouncy
                )
        self.lander.color1 = (0.5, 0.4, 0.9)
        self.lander.color2 = (0.3, 0.3, 0.5)
        self.lander.ApplyForceToCenter( (
            self.np_random.uniform(-self.initial_random, self.initial_random),
            self.np_random.uniform(-self.initial_random, self.initial_random)
            ), True)

        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(VIEWPORT_W/SCALE/2 - i*LEG_AWAY/SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W/SCALE, LEG_H/SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )
            leg.ground_contact = False
            leg.color1 = (0.5, 0.4, 0.9)
            leg.color2 = (0.3, 0.3, 0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY/SCALE, LEG_DOWN/SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
                )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5  # The most esoteric numbers here, angled legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        return self.step(np.array([0, 0]) if self.continuous else 0)[0], {}

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position = (x, y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=circleShape(radius=2/SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
                )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def step(self, action):
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # Engines
        tip  = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action == 2):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0,1.0) + 1.0)*0.5   # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0
                
            ''' attact implement '''
            # location?
            lander_x_pos = self.lander.position[0]
            lander_x_index = int(np.clip(lander_x_pos // 2, 0, CHUNKS - 2))  # clamp: lander can leave the screen mid-step
            # attack_prob?
            prob_attack = self.p_attack[lander_x_index]
            # attack
            bool_attack = np.random.choice(np.arange(len(prob_attack)), p=prob_attack)
            if bool_attack == 1: # bool_attack=1 : implement attack, stay still
                ox = tip[0] * (4/SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]  # 4 is move a bit downwards, +-2 for randomness
                oy = -tip[1] * dispersion[0] # no obvious move, just move a random tiny distance
            else: # bool_attack=0 : no attack, move as usual
                ox = tip[0] * (4/SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]  # 4 is move a bit downwards, +-2 for randomness
                oy = -tip[1] * (4/SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            ''' ............... '''
            
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(3.5,  # 3.5 is here to make particle speed adequate
                                      impulse_pos[0],
                                      impulse_pos[1],
                                      m_power)  # particles are just a decoration
            p.ApplyLinearImpulse((ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
                                 impulse_pos,
                                 True)
            self.lander.ApplyLinearImpulse((-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                                           impulse_pos,
                                           True)

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1, 3]):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                direction = action-2
                s_power = 1.0
                
            ''' attact implement '''
            # location?
            lander_x_pos = self.lander.position[0]
            lander_x_index = int(np.clip(lander_x_pos // 2, 0, CHUNKS - 2))  # clamp: lander can leave the screen mid-step
            # attack_prob?
            prob_attack = self.p_attack[lander_x_index]
            # attack
            bool_attack = np.random.choice(np.arange(len(prob_attack)), p=prob_attack)
            if bool_attack == 1: # bool_attack=1 : implement attack, stay still
                ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY/SCALE)
                oy = -tip[1] * dispersion[0]/10
            else: # bool_attack=0 : no attack, move as usual
                ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY/SCALE)
                oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY/SCALE)
            ''' ............... '''
            
            impulse_pos = (self.lander.position[0] + ox - tip[0] * 17/SCALE,
                           self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT/SCALE)
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse((ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
                                 impulse_pos
                                 , True)
            self.lander.ApplyLinearImpulse((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                                           impulse_pos,
                                           True)

        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
            vel.x*(VIEWPORT_W/SCALE/2)/FPS,
            vel.y*(VIEWPORT_H/SCALE/2)/FPS,
            self.lander.angle,
            20.0*self.lander.angularVelocity/FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0
            ]
        assert len(state) == 8

        reward = 0
        shaping = \
            - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
            - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
            - 100*abs(state[4]) + 10*state[6] + 10*state[7]  # And ten points for legs contact, the idea is if you
                                                             # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power*0.30  # less fuel spent is better, about -30 for heuristic landing
        reward -= s_power*0.03

        done = False
        if self.game_over or abs(state[0]) >= 1.0:
            done = True
            reward = -100
        if not self.lander.awake:
            done = True
            reward = +100
        return np.array(state, dtype=np.float32), reward, done, False, {}


class LunarLanderAttack(_LanderPhysics, VictimEnvironment):
    """The lander behind the VictimEnvironment interface used by the attack pipeline."""

    nA = 4                 # no-op, left engine, main engine, right engine
    max_steps = 1000       # per victim episode (gym's own time limit for LunarLander)

    def step(self, action):
        return super().step(int(action))

    # --- VictimEnvironment interface ---
    @property
    def perturbation_space(self) -> spaces.Space:
        return self.attack_space

    def apply_perturbation(self, perturbation: np.ndarray) -> None:
        self._prev_dynamics = self.get_dynamics().copy()
        self.attack(np.asarray(perturbation, dtype=np.float64).flatten().copy())

    def get_dynamics(self) -> np.ndarray:
        return np.asarray(self.hyper_condition, dtype=np.float64).copy()

    def reset_dynamics(self) -> None:
        self.clear_attack()

    @property
    def dynamics_dim(self) -> int:
        return CHUNKS - 1

    def dynamics_bounds(self):
        return (float(self.hyper_min), float(self.hyper_max))

    def target_trajectories(self):
        """The behaviour the attacker wants: a recorded landing, as a list of
        (prev_state, prev_action, state, action) tuples (the original lander_target_t1)."""
        import os
        data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "targets", "lander_target_t1.npz"))
        s, a = data["states"], data["actions"]
        return [(s[i - 1], int(a[i - 1]), s[i], int(a[i])) for i in range(1, len(s))]
