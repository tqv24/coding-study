import pybullet as p
import pybullet_data
import time
import threading
import queue
import json
import anthropic

# ── 1. Physics & Robot Setup ──────────────────────────────────────────────────
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(1.2, 45, -30, [0, 0, 0.3])

p.loadURDF("plane.urdf")
# The xArm6 is placed at the origin
robot = p.loadURDF("xarm/xarm6_with_gripper.urdf", [0, 0, 0], useFixedBase=True)
# Cube is on the floor
sphere = p.loadURDF("sphere_small.urdf", [0.45, 0.1, 0.05], globalScaling=1.0)
p.changeDynamics(sphere, -1, lateralFriction=20.0, spinningFriction=10.0, rollingFriction=5.0, restitution=0.0)

# Add friction to all gripper finger links so they grip instead of slip
for i in range(p.getNumJoints(robot)):
    p.changeDynamics(robot, i, lateralFriction=5.0)

# Map movable joints (Revolute joints 0-5 for xArm6)
MOVABLE_JOINTS = []
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    if info[2] <= 1: 
        MOVABLE_JOINTS.append(i)

EEF_LINK = 6
GRIPPER_DRIVE = 8
EEF_DOWN = p.getQuaternionFromEuler([0, 3.14159, 0])

# Set arm color to grey
GREY = [0.5, 0.5, 0.5, 1.0]
for i in range(-1, p.getNumJoints(robot)):
    p.changeVisualShape(robot, i, rgbaColor=GREY)

# ── 2. The Intelligence Layer ────────────────────────────────────────────────
client = anthropic.Anthropic() # Reads key from environment

TRANSIT_Z = 0.3 # Safe height to move across the floor
DESCEND_RADIUS = 0.06 

def get_system_prompt(sphere_pos):
    return f"""You are a motion planner for an xArm6 robot on a floor.
Current Sphere Position: {sphere_pos}
Transit Height: z={TRANSIT_Z}

RULES:
1. Start every plan by rising to z={TRANSIT_Z}.
2. Travel to the sphere's X,Y while staying at z={TRANSIT_Z}.
3. Only then descend to z=0.04 to grab.
4. Close gripper (1.0), then lift back to z={TRANSIT_Z}.

Output ONLY raw JSON with no markdown, no code fences, no explanation:
{{
  "message": "Status description",
  "steps": [
    {{"x": float, "y": float, "z": float, "gripper": 0.0-1.0, "duration": seconds}}
  ]
}}"""

command_queue = queue.Queue()
action_queue = queue.Queue()

def ai_worker():
    while True:
        user_input = command_queue.get()
        if user_input is None: break
        
        sphere_pos, _ = p.getBasePositionAndOrientation(sphere)
        print(f"\n[AI] Thinking about: '{user_input}'...")
        
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=get_system_prompt(list(sphere_pos)),
                messages=[{"role": "user", "content": user_input}]
            )
            raw = response.content[0].text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            print(f"[AI] Plan: {data['message']}")
            action_queue.put(data["steps"])
        except Exception as e:
            print(f"[AI Error]: {e}")

threading.Thread(target=ai_worker, daemon=True).start()

# ── 3. Input Reader Thread ───────────────────────────────────────────────────
def input_thread():
    while True:
        cmd = input("\nEnter Command: ")
        if cmd.strip():
            command_queue.put(cmd)

threading.Thread(target=input_thread, daemon=True).start()

# ── 4. Main Simulation Loop ──────────────────────────────────────────────────
active_seq = []
step_idx = 0
step_timer = 0
DT = 1.0/240.0

prev_target = [0.2, 0.0, TRANSIT_Z]
current_target = [0.2, 0.0, TRANSIT_Z]
current_gripper = 0.0
prev_gripper = 0.0

print("\n--- ROBOT READY ---")
print("Commands: 'pick up the sphere', 'go home', 'move right 20cm'")

while True:
    # Logic: Handle the sequence of steps from AI
    if not active_seq or step_idx >= len(active_seq):
        if not action_queue.empty():
            active_seq = action_queue.get()
            step_idx, step_timer = 0, 0

    if active_seq and step_idx < len(active_seq):
        step = active_seq[step_idx]
        current_target = [step["x"], step["y"], step["z"]]
        current_gripper = step["gripper"]

        step_timer += DT
        duration = max(step["duration"], DT)
        t = min(step_timer / duration, 1.0)  # 0.0 → 1.0 over the step

        # Smooth lerp from previous waypoint to current target
        interp_target = [
            prev_target[0] + (current_target[0] - prev_target[0]) * t,
            prev_target[1] + (current_target[1] - prev_target[1]) * t,
            prev_target[2] + (current_target[2] - prev_target[2]) * t,
        ]
        interp_gripper = prev_gripper + (current_gripper - prev_gripper) * t

        if step_timer >= duration:
            prev_target = current_target[:]
            prev_gripper = current_gripper
            step_idx += 1
            step_timer = 0
    else:
        interp_target = current_target[:]
        interp_gripper = current_gripper

    # Hard Safety Guard: Never go below the floor
    final_z = max(interp_target[2], 0.01)
    final_target = [interp_target[0], interp_target[1], final_z]

    # Calculate IK and move joints
    ik_angles = p.calculateInverseKinematics(robot, EEF_LINK, final_target, EEF_DOWN)
    for i, joint_idx in enumerate(MOVABLE_JOINTS[:6]): # Limit to the 6 arm joints
        p.setJointMotorControl2(robot, joint_idx, p.POSITION_CONTROL, ik_angles[i], maxVelocity=2.0)

    # Gripper Logic
    p.setJointMotorControl2(robot, GRIPPER_DRIVE, p.POSITION_CONTROL, interp_gripper * 0.85, force=200)

    p.stepSimulation()
    time.sleep(DT)