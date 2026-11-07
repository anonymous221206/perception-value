# Follow-up E3: the CARLA environment 198 and 204 source (set CARLA_ENV to a copy of this file with your paths).
# CARLA 0.9.15 and Bench2Drive's leaderboard from carla_garage (PDM-Lite), with third_party/patches/carla_garage_e3.patch
# applied. The Python environment needs CARLA 0.9.15's client and this repository's detector requirements (E3 ran
# Python 3.8, torch 2.4.1, ultralytics 8.2.103, numpy 1.24.4, scikit-learn 1.3.2, scipy 1.10.1).
# source /path/to/your/carla/env/bin/activate
export CARLA_ROOT=/path/to/carla_0.9.15
export WORK_DIR=/path/to/carla_garage/Bench2Drive
export SCENARIO_RUNNER_ROOT=$WORK_DIR/scenario_runner LEADERBOARD_ROOT=$WORK_DIR/leaderboard
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla:$SCENARIO_RUNNER_ROOT:$LEADERBOARD_ROOT:$WORK_DIR/leaderboard/team_code:${PYTHONPATH:-}
export CHALLENGE_TRACK_CODENAME=SENSORS IS_BENCH2DRIVE=True PLANNER_TYPE=traj DEBUG_CHALLENGE=0 REPETITIONS=1 RESUME=True
