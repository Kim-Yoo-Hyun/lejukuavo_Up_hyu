#!/bin/bash
# Jetson (leju_kuavo) - camera bag
#   usage: ./record_camera.sh [episode_name]
#   기본 저장: /media/data/kuavo_dataset/raw/  (NVMe 891G. eMMC 루트는 10G뿐이라 절대 쓰지 말 것)
set -u

EP="${1:-episode_$(date +%Y%m%d_%H%M%S)}"
OUTDIR="${KUAVO_DATASET_DIR:-/media/data/kuavo_dataset/raw}"
MIN_FREE_GB=20

TOPICS=(
  # --- 카메라 ---
  /camera/color/image_raw/compressed
  /camera/color/camera_info
  /left_wrist_camera/color/image_raw/compressed
  /left_wrist_camera/color/camera_info
  /right_wrist_camera/color/image_raw/compressed
  /right_wrist_camera/color/camera_info
  # --- 로봇 실측 상태 (observation) ---
  /sensors_data_raw
  /dexhand/state
  /leju_claw_state # 우리는 Leju claw를 안 씀
  /humanoid_ee_State
  # --- 로봇 명령 (action) ---
  /joint_cmd                        # 모터로 나간 최종 저수준 명령 (전신)
  /kuavo_arm_traj
  /control_robot_hand_position
  /dexhand/command
  /leju_claw_command
  # --- VR 입력 ---
  /leju_quest_bone_poses
  /quest_joystick_data
  /quest3/triger_arm_mode
  /cmd_torso_pose_vr
  /vr_whole_torso_ctrl
  /robot_head_motion_data
  # --- IK ---
  # /ik/two_arm_hand_pose_cmd
  # /ik/result
  # /ik/result_free
  /drake_ik/eef_pose
  # --- 휠형 전환 대비 (이족에서는 발행자 없음, 그냥 건너뜀) ---
  # /mobile_manipulator_eef_poses # wheel 형을 위한 topic이므로 현재는 주석 처리
  # --- TF ---
  /tf
  /tf_static


)
HZ_CHECK=(
  /camera/color/image_raw/compressed
  /left_wrist_camera/color/image_raw/compressed
  /right_wrist_camera/color/image_raw/compressed
)

# ROS setup 이 ROS_MASTER_URI 를 localhost 로 덮어쓰므로 먼저 붙잡아 둔다
_PRE_MASTER="${ROS_MASTER_URI:-}"
_PRE_IP="${ROS_IP:-}"
set +u   # ROS setup 스크립트가 미정의 변수를 참조함
source /opt/ros/noetic/setup.bash
[ -f "$HOME/kuavo_ros_application/devel/setup.bash" ] && source "$HOME/kuavo_ros_application/devel/setup.bash"
export ROS_MASTER_URI="${_PRE_MASTER:-http://kuavo_master:11311}"
export ROS_IP="${_PRE_IP:-192.168.26.12}"
set -u

echo "=============================================="
echo " CAMERA BAG (Jetson)   episode: $EP"
echo "=============================================="

# --- 1. 마스터 연결 ---
if ! timeout 5 rostopic list >/dev/null 2>&1; then
  echo "[!] ROS 마스터에 연결 불가 ($ROS_MASTER_URI). 중단."; exit 1
fi

# --- 2. 디스크 ---
mkdir -p "$OUTDIR" || { echo "[!] $OUTDIR 생성 실패. 중단."; exit 1; }
FREE_GB=$(df -BG --output=avail "$OUTDIR" | tail -1 | tr -dc '0-9')
DEV=$(df --output=source "$OUTDIR" | tail -1)
echo "[i] 저장 위치: $OUTDIR  ($DEV, ${FREE_GB}G 여유)"
if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
  echo "[!] 여유 공간 ${FREE_GB}G < ${MIN_FREE_GB}G. 중단."; exit 1
fi
case "$DEV" in
  *mmcblk*) echo "[!] 경고: eMMC 루트에 쓰려 하고 있습니다. /media/data 를 쓰세요."; ;;
esac

# --- 3. 토픽이 실제로 흐르는지 ---
echo "[i] 카메라 토픽 확인 중..."
FAIL=0
for t in "${HZ_CHECK[@]}"; do
  rate=$(timeout 2 rostopic hz "$t" 2>&1 | grep -m1 -oP 'average rate: \K[0-9.]+')
  if [ -z "$rate" ]; then
    echo "    $t : 데이터 없음  <-- 문제"; FAIL=1
  else
    printf "    %-50s %s Hz\n" "$t" "$rate"
  fi
done
if [ "$FAIL" = "1" ]; then
  read -r -p "[?] 일부 카메라 토픽이 비어 있습니다. 그래도 녹화할까요? [y/N] " a
  [ "$a" = "y" ] || [ "$a" = "Y" ] || { echo "중단."; exit 1; }
fi

# --- 4. 녹화 ---
BAG="$OUTDIR/${EP}_camera"
echo
echo "[i] 녹화 시작. 종료는 Ctrl+C 를 '한 번만' 누르세요 (강제 종료 시 .active 로 남습니다)"
echo
trap 'echo; echo "[i] Ctrl+C 감지 - bag 마무리 중..."' INT
rosbag record --buffsize=2048 -O "$BAG" "${TOPICS[@]}"
trap - INT

# --- 5. 마무리 검증 ---
sleep 1
if [ -f "${BAG}.bag.active" ]; then
  echo "[!] .active 파일 발견 - reindex 시도"
  rosbag reindex "${BAG}.bag.active" && mv "${BAG}.bag.active" "${BAG}.bag"
fi
echo
echo "=============================================="
if [ -f "${BAG}.bag" ]; then
  rosbag info "${BAG}.bag"
  echo "----------------------------------------------"
  # 메시지 0개인 토픽 경고
  for t in "${TOPICS[@]}"; do
    rosbag info "${BAG}.bag" 2>/dev/null | grep -q " ${t} " || echo "[!] 비어 있음: $t"
  done
  echo "[i] 저장 완료: ${BAG}.bag"
else
  echo "[!] bag 파일이 생성되지 않았습니다: ${BAG}.bag"
fi

