"""CRANE+ V2 ＋ 天井カメラ ＋ 赤い立方体 のGazebo環境を、軽量に起動する.

rt-net/crane_plus の crane_plus_with_table.launch.py をもとに、
GPUのないDocker環境（CPU描画）でも動くよう次の点を変えている。
  - MoveIt と RViz は起動しない（データ収集・推論には不要で、描画が重い）
  - Gazebo は既定で画面なし（gui:=true で画面あり）
  - カメラ画像を小さくする（既定 224x168）
  - カメラの描画エンジンを選べる（ogre2 / ogre）

使い方:
  ros2 launch pai_reach_sim sim.launch.py
  ros2 launch pai_reach_sim sim.launch.py gui:=true viewer:=true
"""

import os
import re
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.actions import SetParameter


def _build_robot_description(width, height, rate, engine):
    """xacro を展開し、カメラ設定だけを書き換えた URDF 文字列を返す."""
    xacro_path = os.path.join(
        get_package_share_directory('crane_plus_description'),
        'urdf', 'crane_plus.urdf.xacro')
    urdf = subprocess.check_output([
        'xacro', xacro_path,
        'use_gazebo:=true',
        'use_ignition:=true',
        'use_camera:=true',
        'gz_control_config_package:=crane_plus_control',
        'gz_control_config_file_path:=config/crane_plus_controllers.yaml',
    ], text=True)

    def sub_once(pattern, repl, text, label):
        new_text, n = re.subn(pattern, repl, text, flags=re.S)
        if n == 0:
            raise RuntimeError(f'URDF内に {label} が見つかりません。crane_plus の版を確認してください。')
        return new_text

    # 元の内部パラメータ（640x480用）は画像サイズと合わなくなるので削除し、
    # 視野角 horizontal_fov から自動計算させる。
    urdf = sub_once(r'<lens>.*?</lens>', '', urdf, '<lens>')
    urdf = sub_once(r'<width>\s*640\s*</width>', f'<width>{width}</width>', urdf, '<width>')
    urdf = sub_once(r'<height>\s*480\s*</height>', f'<height>{height}</height>', urdf, '<height>')
    urdf = sub_once(r'<update_rate>\s*10\s*</update_rate>',
                    f'<update_rate>{rate}</update_rate>', urdf, '<update_rate>')
    urdf = sub_once(r'<render_engine>\s*ogre2\s*</render_engine>',
                    f'<render_engine>{engine}</render_engine>', urdf, '<render_engine>')
    return urdf


def _launch_setup(context, *args, **kwargs):
    gui = LaunchConfiguration('gui').perform(context).lower() == 'true'
    viewer = LaunchConfiguration('viewer').perform(context).lower() == 'true'
    world = LaunchConfiguration('world').perform(context)
    width = LaunchConfiguration('image_width').perform(context)
    height = LaunchConfiguration('image_height').perform(context)
    rate = LaunchConfiguration('camera_rate').perform(context)
    engine = LaunchConfiguration('render_engine').perform(context)

    robot_description = _build_robot_description(width, height, rate, engine)

    # STLメッシュや models/ を Gazebo に見つけさせるためのパス
    env = {
        # Gazebo の通信をコンテナ内（127.0.0.1）に固定する。PC のスリープや
        # ネットワーク変更で「Host unreachable」となり、サービスが応答しなくなるのを防ぐ。
        'IGN_IP': os.environ.get('IGN_IP', '127.0.0.1'),
        'IGN_GAZEBO_SYSTEM_PLUGIN_PATH': os.environ.get('LD_LIBRARY_PATH', ''),
        'IGN_GAZEBO_RESOURCE_PATH': ':'.join([
            os.path.dirname(get_package_share_directory('crane_plus_description')),
            os.path.join(get_package_share_directory('crane_plus_gazebo'), 'models'),
        ]),
    }

    # -r: 起動と同時にシミュレーションを開始（しないとコントローラが起動しない）
    gz_cmd = ['ign', 'gazebo', '-r', world]
    if gui:
        gz_cmd += ['--gui-config', os.path.join(
            get_package_share_directory('crane_plus_gazebo'), 'gui', 'gui.config')]
    else:
        gz_cmd += ['-s']  # サーバーのみ（画面なし）
    gazebo = ExecuteProcess(cmd=gz_cmd, output='screen', additional_env=env)

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[{'robot_description': robot_description}],
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', '/robot_description',
                   '-name', 'crane_plus',
                   '-z', '1.015',
                   '-allow_renaming', 'true'],
    )

    spawners = [
        ExecuteProcess(
            cmd=[f'ros2 run controller_manager spawner {name}'],
            shell=True, output='screen')
        for name in ['joint_state_broadcaster',
                     'crane_plus_arm_controller',
                     'crane_plus_gripper_controller']
    ]

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
                   'image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
                   'camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo'],
        output='screen',
    )

    actions = [gazebo, robot_state_publisher, spawn_robot, *spawners, bridge]
    if viewer:
        actions.append(Node(
            package='rqt_image_view', executable='rqt_image_view',
            arguments=['/image_raw'], output='log'))
    return actions


def generate_launch_description():
    default_world = os.path.join(
        get_package_share_directory('pai_reach_sim'), 'worlds', 'reach_table.sdf')
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='false',
                              description='Gazebo の画面を表示する'),
        DeclareLaunchArgument('viewer', default_value='false',
                              description='rqt_image_view でカメラ映像を表示する'),
        DeclareLaunchArgument('world', default_value=default_world,
                              description='ワールドファイル(.sdf)'),
        DeclareLaunchArgument('image_width', default_value='224'),
        DeclareLaunchArgument('image_height', default_value='168'),
        DeclareLaunchArgument('camera_rate', default_value='10',
                              description='カメラの撮影周期 [Hz, シミュレーション時間]'),
        DeclareLaunchArgument('render_engine', default_value='ogre2',
                              description='ogre2 または ogre（CPU描画では ogre が軽い場合あり）'),
        SetParameter(name='use_sim_time', value=True),
        OpaqueFunction(function=_launch_setup),
    ])
