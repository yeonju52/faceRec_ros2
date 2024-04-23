from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():

    #
    # ARGS
    #
    model = LaunchConfiguration("fr_weight")
    model_cmd = DeclareLaunchArgument(
        "fr_weight",
        default_value="ir_50",
        description="model name or path")

    device = LaunchConfiguration("device")
    device_cmd = DeclareLaunchArgument(
        "device",
        default_value="cuda:0",
        description="Device to use (GPU/CPU)")
    
    option = LaunchConfiguration("option")
    option_cmd = DeclareLaunchArgument(
        "option",
        default_value="1",
        description="0: 임베딩 저장 1: webcam/video으로 run_video)"
    )

    thresh = LaunchConfiguration("thresh")
    thresh_cmd = DeclareLaunchArgument(
        "thresh",
        default_value="0.2",
        description="Minimum probability of a faces' similarity to be published")

    max_obj = LaunchConfiguration("max_obj")
    max_obj_cmd = DeclareLaunchArgument(
        "max_obj",
        default_value="6",
        description="Maximum counts of Face Detection")

    dataset = LaunchConfiguration("dataset")
    dataset_cmd = DeclareLaunchArgument(
        "dataset",
        default_value="face_dataset/test",
        description="face's dataset & face ID에 대한 경로"
    )

    video = LaunchConfiguration("video")
    video_cmd = DeclareLaunchArgument(
        "video",
        default_value="'0'",
        description="0: webcam 또는 \"video/iAM.mp4\" 특정 비디오 path 경로"
    )

    input_image_topic = LaunchConfiguration("input_image_topic")
    input_image_topic_cmd = DeclareLaunchArgument(
        "input_image_topic",
        default_value="/camera/camera/color/image_raw",
        description="Name of the input image topic")

    image_reliability = LaunchConfiguration("image_reliability")
    image_reliability_cmd = DeclareLaunchArgument(
        "image_reliability",
        default_value="2",
        choices=["0", "1", "2"],
        description="Specific reliability QoS of the input image topic (0=system default, 1=Reliable, 2=Best Effort)")

    namespace = LaunchConfiguration("namespace")
    namespace_cmd = DeclareLaunchArgument(
        "namespace",
        default_value="adaface",
        description="Namespace for the nodes")

    adaface_node_cmd = Node(
        package="yolov8_ros",
        executable="adaface_node",
        name="adaface_node",
        namespace=namespace,
        parameters=[{
            "fr_weight": model,
            "device": device,
            "option": option,
            "thresh": thresh,
            "max_obj": max_obj,
            "dataset": dataset,
            "video": video,
            "image_reliability": image_reliability,
        }],
        remappings=[
            ("image_raw", input_image_topic),
            ("detections", "tracking")
        ]
    )



    ld = LaunchDescription()

    ld.add_action(model_cmd)
    ld.add_action(device_cmd)
    ld.add_action(option_cmd)
    ld.add_action(thresh_cmd)
    ld.add_action(max_obj_cmd)
    ld.add_action(dataset_cmd)
    ld.add_action(video_cmd)
    ld.add_action(input_image_topic_cmd)
    ld.add_action(image_reliability_cmd)
    ld.add_action(namespace_cmd)

    ld.add_action(adaface_node_cmd)

    return ld
